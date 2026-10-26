"""Entity task contracts and real multi-body contact/reset integration."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from motion_data import save_motion
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.mdp import MDP
from supermimic.rotations import mat_to_6d, so3_exp, yaw_mat
from supermimic.rl.entity_state import EntityDynamics
from supermimic.tasks.entities.assets import ObjectCfg, entity_asset
from supermimic.tasks.entities.env_cfgs import EntityEnvCfg, make_env_cfg
from supermimic.tasks.entities.mdp import EntityMDP
from supermimic.tasks.entities.motion import EntityMotionLib


@pytest.fixture
def entity_heads(cpu_kin, tmp_path):
    t = np.arange(6) / 25
    angle = .4 + .7 * t
    path = tmp_path / "clip.npz"
    save_motion(path, dict(fps=25., root_pos=np.tile([0., 0., .8], (6, 1)),
        root_quat=np.tile([1., 0, 0, 0], (6, 1)), joint_names=G1_JOINT_NAMES,
        joint_pos=np.tile(default_joint_pos_list(), (6, 1)),
        object_pos=np.array([1., .3, .5]) + t[:, None] * [.4, -.2, .1],
        object_quat=np.stack((np.cos(angle / 2), t * 0, t * 0, np.sin(angle / 2)), -1),
        object_scale=np.ones(3)))
    extras = (ObjectCfg(name="box"), ObjectCfg(name="wall", dynamic=False, pos=(2., 0., .5)))
    motion = EntityMotionLib([path], cpu_kin, "cpu", max_objects=4, extra_objects=extras)
    mdp = EntityMDP(cpu_kin, motion, torch.tensor(default_joint_pos_list()), max_objects=4)
    dynamics = EntityDynamics(4)
    xi = dynamics.empty_context(2)
    context = dynamics.object_context(xi)
    context[..., 0], context[..., 10], context[..., 11] = 10., motion.valid_mask, motion.dynamic_mask
    context[..., 12] = torch.tensor([1, 2, 3, 0])
    return motion, mdp, dynamics, xi


def test_raw_actor_features_exclude_reference_and_privileged_physics(entity_heads):
    motion, mdp, dynamics, xi = entity_heads
    f, action = torch.tensor([1, 3]), torch.zeros(2, 29)
    state = motion.full_ref_state(f)
    obs = mdp.obs(state, action, xi, f)
    assert obs.shape == (2, 93 + 13 * 4)
    assert mdp.obs_noise.shape == (145,) and mdp.obs_noise[93:].count_nonzero() == 0
    obj = obs[:, 93:].reshape(2, 4, 13)
    torch.testing.assert_close(obj[:, 0, :3], state[6][:, 0] - state[0])
    assert obj[:, 3].count_nonzero() == 0
    assert obj[:, :3, 9].eq(1).all() and obj[:, 0, 11].eq(1).all()
    assert obj[:, 1:, 11].count_nonzero() == 0
    dynamics.object_context(xi)[..., :10] += 100
    motion.object_p.add_(20)
    torch.testing.assert_close(obs, mdp.obs(state, action, xi, f + 100000))


def test_entity_reward_preserves_robot_terms_and_ignores_nontarget_objects(entity_heads):
    motion, mdp, _, xi = entity_heads
    f, action = torch.tensor([2, 4]), torch.zeros(2, 29)
    state = motion.full_ref_state(f)
    baseline = MDP(mdp.kin, motion, mdp.default_q)
    old, old_terms = baseline.reward(state[:6], f, action, action, terms=True)
    reward, new_terms = mdp.reward(state, f, action, action, terms=True)
    torch.testing.assert_close(reward - old, torch.full((2,), .02))
    for key in old_terms:
        torch.testing.assert_close(new_terms[key], old_terms[key])
    torch.testing.assert_close(mdp.termination(state, f), baseline.termination(state[:6], f))
    torch.testing.assert_close(mdp.soft_termination(state, f), baseline.soft_termination(state[:6], f))
    displacement = torch.zeros_like(state[6]); displacement[:, 1:] = 50
    modified = (*state[:6], state[6] + displacement, *state[7:])
    torch.testing.assert_close(reward, mdp.reward(modified, f, action, action))
    displacement[:, 0, 0] = .1
    modified = (*state[:6], state[6] + displacement, *state[7:])
    torch.testing.assert_close(mdp.reward(modified, f, action, action) - old,
                               torch.full((2,), .02 * np.exp(-1)), atol=1e-7, rtol=1e-5)


def test_critic_exposes_future_reference_and_masks_empty_and_nontarget_slots(entity_heads):
    motion, mdp, dynamics, xi = entity_heads
    f = torch.tensor([8, 10])
    state = list(motion.full_ref_state(f))
    state[0] += torch.tensor([.3, -.2, 0.])
    state[1] = so3_exp(torch.tensor([[0., 0., .6], [0., 0., -.2]]))
    features = mdp.critic_features(tuple(state), f, xi)
    assert features.shape == (2, 184 + 4 * 38 + 3 * 6**2 + 4)
    torch.testing.assert_close(features[:, -4:], dynamics.ground_features(tuple(state), xi))
    objects = features[:, 184:184 + 4 * 38].reshape(2, 4, 38)
    assert objects[:, 3].count_nonzero() == 0
    assert objects[:, 1:, 26:36].count_nonzero() == 0
    goal = motion.object_goal(tuple(state), f, 10).reshape(2, 4, 10, 9)
    assert goal[:, 1:].count_nonzero() == 0
    future = motion.future_frames(f, 10)
    heading = yaw_mat(state[1]).transpose(-1, -2)
    expected = (heading[:, None] @ (motion.object_p[future, 0] - state[0][:, None]).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(goal[:, 0, :, :3], expected)
    torch.testing.assert_close(goal[:, 0, :, 3:], mat_to_6d(heading[:, None] @ motion.object_R[future, 0]))
    torch.testing.assert_close(goal[1, 0], goal[1, 0, :1].expand(10, -1))
    first = torch.tensor([1])
    actual = motion.full_ref_state(first)
    before = mdp.critic_features(actual, first, xi[:1])
    before_goal = motion.object_goal(actual, first, 10)
    motion.object_p[2:, 0] += .2
    torch.testing.assert_close(before, mdp.critic_features(actual, first, xi[:1]))
    assert not torch.allclose(before_goal, motion.object_goal(actual, first, 10))


def test_entity_motion_clock_static_references_and_reward_gradients(entity_heads):
    motion, mdp, _, xi = entity_heads
    assert motion.clip_len.tolist() == [11]
    torch.testing.assert_close(motion.object_v[:, 0], torch.tensor([.4, -.2, .1]).expand(11, -1), atol=1e-5, rtol=0)
    torch.testing.assert_close(motion.object_w[:, 0], torch.tensor([0., 0., .7]).expand(11, -1), atol=2e-5, rtol=0)
    assert motion.object_v[:, 1:].count_nonzero() == 0
    assert motion.dynamic_mask.tolist() == [1, 1, 0, 0]
    f = torch.tensor([1, 2]); state = motion.full_ref_state(f)
    position = (state[6] + .02).detach().requires_grad_()
    angle = torch.full_like(state[6], .05, requires_grad=True)
    state = (*state[:6], position, state[7] @ so3_exp(angle), *state[8:])
    action = torch.zeros(2, 29)
    mdp.reward(state, f, action, action).sum().backward()
    assert torch.isfinite(position.grad).all() and (position.grad[:, 0] < 0).all()
    assert position.grad[:, 1:].count_nonzero() == 0
    assert torch.isfinite(angle.grad).all() and (angle.grad[:, 0] < 0).all()


def test_lazy_factory_cli_rebinding_and_primitive_geometry(monkeypatch, tmp_path):
    import mjlab
    import tyro
    from supermimic.tasks.interaction import assets
    monkeypatch.setattr(assets, "load_motion", lambda *_: pytest.fail("Factory must not read motion data"))
    cfg = make_env_cfg(extra_objects=(ObjectCfg(name="box"),))
    parsed = tyro.cli(EntityEnvCfg, default=cfg, config=mjlab.TYRO_FLAGS,
        args=["--commands.motion.motion-dir", "/tmp/new-motion-dir",
              "--commands.motion.extra-objects.0.pos", "(2,1,.2)"])
    assert parsed.scene.entities["chair"].spec_fn.args[0] is parsed.commands["motion"]
    assert parsed.scene.entities["box"].spec_fn.args[0] is parsed.commands["motion"]
    assert parsed.commands["motion"].motion_dir == "/tmp/new-motion-dir"
    assert parsed.scene.entities["box"].init_state.pos == (2, 1, .2)
    scene_file = tmp_path / "objects.json"
    scene_file.write_text('[{"name":"box","pos":[2,1,0.2]}, {"name":"wall","dynamic":false}]')
    from_file = tyro.cli(EntityEnvCfg, default=make_env_cfg(), config=mjlab.TYRO_FLAGS,
        args=["--commands.motion.extra-objects-file", str(scene_file)])
    assert set(from_file.scene.entities) == {"robot", "chair", "box", "wall"}
    assert from_file.scene.entities["box"].init_state.pos == (2, 1, .2)
    assert from_file.scene.entities["wall"].spec_fn.args[0] is from_file.commands["motion"]
    no_target = replace(cfg, commands={"motion": replace(cfg.commands["motion"], object_enabled=False)})
    assert set(no_target.scene.entities) == {"robot", "box"}
    empty = make_env_cfg(object_enabled=False)
    assert set(empty.scene.entities) == {"robot"} and empty.sim.nconmax == 128
    asset = entity_asset(no_target.commands["motion"], 0)
    assert asset.points.shape == (1024, 6)
    np.testing.assert_allclose(np.linalg.norm(asset.points[:, 3:], axis=-1), 1.)
    np.testing.assert_allclose(asset.extent, [.3, .3, .3])
    np.testing.assert_allclose(asset.inertia, np.eye(3) * .015)
    fixed = make_env_cfg(object_enabled=False, extra_objects=(ObjectCfg(name="wall", dynamic=False),))
    model = fixed.scene.entities["wall"].spec_fn().compile()
    assert model.nq == 0 and model.nmocap == 1
    assert model.body_mass[1] == 10
    monkeypatch.setenv("SUPERMIMIC_EXTRA_OBJECTS_FILE", str(scene_file))
    monkeypatch.setenv("SUPERMIMIC_MAX_OBJECTS", "5")
    play_cfg = make_env_cfg(play=True)
    assert play_cfg.commands["motion"].max_objects == 5
    assert set(play_cfg.scene.entities) == {"robot", "chair", "box", "wall"}
    assert make_env_cfg(max_objects=6).commands["motion"].max_objects == 6


@pytest.fixture(scope="module")
def multi_entity_env(cuda_device):
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.assets import DATA_DIR
    from motion_data import load_motion
    from motion_data.collision import DEFAULT_CACHE_DIR, collision_key
    path = DATA_DIR / "motions/sitting/sitting__acadian_rustic_wooden_dining_chair_distressed_white__000.npz"
    if not path.is_file() or not (DEFAULT_CACHE_DIR / f"{collision_key(load_motion(path))}.npz").is_file():
        pytest.skip("Real chair motion and prepared collision cache required")
    cfg = make_env_cfg(max_objects=4, extra_objects=(
        ObjectCfg(name="box", pos=(1., 0., .2), size=(.2, .2, .2)),
        ObjectCfg(name="wall", pos=(2., 0., .5), size=(.1, .5, .5), dynamic=False, friction=.6),
    ))
    cfg.scene.num_envs, cfg.auto_reset = 2, False
    cfg.commands["motion"].motion_dir = str(path)
    cfg.commands["motion"].reset_noise = False
    env = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        env.reset()
        yield env
    finally:
        env.close()


def reset_frames(env, frames=(3, 137), ids=None):
    from tools._smoke import reset_frames as reset
    return reset(env, torch.tensor(frames, device=env.device), ids)


def test_native_entities_reset_physics_context_and_partial_reset(multi_entity_env):
    env = multi_entity_env
    cmd = env.command_manager.get_term("motion")
    obs, _ = reset_frames(env)
    assert cmd.object_entity_names == ("chair", "box", "wall")
    assert obs["policy_history"].shape == (2, 10 * (93 + 4 * 13))
    assert obs["state"].shape == (2, 71 + 4 * 13)
    reference = cmd.motion.full_ref_state(cmd.time_steps)
    for actual, expected in zip(cmd.state(), reference):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    objects = cmd.dynamics.object_context(cmd.xi)
    assert objects[..., 10].tolist() == [[1, 1, 1, 0], [1, 1, 1, 0]]
    assert objects[..., 11].tolist() == [[1, 1, 0, 0], [1, 1, 0, 0]]
    assert torch.isfinite(cmd.xi).all()
    assert ((objects[:, :2, 0] >= 5) & (objects[:, :2, 0] <= 15)).all()
    torch.testing.assert_close(objects[:, 2, 0], objects.new_full((2,), 10))
    pairs = cmd.dynamics.pair_friction(cmd.xi)
    torch.testing.assert_close(pairs[:, 1, 2], torch.maximum(pairs[:, 0, 1], pairs[:, 0, 2]))
    torch.testing.assert_close(pairs[:, 0, 3, 0], pairs.new_full((2,), .6))
    before = tuple(v.clone() for v in cmd.state()); context = cmd.xi.clone()
    masses = env.sim.model.body_mass.clone(); inertias = env.sim.model.body_inertia.clone()
    reset_frames(env, (63,), torch.tensor([0], device=env.device))
    for after, old in zip(cmd.state(), before):
        torch.testing.assert_close(after[1], old[1], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cmd.xi[1], context[1], atol=0, rtol=0)
    torch.testing.assert_close(env.sim.model.body_mass[1], masses[1], atol=0, rtol=0)
    torch.testing.assert_close(env.sim.model.body_inertia[1], inertias[1], atol=0, rtol=0)
    assert not torch.equal(objects[0, :2, 0], cmd.dynamics.object_context(context)[0, :2, 0])
    torch.testing.assert_close(cmd.xi[:, 0], context[:, 0], atol=0, rtol=0)


def test_real_multi_object_contacts_match_pair_friction_without_duplicate_copies(multi_entity_env):
    env = multi_entity_env; cmd = env.command_manager.get_term("motion")
    reset_frames(env)
    ids = torch.arange(env.num_envs, device=env.device)
    box = env.scene["box"]
    pose = torch.cat((box.data.root_link_pos_w, box.data.root_link_quat_w), -1).clone()
    pose[:, :3] = torch.tensor([0., 0., .45], device=env.device) + env.scene.env_origins
    box.write_root_link_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()
    count = int(env.sim.data.nacon[0])
    assert count < env.sim.wp_data.naconmax
    contact = env.sim.data.contact
    geom, world, friction = contact.geom[:count], contact.worldid[:count].long(), contact.friction[:count]
    types = torch.as_tensor(env.sim.mj_model.geom_contype, device=env.device)
    chair_ids, box_ids = env.scene["chair"].indexing.geom_ids, box.indexing.geom_ids
    pair_contact = torch.isin(geom, chair_ids).any(-1) & torch.isin(geom, box_ids).any(-1)
    assert pair_contact.any(), "The test box must intersect the real chair"
    assert types[geom[pair_contact]].eq(2).all(), "Ground-only copies must not produce object-object contacts"
    expected = cmd.dynamics.pair_friction(cmd.xi)[world[pair_contact], 1, 2, 0, None].expand(-1, 2)
    torch.testing.assert_close(friction[pair_contact, :2], expected, atol=1e-6, rtol=1e-6)
    # Move the robot and chair away, then let the second independent rigid body
    # fall. This verifies actual dynamics and floor support, not just masks.
    for name in ("robot", "chair"):
        entity = env.scene[name]
        p = torch.cat((entity.data.root_link_pos_w, entity.data.root_link_quat_w), -1).clone()
        p[:, 0] += 20
        entity.write_root_link_pose_to_sim(p, env_ids=ids)
    pose[:, 0], pose[:, 2] = 0., .6
    box.write_root_link_pose_to_sim(pose, env_ids=ids)
    box.write_root_link_velocity_to_sim(torch.zeros(2, 6, device=env.device), env_ids=ids)
    for _ in range(160):
        env.sim.step()
    env.sim.forward()
    z = env.sim.data.qpos[:, box.indexing.free_joint_q_adr][:, 2]
    torch.testing.assert_close(z, torch.full_like(z, .2), atol=.01, rtol=0)
    reset_frames(env)


def test_native_reward_clock_and_evaluation_parameters(multi_entity_env):
    env = multi_entity_env; cmd = env.command_manager.get_term("motion")
    reset_frames(env)
    a = torch.zeros(2, 29, device=env.device)
    exact = cmd.mdp.reward(cmd.state(), cmd.time_steps, a, a)
    torch.testing.assert_close(exact, torch.full_like(exact, cmd.mdp.max_step_reward), atol=2e-6, rtol=2e-6)
    _, reward, _, _, _ = env.step(a)
    torch.testing.assert_close(reward, cmd.mdp.reward(cmd.state(), cmd.time_steps, a, a))
    cmd.cfg.play = True
    try:
        reset_frames(env)
        objects = cmd.dynamics.object_context(cmd.xi)
        torch.testing.assert_close(objects[:, :3, 0], objects.new_full((2, 3), 10))
        pairs = cmd.dynamics.pair_friction(cmd.xi)
        torch.testing.assert_close(pairs[:, 0, 1:3, 0], pairs.new_ones((2, 2)))
    finally:
        cmd.cfg.play = False
