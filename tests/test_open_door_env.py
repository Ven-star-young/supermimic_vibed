"""Real CUDA task registration, passive hinge, and native/replay boundaries."""

from dataclasses import asdict, replace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.tasks.open_door.assets import DEFAULT_MOTION_DIR


TASK = "Mjlab-SuperMimic-OpenDoor-Unitree-G1"


@pytest.fixture(scope="module")
def door_runner(cuda_device):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from supermimic.rl.runner import SuperMimicRunner

    if not list(DEFAULT_MOTION_DIR.glob("*.npz")):
        pytest.skip("Convert the supplied OpenDoor motion with motion_data.converters.gmr first")
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs, cfg.scene.env_spacing, cfg.auto_reset = 4, 2.5, False
    cfg.commands["motion"].motion_dir = str(DEFAULT_MOTION_DIR)
    cfg.commands["motion"].reset_noise = False
    # Non-default values catch an asset factory that ignores command overrides.
    cfg.commands["motion"].door_mass = 9.5
    cfg.commands["motion"].door_damping = 3.25
    cfg.commands["motion"].door_friction = .35
    cfg.commands["motion"].robot_door_friction = .8
    cfg.commands["motion"].door_stiffness = 14.
    cfg.commands["motion"].door_spring_reference = -.07
    assert load_runner_cls(TASK) is SuperMimicRunner
    env = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        env.reset()
        training = asdict(load_rl_cfg(TASK)) | {"no_compile": True, "amp": "none"}
        runner = load_runner_cls(TASK)(env, training, device=cuda_device)
        runner.replay_env = runner.make_replay_env()
        yield runner
    finally:
        env.close()


def _reset(runner, frames=(80, 140, 200, 270)):
    replay = runner.replay_env
    frame = torch.tensor(frames, device=replay.device)
    observations, _ = replay.reset(frames=frame)
    return observations, frame


def test_registered_runner_observation_contract_and_midclip_reset(door_runner):
    runner, env = door_runner, door_runner.replay_env
    observations, frame = _reset(runner)
    assert env.task.name == "g1_open_door"
    assert env.task.modeled_entities == ("robot", "door")
    assert set(env.env.cfg.scene.entities) == {"robot", "door"}
    assert set(env.env.scene.entities) == {"robot", "door", "terrain"}
    assert env.env.action_manager.total_action_dim == 29
    for key, width in (("state", 73), ("policy_history", 1020), ("goal", 750), ("context", 46)):
        assert observations[key].shape == (4, width)
    assert "geometry_id" not in observations and runner.geometry_bank is None
    reference = env.motion.full_ref_state(frame)
    for actual, expected in zip(env.state(), reference):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(observations["goal"], env.motion.goal(frame, 10))
    history = observations["policy_history"].reshape(4, 10, 102)
    torch.testing.assert_close(history, history[:, :1].expand_as(history))
    with torch.no_grad():
        action = runner.pi(history, observations["goal"])
    assert action.shape == (4, 29) and torch.isfinite(action).all()


def test_hinge_asset_physics_and_nonzero_scene_origins_match_context(door_runner):
    env = door_runner.replay_env
    _reset(door_runner)
    native, command = env.env, env.command
    door, model = native.scene["door"], native.sim.mj_model
    assert door.joint_names == ("door_hinge",) and door.is_fixed_base and not door.is_actuated
    jid = int(door.indexing.joint_ids[0])
    panel_id = next(int(g) for g in door.indexing.geom_ids
                    if model.geom(int(g)).name.endswith("door_panel"))
    body = model.geom_bodyid[panel_id]
    dof, qpos = model.jnt_dofadr[jid], model.jnt_qposadr[jid]
    assert model.joint(jid).name.endswith("door_hinge")
    np.testing.assert_array_equal(model.jnt_axis[jid], [0., 0., -1.])
    metadata = command.motion.door_metadata
    np.testing.assert_allclose(model.geom_size[panel_id], metadata["panel_half_size"])
    np.testing.assert_allclose(model.jnt_range[jid], metadata["joint_range"])
    assert model.actuator_trnid[:, 0].tolist().count(jid) == 0
    actual_physics = [model.body_mass[body], model.dof_damping[dof], model.dof_frictionloss[dof],
                      command.cfg.physical_cfg.evaluation_friction, model.geom_friction[panel_id, 0],
                      model.jnt_stiffness[jid], model.qpos_spring[qpos]]
    expected = env.xi.new_tensor(actual_physics).expand(4, -1)
    torch.testing.assert_close(env.xi[:, 39:46], expected)
    np.testing.assert_allclose(actual_physics, [9.5, 3.25, .35, 1., .8, 14., -.07])
    floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    actual_floor = native.sim.model.geom_friction[:, floor, 0]
    torch.testing.assert_close(actual_floor, env.xi[:, 42:43].expand_as(actual_floor))
    torch.testing.assert_close(env.xi[:, 36:39],
                               env.xi.new_tensor(2 * metadata["panel_half_size"][[1, 2, 0]]).expand(4, -1))
    origins = native.scene.env_origins
    assert (origins.abs().sum(-1) > 0).all()
    torch.testing.assert_close(native.sim.data.xanchor[:, jid] - origins, env.xi[:, 33:36],
                               atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(command.robot.data.root_link_pos_w - origins, env.state()[0])
    theta = env.state()[6].squeeze(-1)
    yaw = theta * float(metadata["joint_axis"][2]) + float(metadata["base_yaw"])
    offset = env.xi.new_tensor(metadata["hinge_to_panel"])
    rotated = torch.stack((yaw.cos() * offset[0] - yaw.sin() * offset[1],
                           yaw.sin() * offset[0] + yaw.cos() * offset[1],
                           torch.ones_like(yaw) * offset[2]), -1)
    torch.testing.assert_close(native.sim.data.geom_xpos[:, panel_id] - origins,
                               env.xi[:, 33:36] + rotated, atol=2e-6, rtol=2e-6)


def test_partial_reference_reset_preserves_other_worlds(door_runner):
    env = door_runner.replay_env
    _reset(door_runner)
    before = tuple(value.clone() for value in env.state())
    ids = torch.tensor([0, 2], device=env.device)
    frames = torch.tensor([180, 240], device=env.device)
    env.reset(ids, frames)
    for actual, expected, old in zip(env.state(), env.motion.full_ref_state(frames), before):
        torch.testing.assert_close(actual[ids], expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual[[1, 3]], old[[1, 3]], atol=0, rtol=0)


def test_real_transition_reward_is_the_same_state_head_at_next_frame(door_runner):
    runner, env = door_runner, door_runner.replay_env
    _reset(runner)
    action = torch.linspace(-.05, .05, 29, device=env.device).expand(4, -1).clone()
    before, frames = env.dynamics.pack(env.state()), env.f.clone()
    result = env.step(action)
    torch.testing.assert_close(result["s"], before)
    torch.testing.assert_close(result["f"], frames)
    state = env.dynamics.unpack(result["s1"])
    expected = env.mdp.reward(state, frames + 1, result["a"], result["a_prev"])
    imagined, _ = runner._reward_term(state, frames + 1, result["a"], result["a_prev"])
    torch.testing.assert_close(imagined, expected, atol=0, rtol=0)
    torch.testing.assert_close(result["r"], expected, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(result["s1"]).all() and not torch.allclose(result["s1"], result["s"])


def test_contact_diagnostics_cannot_change_training_reward(door_runner, monkeypatch):
    from supermimic.mdp.native import tracking_reward
    env = door_runner.replay_env
    _reset(door_runner)
    baseline = tracking_reward(env.env).clone()
    for name, value in (("allowed_back_contact", 1.), ("forbidden_door_contact", 0.)):
        monkeypatch.setitem(env.command.metrics, name, torch.full((4,), value, device=env.device))
    torch.testing.assert_close(tracking_reward(env.env), baseline, atol=0, rtol=0)
    env.command.metrics["allowed_back_contact"].zero_()
    env.command.metrics["forbidden_door_contact"].fill_(1.)
    torch.testing.assert_close(tracking_reward(env.env), baseline, atol=0, rtol=0)


def test_replay_retains_terminal_hinge_state_and_reward_before_reset(door_runner):
    from supermimic.rl.storage import ReplayBuffer
    env = door_runner.replay_env
    _reset(door_runner)
    env.reset(torch.tensor([0], device=env.device), env.motion.clip_end[:1] - 1)
    before = env.dynamics.pack(env.state())
    action = torch.zeros(4, 29, device=env.device)
    result = env.step(action)
    assert result["done"][0] and env.ep_step[0] == 0
    torch.testing.assert_close(result["s"], before)
    state = env.dynamics.unpack(result["s1"])
    expected = env.mdp.reward(state, result["f"] + 1, action, result["a_prev"])
    torch.testing.assert_close(result["r"], expected, atol=2e-6, rtol=2e-5)
    assert not torch.allclose(result["s1"][0, 71:73], env.dynamics.pack(env.state())[0, 71:73])
    replay = ReplayBuffer(2, 4, env.device, state_dim=73, action_dim=29, context_dim=46)
    replay.add(result)
    saved = result["s1"].clone()
    env.step(action)
    torch.testing.assert_close(result["s1"], saved, atol=0, rtol=0)
    torch.testing.assert_close(replay.data["s1"][0], saved, atol=0, rtol=0)


def test_unmodeled_entities_are_rejected_before_scene_rebuild():
    from mjlab.entity import EntityCfg
    from mjlab.tasks.registry import load_env_cfg
    cfg = load_env_cfg(TASK)
    scene = replace(cfg.scene, entities={**cfg.scene.entities, "obstacle": EntityCfg()})
    with pytest.raises(ValueError, match="unsupported entities.*obstacle"):
        replace(cfg, scene=scene)
