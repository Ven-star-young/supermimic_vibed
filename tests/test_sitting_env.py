"""Real chair physics, independent contact checks and native reset integration."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def sitting_file():
    from supermimic.assets import DATA_DIR
    path = DATA_DIR / "motions/sitting/sitting__acadian_rustic_wooden_dining_chair_distressed_white__000.npz"
    if not path.is_file():
        pytest.skip("Real sitting motion fixture is unavailable")
    from motion_data import load_motion
    from motion_data.collision import DEFAULT_CACHE_DIR, collision_key
    if not (DEFAULT_CACHE_DIR / f"{collision_key(load_motion(path))}.npz").is_file():
        pytest.skip("Prepare the real chair collision cache to run physics tests")
    return path


@pytest.fixture(scope="module")
def sitting_env(cuda_device, sitting_file):
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.tasks.sitting.env_cfgs import make_env_cfg
    cfg = make_env_cfg(play=False)
    cfg.scene.num_envs, cfg.auto_reset = 2, False
    cfg.commands["motion"].motion_dir = str(sitting_file)
    cfg.commands["motion"].reset_noise = False
    env = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        env.reset()
        yield env
    finally:
        env.close()


def reset_frames(env, frames=(3, 137), ids=None):
    from tests.helpers import reset_frames as reset
    return reset(env, torch.tensor(frames, device=env.device), ids)


def contact_data(env):
    env.sim.forward()
    count = int(env.sim.data.nacon[0])
    assert count < env.sim.wp_data.naconmax, "The contact buffer overflowed"
    contacts = env.sim.data.contact
    return contacts.geom[:count], contacts.friction[:count], contacts.worldid[:count]


def test_sitting_config_is_lazy_and_cli_uses_updated_command(monkeypatch):
    import mjlab
    import tyro
    from supermimic.tasks.sitting import assets
    from supermimic.tasks.sitting.env_cfgs import SittingEnvCfg, make_env_cfg
    monkeypatch.setattr(assets, "load_motion", lambda *_: pytest.fail("Factory must not read dataset"))
    cfg = make_env_cfg(play=True)
    assert set(cfg.scene.entities) == {"robot", "chair"}
    parsed = tyro.cli(SittingEnvCfg, default=cfg, config=mjlab.TYRO_FLAGS,
                      args=["--commands.motion.motion-dir", "/tmp/updated-motion-directory"])
    assert parsed.scene.entities["chair"].spec_fn.args[0] is parsed.commands["motion"]
    assert parsed.scene.entities["chair"].spec_fn.args[0].motion_dir == "/tmp/updated-motion-directory"
    assert cfg.commands["motion"].motion_dir != parsed.commands["motion"].motion_dir
    assert cfg.sim.nconmax >= 512 and cfg.sim.njmax >= 2000
    assert "geometry_id" in cfg.observations


def test_geometry_sampling_scale_and_missing_cache_are_explicit(sitting_file, tmp_path):
    from supermimic.tasks.sitting.assets import sample_geometry, selected_asset
    from supermimic.tasks.sitting.commands import SittingCommandCfg
    cfg = SittingCommandCfg(motion_dir=str(sitting_file))
    asset = selected_asset(cfg)
    points, extent = sample_geometry(asset.parts, asset.clip["object_scale"])
    np.testing.assert_array_equal(points, asset.points)
    doubled, double_extent = sample_geometry(asset.parts, asset.clip["object_scale"] * 2)
    np.testing.assert_allclose(doubled[:, :3], 2 * points[:, :3], atol=1e-7)
    np.testing.assert_allclose(doubled[:, 3:], points[:, 3:], atol=1e-7)
    np.testing.assert_allclose(double_extent, 2 * extent, atol=1e-7)
    with pytest.raises(FileNotFoundError, match="prepare_collisions"):
        selected_asset(replace(cfg, collision_cache_dir=str(tmp_path)))


def test_scene_selection_filters_shape_and_scale_without_using_filenames(sitting_file, tmp_path):
    from motion_data import load_motion, save_motion
    from supermimic.tasks.sitting.assets import select_motion_files
    from supermimic.tasks.sitting.commands import SittingCommandCfg
    clip = load_motion(sitting_file)
    save_motion(tmp_path / "a.npz", clip)
    save_motion(tmp_path / "unrelated-name.npz", {**clip, "root_pos": clip["root_pos"] + .01})
    save_motion(tmp_path / "different-scale.npz", {**clip, "object_scale": clip["object_scale"] * 1.01})
    cfg = SittingCommandCfg(motion_dir=str(tmp_path))
    assert {p.name for p in select_motion_files(cfg)} == {"a.npz", "unrelated-name.npz"}
    cfg.object_motion_file = str(tmp_path / "different-scale.npz")
    assert [p.name for p in select_motion_files(cfg)] == ["different-scale.npz"]


def test_sitting_reset_uses_one_reference_frame_for_robot_and_object(sitting_env):
    env = sitting_env
    cmd = env.command_manager.get_term("motion")
    obs, _ = reset_frames(env)
    expected = cmd.motion.full_ref_state(torch.tensor([3, 137], device=env.device))
    for actual, reference in zip(cmd.state(), expected):
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    assert obs["state"].shape == (2, 84)
    assert obs["policy_history"].shape == (2, 1020)
    assert obs["context"].shape == (2, 46)
    assert obs["geometry_id"].shape == (2, 1)
    torch.testing.assert_close(obs["geometry_id"], obs["context"][:, 45:46])
    assert env.scene["chair"].indexing.free_joint_q_adr.numel() == 7


def test_object_randomization_updates_actual_mass_inertia_and_partial_reset(sitting_env):
    from supermimic.tasks.sitting.assets import selected_asset
    env = sitting_env
    cmd = env.command_manager.get_term("motion")
    reset_frames(env)
    body = env.scene["chair"].indexing.root_body_id
    mass = cmd.xi[:, 33]
    assert ((5 <= mass) & (mass <= 15)).all() and mass.std() > 0
    assert ((.3 <= cmd.xi[:, 43]) & (cmd.xi[:, 43] <= 1.2)).all()
    torch.testing.assert_close(env.sim.model.body_mass[:, body], mass)
    torch.testing.assert_close(env.sim.model.body_subtreemass[:, body], mass)
    actual = cmd.xi[:, 37:43]
    inertia = selected_asset(cmd.cfg).inertia
    expected = torch.tensor(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                            device=env.device, dtype=torch.float32)[None] * (mass / 10)[:, None]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    env.step(torch.zeros(2, 29, device=env.device))
    before = [value.clone() for value in cmd.state()]
    xi, model_mass, model_inertia = (cmd.xi.clone(), env.sim.model.body_mass.clone(),
                                    env.sim.model.body_inertia.clone())
    reset_frames(env, (63,), torch.tensor([0], device=env.device))
    for actual, previous in zip(cmd.state(), before):
        torch.testing.assert_close(actual[1], previous[1], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cmd.xi[1], xi[1], atol=0, rtol=0)
    torch.testing.assert_close(env.sim.model.body_mass[1], model_mass[1], atol=0, rtol=0)
    torch.testing.assert_close(env.sim.model.body_inertia[1], model_inertia[1], atol=0, rtol=0)
    assert cmd.xi[0, 33] != xi[0, 33] and cmd.xi[0, 43] != xi[0, 43]
    # Existing G1 startup friction and COM randomization retain their lifecycle.
    torch.testing.assert_close(cmd.xi[:, 0], xi[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(cmd.xi[:, 30:33], xi[:, 30:33], atol=0, rtol=0)


def test_reference_reward_and_real_transition_share_the_same_head(sitting_env):
    env = sitting_env
    cmd = env.command_manager.get_term("motion")
    reset_frames(env)
    action = torch.zeros(2, 29, device=env.device)
    exact = cmd.mdp.reward(cmd.state(), cmd.time_steps, action, action)
    torch.testing.assert_close(exact, torch.full_like(exact, cmd.mdp.max_step_reward), atol=2e-6, rtol=2e-6)
    _, reward, _, _, _ = env.step(action)
    expected = cmd.mdp.reward(cmd.state(), cmd.time_steps, action, action)
    torch.testing.assert_close(reward, expected)


def test_robot_chair_contacts_use_sampled_friction_and_no_ground_duplicates(sitting_env):
    env = sitting_env
    cmd = env.command_manager.get_term("motion")
    robot_geoms = env.scene["robot"].indexing.geom_ids
    chair_geoms = env.scene["chair"].indexing.geom_ids
    types = torch.as_tensor(env.sim.mj_model.geom_contype, device=env.device)
    chair_contact = chair_geoms[types[chair_geoms] == 2]
    ground_copy = chair_geoms[torch.tensor([
        env.sim.mj_model.geom(int(index)).name.endswith("_ground") for index in chair_geoms
    ], device=env.device)]
    found = False
    previous = cmd.cfg.object_physical_cfg.robot_friction
    cmd.cfg.object_physical_cfg.robot_friction = (.3, .3)
    try:
        assert (cmd.xi[:, 0] > .3).all()  # Normal mixing would choose this higher robot value.
        for frame in (30, 80, 130, 180, 230):
            reset_frames(env, (frame, frame))
            geom, friction, world = contact_data(env)
            both = torch.isin(geom, robot_geoms).any(-1) & torch.isin(geom, chair_contact).any(-1)
            assert not (torch.isin(geom, robot_geoms).any(-1) & torch.isin(geom, ground_copy).any(-1)).any()
            if both.any():
                expected = cmd.xi[world[both].long(), 43:44].expand(-1, 2)
                torch.testing.assert_close(friction[both, :2], expected, atol=1e-6, rtol=1e-6)
                found = True
    finally:
        cmd.cfg.object_physical_cfg.robot_friction = previous
    assert found, "Test reference frames must exercise real robot-chair contacts"


def test_chair_falls_and_is_supported_by_ground(sitting_env):
    env = sitting_env
    cmd = env.command_manager.get_term("motion")
    reset_frames(env)
    ids = torch.arange(env.num_envs, device=env.device)
    chair = env.scene["chair"]
    pose = torch.cat((chair.data.root_link_pos_w, chair.data.root_link_quat_w), -1).clone()
    pose[:, 2] += .25
    chair.write_root_link_pose_to_sim(pose, env_ids=ids)
    chair.write_root_link_velocity_to_sim(torch.zeros(2, 6, device=env.device), env_ids=ids)
    robot_pose = torch.cat((cmd.robot.data.root_link_pos_w, cmd.robot.data.root_link_quat_w), -1).clone()
    robot_pose[:, 0] += 20
    cmd.robot.write_root_link_pose_to_sim(robot_pose, env_ids=ids)
    start = env.sim.data.qpos[:, chair.indexing.free_joint_q_adr][:, 2].clone()
    for _ in range(160):
        env.sim.step()
    env.sim.forward()
    finish = env.sim.data.qpos[:, chair.indexing.free_joint_q_adr][:, 2]
    assert (finish < start - .15).all()
    assert (finish > start - .4).all(), "Chair should settle on the floor, not fall through"
    geom, friction, world = contact_data(env)
    planes = torch.as_tensor(np.flatnonzero(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE),
                             device=env.device)
    contacts = torch.isin(geom, chair.indexing.geom_ids).any(-1) & torch.isin(geom, planes).any(-1)
    assert set(world[contacts].cpu().tolist()) == {0, 1}
    torch.testing.assert_close(friction[contacts, :2], torch.ones_like(friction[contacts, :2]))
    reset_frames(env)


def test_evaluation_object_parameters_are_fixed(sitting_env):
    env = sitting_env
    cmd = env.command_manager.get_term("motion")
    cmd.cfg.play = True
    try:
        reset_frames(env)
        torch.testing.assert_close(cmd.xi[:, 33], torch.full_like(cmd.xi[:, 33], 10))
        torch.testing.assert_close(cmd.xi[:, 43:45], torch.ones_like(cmd.xi[:, 43:45]))
    finally:
        cmd.cfg.play = False
