"""Migration regressions using real MuJoCo-Warp, independent of training data."""

import mujoco
import torch

from supermimic.mdp import MAX_STEP_REWARD, flatten_state, unflatten_state
from supermimic.motion import CONTROL_DT
from supermimic.robot import G1_JOINT_NAMES, TRACKED_BODIES
from supermimic.rotations import quat_to_mat


def reset_reference(env):
    ids = torch.arange(env.N, device=env.device)
    frames = torch.tensor([3, 29, 57, 91], device=env.device)
    env.reset(ids, frames)
    return ids, frames


def test_immediate_reset_preserves_pose_velocity_and_joint_order(eval_env):
    env = eval_env
    _, frames = reset_reference(env)
    for actual, expected in zip(env.state(), env.motion.ref_state(frames)):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    # Nonzero root COM offset and angular velocity make an accidental COM
    # velocity reset measurably different from the required link-origin velocity.
    link_v = env.robot.data.root_link_lin_vel_w
    com_v = env.robot.data.root_com_lin_vel_w
    assert (link_v - com_v).abs().max() > 1e-3
    joint_ids = [env.robot.joint_names.index(name) for name in G1_JOINT_NAMES]
    torch.testing.assert_close(env.robot.data.joint_pos[:, joint_ids], env.motion.q[frames])
    torch.testing.assert_close(env.robot.data.joint_vel[:, joint_ids], env.motion.qd[frames])


def test_link_fk_matches_mjlab_and_reference_reward(eval_env):
    env = eval_env
    _, frames = reset_reference(env)
    state = env.state()
    body_ids = [env.robot.body_names.index(name) for name in TRACKED_BODIES]
    data = env.robot.data
    simulator = (
        data.body_link_pos_w[:, body_ids] - env.scene.env_origins[:, None],
        quat_to_mat(data.body_link_quat_w[:, body_ids]),
        data.body_link_lin_vel_w[:, body_ids],
        data.body_link_ang_vel_w[:, body_ids],
    )
    for actual, expected in zip(env.mdp.kin.forward(*state), simulator):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    action = torch.zeros(env.N, 29, device=env.device)
    reward = env.mdp.reward(state, frames, action, action)
    torch.testing.assert_close(reward, torch.full_like(reward, MAX_STEP_REWARD), atol=1e-6, rtol=1e-6)


def test_partial_reset_leaves_other_worlds_unchanged(eval_env):
    env = eval_env
    reset_reference(env)
    env.step(torch.zeros(env.N, 29, device=env.device))
    before = env.state()
    before_f, before_steps = env.f.clone(), env.ep_step.clone()
    ids = torch.tensor([0, 2], device=env.device)
    untouched = torch.tensor([1, 3], device=env.device)
    frames = torch.tensor([41, 105], device=env.device)
    env.reset(ids, frames)
    for actual, expected, previous in zip(env.state(), env.motion.ref_state(frames), before):
        torch.testing.assert_close(actual[ids], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[untouched], previous[untouched], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(env.f[untouched], before_f[untouched])
    torch.testing.assert_close(env.ep_step[untouched], before_steps[untouched])


def test_control_step_and_next_state_are_integrated(eval_env):
    env = eval_env
    reset_reference(env)
    before = flatten_state(env.state())
    before_time = env.sim.data.time.clone()
    result = env.step(torch.zeros(env.N, 29, device=env.device))
    torch.testing.assert_close(env.sim.data.time - before_time, torch.full_like(before_time, CONTROL_DT), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["s"], before)
    torch.testing.assert_close(result["s1"], flatten_state(env.state()))
    assert torch.isfinite(result["r"]).all()
    assert torch.isfinite(result["s1"]).all()
    assert not torch.allclose(result["s"], result["s1"])
    # Derived poses/velocities must describe the integrated generalized state,
    # not the pre-integration kinematics that mj_step otherwise leaves behind.
    root_qpos = env.sim.data.qpos[:, env.robot.indexing.free_joint_q_adr]
    root_qvel = env.sim.data.qvel[:, env.robot.indexing.free_joint_v_adr]
    next_state = unflatten_state(result["s1"])
    torch.testing.assert_close(next_state[0], root_qpos[:, :3] - env.scene.env_origins)
    torch.testing.assert_close(next_state[1], quat_to_mat(root_qpos[:, 3:7]))
    torch.testing.assert_close(next_state[2], root_qvel[:, :3], atol=2e-5, rtol=2e-5)
    world_w = (next_state[1] @ root_qvel[:, 3:6, None]).squeeze(-1)
    torch.testing.assert_close(next_state[3], world_w, atol=2e-5, rtol=2e-5)


def test_terminal_transition_survives_auto_reset(train_env):
    env = train_env
    reset_reference(env)
    env.ep_step[0] = env.max_episode_steps - 1
    before = flatten_state(env.state())
    result = env.step(torch.zeros(env.N, 29, device=env.device))
    assert result["done"][0]
    assert not result["term"][0]
    assert not result["done"][1:].any()
    assert env.ep_step[0] == 0
    torch.testing.assert_close(result["s"], before)
    terminal_state = unflatten_state(result["s1"])
    expected_reward = env.mdp.reward(terminal_state, result["f"] + 1, result["a"], result["a_prev"])
    torch.testing.assert_close(result["r"], expected_reward)
    torch.testing.assert_close(result["s1"][1:], flatten_state(env.state())[1:])
    assert not torch.allclose(result["s1"][0], flatten_state(env.state())[0])
    saved = result["s1"].clone()
    env.step(torch.zeros(env.N, 29, device=env.device))
    torch.testing.assert_close(result["s1"], saved)


def assert_physical_context(env):
    geoms = env.robot.indexing.geom_ids
    friction = env.sim.model.geom_friction[:, geoms, 0]
    torch.testing.assert_close(friction, env.xi[:, :1].expand_as(friction))
    torso = int(env.robot.indexing.body_ids[env.torso_id])
    base_com = torch.tensor(env.sim.mj_model.body_ipos[torso], device=env.device, dtype=torch.float32)
    torch.testing.assert_close(env.sim.model.body_ipos[:, torso], base_com + env.xi[:, 30:33])
    planes = torch.as_tensor(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=env.device)
    ground_friction = env.sim.model.geom_friction[:, planes, 0]
    torch.testing.assert_close(ground_friction, torch.full_like(ground_friction, 1.0 if env.eval_mode else 0.3))


def test_randomized_context_changes_actual_model(train_env):
    env = train_env
    assert_physical_context(env)
    assert ((env.xi[:, 0] >= 0.3) & (env.xi[:, 0] <= 1.2)).all()
    assert env.xi[:, 0].std() > 0
    assert (env.xi[:, 30].abs() <= 0.025).all()
    assert (env.xi[:, 31:33].abs() <= 0.05).all()
    assert env.xi[:, 30:33].abs().max() > 0
    old = env.xi.clone()
    env.reset(torch.tensor([0], device=env.device), torch.tensor([10], device=env.device))
    torch.testing.assert_close(env.xi[:, 0], old[:, 0])
    torch.testing.assert_close(env.xi[:, 30:33], old[:, 30:33])
    torch.testing.assert_close(env.xi[1:, 1:30], old[1:, 1:30])
    assert not torch.allclose(env.xi[0, 1:30], old[0, 1:30])
    assert (env.xi[:, 1:30].abs() <= 0.01).all()
    assert_physical_context(env)


def test_eval_context_has_no_randomization(eval_env):
    env = eval_env
    assert_physical_context(env)
    torch.testing.assert_close(env.xi[:, 0], torch.ones(env.N, device=env.device))
    torch.testing.assert_close(env.xi[:, 1:], torch.zeros(env.N, 32, device=env.device))
