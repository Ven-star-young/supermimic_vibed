"""Actual ManagerBasedRlEnv lifecycle, replay boundaries and shared MDP heads."""

import mujoco
import torch

from supermimic.mdp import MAX_STEP_REWARD, flatten_state, unflatten_state
from supermimic.motion import CONTROL_DT
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, TRACKED_BODIES
from supermimic.rotations import quat_to_mat


def command(env):
    return env.env.command_manager.get_term("motion")


def reset_reference(env):
    ids = torch.arange(env.N, device=env.device)
    frames = torch.tensor([3, 29, 57, 91], device=env.device)
    env.reset(ids, frames)
    return ids, frames


def test_runtime_is_stock_manager_environment(eval_env):
    from mjlab.envs import ManagerBasedRlEnv
    assert type(eval_env.env) is ManagerBasedRlEnv
    assert not eval_env.env.cfg.auto_reset
    assert eval_env.env.action_manager.total_action_dim == 29


def test_immediate_reset_preserves_pose_velocity_and_joint_order(eval_env):
    env, cmd = eval_env, command(eval_env)
    _, frames = reset_reference(env)
    for actual, expected in zip(env.state(), env.motion.ref_state(frames)):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    link_v = cmd.robot.data.root_link_lin_vel_w
    com_v = cmd.robot.data.root_com_lin_vel_w
    assert (link_v - com_v).abs().max() > 1e-3
    joint_ids = [cmd.robot.joint_names.index(name) for name in G1_JOINT_NAMES]
    torch.testing.assert_close(cmd.robot.data.joint_pos[:, joint_ids], env.motion.q[frames])
    torch.testing.assert_close(cmd.robot.data.joint_vel[:, joint_ids], env.motion.qd[frames])


def test_link_fk_matches_mjlab_and_reference_reward(eval_env):
    env, cmd = eval_env, command(eval_env)
    _, frames = reset_reference(env)
    state = env.state()
    body_ids = [cmd.robot.body_names.index(name) for name in TRACKED_BODIES]
    data = cmd.robot.data
    simulator = (
        data.body_link_pos_w[:, body_ids] - env.env.scene.env_origins[:, None],
        quat_to_mat(data.body_link_quat_w[:, body_ids]),
        data.body_link_lin_vel_w[:, body_ids],
        data.body_link_ang_vel_w[:, body_ids],
    )
    for actual, expected in zip(env.mdp.kin.forward(*state), simulator):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    action = torch.zeros(env.N, 29, device=env.device)
    reward = env.mdp.reward(state, frames, action, action)
    torch.testing.assert_close(reward, torch.full_like(reward, MAX_STEP_REWARD), atol=1e-6, rtol=1e-6)


def test_partial_reset_preserves_other_worlds_and_observation_history(eval_env):
    env = eval_env
    reset_reference(env)
    env.step(torch.zeros(env.N, 29, device=env.device))
    before = tuple(value.clone() for value in env.state())
    before_f, before_steps = env.f.clone(), env.ep_step.clone()
    before_obs = {key: value.clone() for key, value in env.env.get_observations().items()}
    ids = torch.tensor([0, 2], device=env.device)
    untouched = torch.tensor([1, 3], device=env.device)
    frames = torch.tensor([41, 105], device=env.device)
    env.reset(ids, frames)
    for actual, expected, previous in zip(env.state(), env.motion.ref_state(frames), before):
        torch.testing.assert_close(actual[ids], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[untouched], previous[untouched], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(env.f[untouched], before_f[untouched])
    torch.testing.assert_close(env.ep_step[untouched], before_steps[untouched])
    after_obs = env.env.get_observations()
    for key in before_obs:
        torch.testing.assert_close(after_obs[key][untouched], before_obs[key][untouched], atol=0, rtol=0)
    history = after_obs["policy_history"][ids].reshape(len(ids), 10, -1)
    torch.testing.assert_close(history, history[:, -1:].expand_as(history))


def test_control_step_uses_integrated_state_and_next_reference(eval_env):
    env, cmd = eval_env, command(eval_env)
    reset_reference(env)
    before = flatten_state(env.state())
    before_time = env.env.sim.data.time.clone()
    result = env.step(torch.zeros(env.N, 29, device=env.device))
    torch.testing.assert_close(env.env.sim.data.time - before_time, torch.full_like(before_time, CONTROL_DT), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["s"], before)
    torch.testing.assert_close(result["s1"], flatten_state(env.state()))
    assert torch.isfinite(result["r"]).all()
    assert not torch.allclose(result["s"], result["s1"])
    root_qpos = env.env.sim.data.qpos[:, cmd.robot.indexing.free_joint_q_adr]
    root_qvel = env.env.sim.data.qvel[:, cmd.robot.indexing.free_joint_v_adr]
    next_state = unflatten_state(result["s1"])
    torch.testing.assert_close(next_state[0], root_qpos[:, :3] - env.env.scene.env_origins)
    torch.testing.assert_close(next_state[1], quat_to_mat(root_qpos[:, 3:7]))
    torch.testing.assert_close(next_state[2], root_qvel[:, :3], atol=2e-5, rtol=2e-5)
    world_w = (next_state[1] @ root_qvel[:, 3:6, None]).squeeze(-1)
    torch.testing.assert_close(next_state[3], world_w, atol=2e-5, rtol=2e-5)
    expected = env.mdp.reward(next_state, result["f"] + 1, result["a"], result["a_prev"])
    torch.testing.assert_close(result["r"], expected)
    torch.testing.assert_close(env.f, result["f"] + 1)


def test_terminal_transition_is_saved_before_adapter_reset(eval_env):
    env = eval_env
    reset_reference(env)
    env.reset(torch.tensor([0], device=env.device), env.motion.clip_end[:1] - 1)
    before = flatten_state(env.state())
    result = env.step(torch.zeros(env.N, 29, device=env.device))
    assert result["done"][0] and not result["term"][0]
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


def assert_physical_context(env, *, play):
    cmd, sim = command(env), env.env.sim
    geoms = cmd.robot.indexing.geom_ids
    friction = sim.model.geom_friction[:, geoms, 0]
    torch.testing.assert_close(friction, env.xi[:, :1].expand_as(friction))
    torso_id = cmd.robot.body_names.index("torso_link")
    torso = int(cmd.robot.indexing.body_ids[torso_id])
    base_com = torch.tensor(sim.mj_model.body_ipos[torso], device=env.device, dtype=torch.float32)
    torch.testing.assert_close(sim.model.body_ipos[:, torso], base_com + env.xi[:, 30:33])
    planes = torch.as_tensor(sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=env.device)
    ground_friction = sim.model.geom_friction[:, planes, 0]
    torch.testing.assert_close(ground_friction, torch.full_like(ground_friction, 1.0 if play else 0.3))


def test_randomized_context_changes_actual_model(train_env):
    env = train_env
    assert_physical_context(env, play=False)
    assert ((env.xi[:, 0] >= 0.3) & (env.xi[:, 0] <= 1.2)).all()
    assert env.xi[:, 0].std() > 0
    assert (env.xi[:, 30].abs() <= 0.025).all()
    assert (env.xi[:, 31:33].abs() <= 0.05).all()
    old = env.xi.clone()
    env.reset(torch.tensor([0], device=env.device), torch.tensor([10], device=env.device))
    torch.testing.assert_close(env.xi[:, 0], old[:, 0])
    torch.testing.assert_close(env.xi[:, 30:33], old[:, 30:33])
    torch.testing.assert_close(env.xi[1:, 1:30], old[1:, 1:30])
    assert not torch.allclose(env.xi[0, 1:30], old[0, 1:30])


def test_play_context_has_no_randomization(eval_env):
    assert_physical_context(eval_env, play=True)
    torch.testing.assert_close(eval_env.xi[:, 0], torch.ones(eval_env.N, device=eval_env.device))
    torch.testing.assert_close(eval_env.xi[:, 1:], torch.zeros_like(eval_env.xi[:, 1:]))


def test_stock_play_auto_reset_refills_only_finished_history(cuda_device, motion_dir):
    from tools._smoke import native_env, reset_frames
    with native_env(2, device=cuda_device, motion_dir=motion_dir, auto_reset=True) as env:
        cmd = env.command_manager.get_term("motion")
        reset_frames(env, torch.tensor([3, 29], device=cuda_device))
        action = torch.zeros(2, 29, device=cuda_device)
        env.step(action)
        frames = cmd.time_steps.clone()
        reset_frames(env, cmd.motion.clip_end[:1] - 1, torch.tensor([0], device=cuda_device))
        obs, _, terminated, timeout, _ = env.step(action)
        assert timeout[0] and not terminated[0]
        assert env.episode_length_buf[0] == 0
        assert cmd.time_steps[1] == frames[1] + 1
        history = obs["policy_history"].reshape(2, 10, -1)
        torch.testing.assert_close(history[0], history[0, -1:].expand_as(history[0]))
        assert not torch.allclose(history[1, 0], history[1, -1])
