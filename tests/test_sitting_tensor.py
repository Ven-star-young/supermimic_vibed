"""Analytic object references, task isolation and differentiable critic contracts."""

import numpy as np
import pytest
import torch

from motion_data import save_motion
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.mdp import MDP
from supermimic.rotations import so3_exp, yaw_mat
from supermimic.rl.shared_world_model import SharedDynamics
from supermimic.tasks.sitting.mdp import SittingMDP, ObjectRewardCfg
from supermimic.tasks.sitting.motion import SittingMotionLib
from supermimic.tasks.sitting.task import SittingTask


def write_clip(path, *, offset=0.0, with_object=True):
    t = np.arange(6) / 25
    angle = 0.4 + 0.7 * t
    quaternion = np.stack((np.cos(angle / 2), t * 0, t * 0, np.sin(angle / 2)), -1)
    clip = dict(
        fps=25.0, root_pos=np.tile([0.0, 0.0, 0.8], (6, 1)),
        root_quat=np.tile([1.0, 0, 0, 0], (6, 1)),
        joint_pos=np.tile(default_joint_pos_list(), (6, 1)), joint_names=np.array(G1_JOINT_NAMES),
    )
    if with_object:
        clip.update(object_pos=np.array([1.0 + offset, 0.3, 0.5]) + t[:, None] * [0.4, -0.2, 0.1],
                    object_quat=quaternion, object_scale=np.ones(3))
    save_motion(path, clip)
    return path


@pytest.fixture
def interaction(cpu_kin, tmp_path):
    path = write_clip(tmp_path / "motion.npz")
    motion = SittingMotionLib([path], cpu_kin, "cpu")
    mdp = SittingMDP(cpu_kin, motion, torch.tensor(default_joint_pos_list()))
    return motion, mdp


def context(batch=2, present=True):
    xi = torch.zeros(batch, 47)
    xi[:, 0], xi[:, 33], xi[:, 43:45] = 0.8, 10.0, 1.0
    xi[:, 45:] = float(present)
    return xi


def test_object_reference_clock_and_clip_boundary_velocity(cpu_kin, tmp_path):
    files = [write_clip(tmp_path / "a.npz"), write_clip(tmp_path / "b.npz", offset=100)]
    motion = SittingMotionLib(files, cpu_kin, "cpu")
    assert motion.clip_len.tolist() == [11, 11]
    time = torch.arange(11) * 0.02
    expected_position = torch.tensor([1., .3, .5]) + time[:, None] * torch.tensor([.4, -.2, .1])
    torch.testing.assert_close(motion.object_p[:11], expected_position)
    torch.testing.assert_close(motion.object_v, torch.tensor([.4, -.2, .1]).expand(22, -1), atol=4e-4, rtol=0)
    torch.testing.assert_close(motion.object_w, torch.tensor([0., 0., .7]).expand(22, -1), atol=2e-5, rtol=0)
    expected_rotation = so3_exp(torch.stack((time * 0, time * 0, .4 + .7 * time), -1))
    torch.testing.assert_close(motion.object_R[:11], expected_rotation, atol=2e-6, rtol=0)
    frame = torch.tensor([0, 11])
    assert len(motion.ref_state(frame)) == 6
    assert SharedDynamics.pack(motion.full_ref_state(frame)).shape == (2, 84)


def test_object_reference_is_required_only_for_interaction(cpu_kin, tmp_path):
    path = write_clip(tmp_path / "robot.npz", with_object=False)
    with pytest.raises(ValueError, match="requires object reference poses"):
        SittingMotionLib([path], cpu_kin, "cpu")
    motion = SittingMotionLib([path], cpu_kin, "cpu", object_enabled=False)
    torch.testing.assert_close(motion.object_R, torch.eye(3).expand(11, -1, -1))
    assert motion.object_p.count_nonzero() == 0
    assert motion.object_w.count_nonzero() == 0


def test_observation_uses_actual_pose_without_reference_or_privileged_physics(interaction):
    motion, mdp = interaction
    f = torch.tensor([1, 3])
    state = motion.full_ref_state(f)
    xi, action = context(), torch.zeros(2, 29)
    actual = mdp.obs(state, action, xi, f)
    assert actual.shape == (2, 99)
    assert mdp.obs_noise.shape == (99,)
    assert mdp.obs_noise[-6:].count_nonzero() == 0
    xi[:, 33:45] += 100
    motion.object_p.add_(10)
    torch.testing.assert_close(actual, mdp.obs(state, action, xi, torch.full_like(f, 10**6)))
    assert (actual[:, -6:-3] == state[6] - state[0]).all()
    xi[:, 45] = 0
    assert mdp.obs(state, action, xi, f)[:, -6:].count_nonzero() == 0


def test_reward_only_adds_object_pose_term_and_preserves_termination(interaction):
    motion, mdp = interaction
    baseline = MDP(mdp.kin, motion, mdp.default_q)
    f = torch.tensor([2, 4])
    state = motion.full_ref_state(f)
    action = torch.linspace(-.1, .1, 58).reshape(2, 29)
    previous = action * .7
    expected, old_terms = baseline.reward(state[:6], f, action, previous, terms=True)
    reward, new_terms = mdp.reward(state, f, action, previous, terms=True)
    torch.testing.assert_close(reward - expected, torch.full((2,), .02))
    for key in old_terms:
        torch.testing.assert_close(new_terms[key], old_terms[key], atol=0, rtol=0)
    assert mdp.max_step_reward == pytest.approx(baseline.max_step_reward + .02)
    torch.testing.assert_close(mdp.termination(state, f), baseline.termination(state[:6], f))
    torch.testing.assert_close(mdp.soft_termination(state, f), baseline.soft_termination(state[:6], f))
    displaced = (*state[:6], state[6] + torch.tensor([.1, 0, 0]), *state[7:])
    torch.testing.assert_close(mdp.reward(displaced, f, action, previous) - expected,
                               torch.full((2,), .02 * np.exp(-1)), atol=1e-7, rtol=1e-5)


def test_object_reward_gradients_pull_pose_toward_reference(interaction):
    motion, mdp = interaction
    f = torch.tensor([1, 2])
    state = motion.full_ref_state(f)
    displacement = torch.tensor([[.05, 0, 0], [.02, 0, 0]], requires_grad=True)
    rotation = torch.tensor([[0, 0, .1], [0, 0, .2]], requires_grad=True)
    changed = (*state[:6], state[6] + displacement, state[7] @ so3_exp(rotation), *state[8:])
    action = torch.zeros(2, 29)
    mdp.reward(changed, f, action, action).sum().backward()
    assert torch.isfinite(displacement.grad).all() and (displacement.grad[:, 0] < 0).all()
    assert torch.isfinite(rotation.grad).all() and (rotation.grad[:, 2] < 0).all()


def test_critic_layout_mask_geometry_detach_and_future_reference(interaction):
    motion, mdp = interaction
    f = torch.tensor([1, 10])
    state = motion.full_ref_state(f)
    xi = context()
    geometry = torch.randn(2, 67, requires_grad=True)
    position = (state[6] + .03).detach().requires_grad_()
    state = (*state[:6], position, *state[7:])
    features = mdp.critic_features(state, f, xi, geometry)
    task = SittingTask("g1_sitting", motion, mdp, SharedDynamics(), None)
    assert features.shape == (2, 285)
    assert task.critic_goal_dim(10) == 790
    goal = task.critic_goal(state, f, 10)
    assert goal.shape == (2, 60)
    full = torch.cat((features, motion.goal(f, 10), goal), -1)
    assert full.shape == (2, 1075)
    full.sum().backward()
    assert position.grad is not None and torch.isfinite(position.grad).all()
    assert geometry.grad is None
    xi[:, 45] = 0
    masked = mdp.critic_features(state, f, xi, geometry)
    assert masked[:, 184:].count_nonzero() == 0
    # Current error is unchanged when only future reference frames change, but
    # the additional critic goal exposes the changed future object objective.
    frame = torch.tensor([1])
    current = motion.full_ref_state(frame)
    before = mdp.critic_features(current, frame, context(1))
    future_before = task.critic_goal(current, frame, 10)
    motion.object_p[2:] += .2
    torch.testing.assert_close(before, mdp.critic_features(current, frame, context(1)))
    assert not torch.allclose(future_before, task.critic_goal(current, frame, 10))


def test_object_future_goal_uses_one_current_heading_and_clamps_clip(interaction):
    motion, _ = interaction
    f = torch.tensor([8, 10])
    state = list(motion.full_ref_state(f))
    state[0] += torch.tensor([.3, -.2, 0])
    state[1] = so3_exp(torch.tensor([[0., 0., .6], [0., 0., -.2]]))
    goal = motion.object_goal(tuple(state), f, 10).reshape(2, 10, 6)
    future = motion.future_frames(f, 10)
    heading = yaw_mat(state[1]).transpose(-1, -2)
    expected = (heading[:, None] @ (motion.object_p[future] - state[0][:, None]).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(goal[..., :3], expected)
    torch.testing.assert_close(goal[0, 1:], goal[0, 1].expand(9, -1))
    torch.testing.assert_close(goal[1], goal[1, 0].expand(10, -1))


def test_shared_tracking_retains_original_task_heads(cpu_kin, tmp_path):
    path = write_clip(tmp_path / "robot.npz", with_object=False)
    motion = SittingMotionLib([path], cpu_kin, "cpu", object_enabled=False)
    mdp = SittingMDP(cpu_kin, motion, torch.tensor(default_joint_pos_list()), object_enabled=False)
    baseline = MDP(cpu_kin, motion, mdp.default_q)
    f = torch.tensor([1, 2])
    state, action, xi = motion.full_ref_state(f), torch.zeros(2, 29), context(present=False)
    torch.testing.assert_close(mdp.obs(state, action, xi, f), baseline.obs(state[:6], action, xi[:, :33], f))
    torch.testing.assert_close(mdp.reward(state, f, action, action), baseline.reward(state[:6], f, action, action))
    assert mdp.max_step_reward == baseline.max_step_reward
    assert mdp.observation_dim == 93 and mdp.critic_feature_dim == 285
    assert motion.object_goal(state, f, 10).count_nonzero() == 0
    metadata = SittingTask("g1_shared_tracking", motion, mdp, SharedDynamics(), None).checkpoint_metadata()
    assert metadata["schema_version"] != 2


@pytest.mark.parametrize("kwargs", [{"weight": -1}, {"position_std": 0}, {"orientation_std": -1}])
def test_invalid_object_reward_config(kwargs):
    with pytest.raises(ValueError):
        ObjectRewardCfg(**kwargs)
