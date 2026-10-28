"""Sitting coordinate, learning-gradient and reference contracts."""

import numpy as np
import pytest
import torch

from motion_data import save_motion
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.mdp import MDP
from supermimic.rl.geometry import GeometryBank, GeometryEncoder
from supermimic.rotations import mat_to_6d, so3_exp, yaw_mat
from supermimic.tasks.sitting.dynamics import SittingDynamics
from supermimic.tasks.sitting.mdp import ObjectRewardCfg, SittingMDP
from supermimic.tasks.sitting.models import SittingCritic, SittingPolicy, SittingWorldModel
from supermimic.tasks.sitting.motion import SittingMotionLib
from supermimic.tasks.sitting.task import SittingTask


def state(shape=(4,)):
    def rand(n):
        return torch.randn(*shape, n) * 0.2
    return (rand(3), so3_exp(rand(3)), rand(3), rand(3), rand(29), rand(29),
            rand(3), so3_exp(rand(3)), rand(3), rand(3))


def context(shape=(4,), geometry_id=0):
    value = torch.zeros(*shape, 46)
    value[..., 0], value[..., 33] = .8, 10.
    value[..., 37:40] = torch.tensor([.2, .3, .4])
    value[..., 43:45] = torch.tensor([.7, .8])
    value[..., 45] = geometry_id
    return value


def bank_with_chair():
    bank = GeometryBank()
    xyz = torch.linspace(-.5, .5, 1024 * 3).reshape(1024, 3)
    points = torch.cat((xyz, torch.tensor([0., 0., 1.]).expand(1024, -1)), -1)
    bank.add("chair", points, torch.ones(3))
    return bank


def write_clip(path, *, offset=0., with_object=True):
    time = np.arange(6) / 25
    angle = .4 + .7 * time
    clip = dict(
        fps=25., root_pos=np.tile([0., 0., .8], (6, 1)),
        root_quat=np.tile([1., 0., 0., 0.], (6, 1)),
        joint_pos=np.tile(default_joint_pos_list(), (6, 1)), joint_names=np.array(G1_JOINT_NAMES),
    )
    if with_object:
        clip.update(object_pos=np.array([1. + offset, .3, .5]) + time[:, None] * [.4, -.2, .1],
                    object_quat=np.stack((np.cos(angle / 2), time * 0, time * 0, np.sin(angle / 2)), -1),
                    object_scale=np.ones(3))
    save_motion(path, clip)
    return path


@pytest.fixture
def interaction(cpu_kin, tmp_path):
    motion = SittingMotionLib([write_clip(tmp_path / "motion.npz")], cpu_kin, "cpu")
    mdp = SittingMDP(cpu_kin, motion, torch.tensor(default_joint_pos_list()))
    return motion, mdp


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
def test_state_integration_and_pack_roundtrip(shape):
    torch.manual_seed(31)
    dynamics = SittingDynamics()
    initial, delta = state(shape), torch.randn(*shape, 82) * .02
    final = dynamics.integrate(initial, delta)
    torch.testing.assert_close(dynamics.targets(initial, final), delta, atol=1e-6, rtol=1e-4)
    packed = dynamics.pack(final)
    assert packed.shape == (*shape, 84)
    for restored, expected in zip(dynamics.unpack(packed), final):
        torch.testing.assert_close(restored, expected, atol=1e-6, rtol=1e-4)


def test_dynamics_invariant_to_scene_translation_and_yaw():
    torch.manual_seed(8)
    initial, final = state(), state()
    rotation, shift = so3_exp(torch.tensor([0., 0., 1.2])), torch.tensor([3., -7., 0.])

    def transformed(S):
        result = list(S)
        for start in (0, 6):
            result[start] = (rotation @ S[start][..., None]).squeeze(-1) + shift
            result[start + 1] = rotation @ S[start + 1]
            for i in (start + 2, start + 3):
                result[i] = (rotation @ S[i][..., None]).squeeze(-1)
        return tuple(result)

    dynamics = SittingDynamics()
    torch.testing.assert_close(dynamics.features(initial), dynamics.features(transformed(initial)),
                               atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(dynamics.targets(initial, final),
                               dynamics.targets(transformed(initial), transformed(final)), atol=2e-6, rtol=1e-5)


def test_geometry_metadata_roundtrip_and_invalid_assets():
    bank = bank_with_chair()
    restored = GeometryBank.from_dict(bank.state_dict())
    assert restored.keys == ["chair"]
    torch.testing.assert_close(restored.points, bank.points)
    assert restored.add("chair", bank.points[0], bank.extents[0]) == 0
    with pytest.raises(ValueError, match="different content"):
        restored.add("chair", bank.points[0], bank.extents[0] * 2)
    with pytest.raises(ValueError, match="unit length"):
        restored.add("bad", bank.points[0] * 2, bank.extents[0])
    with pytest.raises(ValueError, match="version 3"):
        GeometryBank.from_dict({**bank.state_dict(), "format_version": 2})


def test_world_model_has_no_geometry_id_or_embedding_statistics():
    bank = bank_with_chair()
    bank.add("same-shape-new-name", bank.points[0], bank.extents[0])
    model = SittingWorldModel(bank, hidden=(16,))
    initial, action, ctx = state((2, 3)), torch.zeros(2, 3, 29), context((2, 3))
    first = model.inputs(initial, action, model.dynamics.normalize_context(ctx))
    ctx[..., 45] = 1
    second = model.inputs(initial, action, model.dynamics.normalize_context(ctx))
    assert first.shape == (2, 3, 224)
    torch.testing.assert_close(first, second)
    model.update_normalizers(initial, action, ctx, initial)
    assert model.in_norm.mean.shape == (160,)
    assert model.out_norm.mean.shape == (82,)
    assert model.in_norm.count == 6


def test_shared_geometry_updates_only_through_world_model_and_has_no_stale_cache():
    torch.manual_seed(17)
    bank, encoder = bank_with_chair(), GeometryEncoder()
    policy = SittingPolicy(bank, geometry_encoder=encoder, hidden=(16,))
    wm = SittingWorldModel(bank, geometry_encoder=encoder, hidden=(16,))
    assert policy.geometry_encoder is wm.geometry_encoder
    history, goal = torch.randn(4, 10, 102), torch.randn(4, 730)
    before = policy(history, goal)
    assert before.shape == (4, 29) and policy.dec[0].in_features == 1116
    before.square().sum().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert policy.dec[0].weight.grad.abs().sum() > 0

    torch.nn.init.normal_(wm.net[-1].weight, std=.02)
    initial = state()
    action = torch.randn(4, 29, requires_grad=True)
    normalized = wm.dynamics.normalize_context(context())
    following = wm.step(initial, action, normalized)
    loss = following[0].square().sum() + following[4].square().sum()
    action_grad, geometry_grad = torch.autograd.grad(loss, (action, encoder.project.weight))
    assert torch.isfinite(action_grad).all() and action_grad.abs().sum() > 1e-7
    assert torch.isfinite(geometry_grad).all() and geometry_grad.abs().sum() > 1e-7
    with torch.no_grad():
        encoder.project.bias.add_(.5)
    assert not torch.allclose(before, policy(history, goal))

    # Freezing learned physics must retain its derivative with respect to actions.
    wm.requires_grad_(False)
    frozen = wm.step(initial, action, normalized)
    gradient, = torch.autograd.grad(frozen[0].square().sum(), action)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-7


def test_policy_requires_asset_identity_for_multiple_chairs():
    bank = bank_with_chair()
    bank.add("second-chair", bank.points[0], bank.extents[0] * 2)
    policy = SittingPolicy(bank, hidden=(16,))
    history, goal = torch.zeros(2, 10, 102), torch.zeros(2, 730)
    with pytest.raises(ValueError, match="requires geometry IDs"):
        policy(history, goal)
    output = policy(history, goal, torch.tensor([0, 1]))
    assert not torch.allclose(output[0], output[1])


def test_object_reference_clock_and_clip_boundary_velocity(cpu_kin, tmp_path):
    files = [write_clip(tmp_path / "a.npz"), write_clip(tmp_path / "b.npz", offset=100)]
    motion = SittingMotionLib(files, cpu_kin, "cpu")
    assert motion.clip_len.tolist() == [11, 11]
    time = torch.arange(11) * .02
    expected = torch.tensor([1., .3, .5]) + time[:, None] * torch.tensor([.4, -.2, .1])
    torch.testing.assert_close(motion.object_p[:11], expected)
    torch.testing.assert_close(motion.object_v, torch.tensor([.4, -.2, .1]).expand(22, -1), atol=4e-4, rtol=0)
    torch.testing.assert_close(motion.object_w, torch.tensor([0., 0., .7]).expand(22, -1), atol=2e-5, rtol=0)
    with pytest.raises(ValueError, match="requires object reference poses"):
        SittingMotionLib([write_clip(tmp_path / "robot.npz", with_object=False)], cpu_kin, "cpu")


def test_actor_observation_excludes_reference_and_privileged_physics(interaction):
    motion, mdp = interaction
    frame = torch.tensor([1, 3])
    initial, ctx, action = motion.full_ref_state(frame), context((2,)), torch.zeros(2, 29)
    actual = mdp.obs(initial, action, ctx, frame)
    assert actual.shape == (2, 102) and mdp.obs_noise.shape == (102,)
    assert mdp.obs_noise[-9:].count_nonzero() == 0
    ctx[:, 33:] += 100
    motion.object_p.add_(10)
    torch.testing.assert_close(actual, mdp.obs(initial, action, ctx, torch.full_like(frame, 10**6)))
    torch.testing.assert_close(actual[:, -9:-6], initial[6] - initial[0])
    torch.testing.assert_close(actual[:, -6:], mat_to_6d(initial[7]))


def test_sitting_adds_only_object_reward_and_preserves_tracking_termination(interaction):
    motion, mdp = interaction
    baseline = MDP(mdp.kin, motion, mdp.default_q)
    frame = torch.tensor([2, 4])
    initial, action = motion.full_ref_state(frame), torch.linspace(-.1, .1, 58).reshape(2, 29)
    expected, old_terms = baseline.reward(initial[:6], frame, action, action * .7, terms=True)
    reward, terms = mdp.reward(initial, frame, action, action * .7, terms=True)
    torch.testing.assert_close(reward - expected, torch.full((2,), .02))
    for key in old_terms:
        torch.testing.assert_close(terms[key], old_terms[key], atol=0, rtol=0)
    torch.testing.assert_close(mdp.termination(initial, frame), baseline.termination(initial[:6], frame))
    torch.testing.assert_close(mdp.soft_termination(initial, frame), baseline.soft_termination(initial[:6], frame))
    displacement = torch.tensor([[.05, 0, 0], [.02, 0, 0]], requires_grad=True)
    rotation = torch.tensor([[0, 0, .1], [0, 0, .2]], requires_grad=True)
    changed = (*initial[:6], initial[6] + displacement, initial[7] @ so3_exp(rotation), *initial[8:])
    mdp.reward(changed, frame, action, action).sum().backward()
    assert (displacement.grad[:, 0] < 0).all() and (rotation.grad[:, 2] < 0).all()


def test_critic_dimensions_geometry_detach_and_state_gradient(interaction):
    motion, mdp = interaction
    task = SittingTask("g1_sitting", motion, mdp, SittingDynamics(), None)
    frame, ctx = torch.tensor([1, 3]), context((2,))
    initial = motion.full_ref_state(frame)
    position = (initial[6] + .03).detach().requires_grad_()
    initial = (*initial[:6], position, *initial[7:])
    geometry = torch.randn(2, 64, requires_grad=True)
    features = task.critic_input(initial, frame, ctx, motion.goal(frame, 10), 10, geometry)
    assert task.dimensions.critic_features == 281 and features.shape == (2, 1101)
    critic = SittingCritic()
    critic.update_normalizers(features)
    assert critic.norm.mean.shape == (1037,)
    critic.requires_grad_(False)
    critic(features).sum().backward()
    assert position.grad is not None and torch.isfinite(position.grad).all()
    assert position.grad.abs().sum() > 0 and geometry.grad is None
    metadata = task.checkpoint_metadata()
    assert metadata["dimensions"]["context"] == 46
    assert metadata["geometry"]["latent"] == 64
    assert metadata["critic_object_goal_per_frame"] == 9


def test_object_future_goal_uses_actual_heading_clamps_and_exposes_future(interaction):
    motion, mdp = interaction
    frame = torch.tensor([8, 10])
    initial = list(motion.full_ref_state(frame))
    initial[0] += torch.tensor([.3, -.2, 0])
    initial[1] = so3_exp(torch.tensor([[0., 0., .6], [0., 0., -.2]]))
    goal = motion.object_goal(tuple(initial), frame, 10).reshape(2, 10, 9)
    future, heading = motion.future_frames(frame, 10), yaw_mat(initial[1]).transpose(-1, -2)
    expected = (heading[:, None] @ (motion.object_p[future] - initial[0][:, None]).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(goal[..., :3], expected)
    torch.testing.assert_close(goal[..., 3:], mat_to_6d(heading[:, None] @ motion.object_R[future]))
    torch.testing.assert_close(goal[0, 1:], goal[0, 1].expand(9, -1))
    torch.testing.assert_close(goal[1], goal[1, 0].expand(10, -1))
    frame = torch.tensor([1])
    initial = motion.full_ref_state(frame)
    before = mdp.critic_features(initial, frame, context((1,)))
    future_before = motion.object_goal(initial, frame, 10)
    motion.object_p[2:] += .2
    torch.testing.assert_close(before, mdp.critic_features(initial, frame, context((1,))))
    assert not torch.allclose(future_before, motion.object_goal(initial, frame, 10))


@pytest.mark.parametrize("kwargs", [{"weight": -1}, {"position_std": 0}, {"orientation_std": -1}])
def test_invalid_object_reward_config(kwargs):
    with pytest.raises(ValueError):
        ObjectRewardCfg(**kwargs)
