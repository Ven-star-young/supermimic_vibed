"""Open-door state coordinates and gradients use the shared tracking models."""

import math

import pytest
import torch

from supermimic.mdp.state import G1Dynamics
from supermimic.rotations import so3_exp
from supermimic.tasks.open_door.dynamics import DoorDynamics
from supermimic.tasks.open_door.models import Critic, Policy, WorldModel


def state(shape=(4,)):
    def rand(n):
        return torch.randn(*shape, n) * .2
    return (rand(3), so3_exp(rand(3)), rand(3), rand(3), rand(29), rand(29), rand(1), rand(1))


def context(shape=(4,)):
    value = torch.zeros(*shape, 46)
    value[..., 0] = .8
    value[..., 33:36] = torch.tensor([-.127574, .132628, .92])
    value[..., 36:] = torch.tensor([.55, 1.8, .04, 8., 5., .5, 1., .7, 20., -.1])
    return value


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
def test_hinge_state_roundtrip_and_local_delta(shape):
    torch.manual_seed(31)
    dynamics = DoorDynamics()
    initial, delta = state(shape), torch.randn(*shape, 72) * .02
    final = dynamics.integrate(initial, delta)
    torch.testing.assert_close(dynamics.targets(initial, final), delta, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(dynamics.state_error(final, initial), delta, atol=1e-6, rtol=1e-4)
    packed = dynamics.pack(final)
    assert packed.shape == (*shape, 73)
    for restored, expected in zip(dynamics.unpack(packed), final):
        torch.testing.assert_close(restored, expected, atol=1e-6, rtol=1e-4)
    assert dynamics.error_floor("cpu").shape == (72,)


def test_door_features_follow_negative_asset_hinge_without_flipping_joint_angle():
    initial = list(state((1,)))
    initial[1] = torch.eye(3).unsqueeze(0)
    initial[6] = torch.tensor([[math.pi / 2]])
    initial[7] = torch.tensor([[.3]])
    feature = DoorDynamics().door_features(tuple(initial), context((1,)))
    torch.testing.assert_close(feature[:, 3:6], torch.tensor([[0., -1., 0.]]), atol=1e-7, rtol=0)
    torch.testing.assert_close(feature[:, 6:], torch.tensor([[1., 0., .3]]), atol=1e-7, rtol=0)
    rotated = DoorDynamics(base_yaw=math.pi / 2).door_features(tuple(initial), context((1,)))
    torch.testing.assert_close(rotated[:, 3:6], torch.tensor([[1., 0., 0.]]), atol=1e-7, rtol=0)


def test_door_features_are_invariant_when_the_whole_asset_frame_moves():
    torch.manual_seed(8)
    initial, final, ctx = state(), state(), context()
    angle, shift = 1.2, torch.tensor([3., -7., 0.])
    rotation = so3_exp(torch.tensor([0., 0., angle]))

    def transformed(S):
        return ((rotation @ S[0][..., None]).squeeze(-1) + shift,
                rotation @ S[1],
                (rotation @ S[2][..., None]).squeeze(-1),
                (rotation @ S[3][..., None]).squeeze(-1), *S[4:])

    transformed_context = ctx.clone()
    transformed_context[:, 33:36] = (rotation @ ctx[:, 33:36, None]).squeeze(-1) + shift
    dynamics, rotated = DoorDynamics(base_yaw=.3), DoorDynamics(base_yaw=.3 + angle)
    torch.testing.assert_close(dynamics.features(initial, ctx),
                               rotated.features(transformed(initial), transformed_context),
                               atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(dynamics.targets(initial, final),
                               rotated.targets(transformed(initial), transformed(final)),
                               atol=2e-6, rtol=1e-5)


def test_world_model_context_and_action_gradient():
    torch.manual_seed(17)
    dynamics = DoorDynamics()
    wm = WorldModel(dynamics=dynamics, hidden=(16,))
    initial, ctx = state((2, 3)), context((2, 3))
    normalized = dynamics.normalize_context(ctx)
    torch.testing.assert_close(normalized[..., 33:36], ctx[..., 33:36])
    torch.testing.assert_close(normalized[..., :33], G1Dynamics.normalize_context(ctx[..., :33]))
    action = torch.randn(2, 3, 29, requires_grad=True)
    inputs = wm.inputs(initial, action, normalized)
    assert inputs.shape == (2, 3, 152)
    torch.testing.assert_close(inputs[..., 109:142], normalized[..., :33])
    torch.testing.assert_close(inputs[..., 142:], normalized[..., 36:])
    wm.update_normalizers(initial, action, ctx, initial)
    assert wm.in_norm.count == 6 and wm.out_norm.mean.shape == (72,)
    assert wm.encode_geometry(normalized) is None
    torch.nn.init.normal_(wm.net[-1].weight, std=.02)
    wm.requires_grad_(False)
    following = wm.step(initial, action, normalized)
    gradient, = torch.autograd.grad(following[6].sum() + following[0].square().sum(), action)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-7


def test_policy_and_critic_use_joint_future_goal_contract():
    policy, critic = Policy(), Critic()
    history, goal = torch.randn(2, 10, 102), torch.randn(2, 750)
    assert policy.enc[0].in_features == 750 and policy.dec[0].in_features == 1052
    assert policy(history, goal, geometry_ids=None).shape == (2, 29)
    assert critic.in_dim == 955 and critic(torch.randn(2, 955)).shape == (2,)
