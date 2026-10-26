"""Physical and batching contracts of the shared multi-entity dynamics model."""

import copy
import math

import pytest
import torch

from supermimic.rl.entity_state import EntityDynamics
from supermimic.rl.entity_world_model import EntityGeometryBank, EntityWorldModel, RelationAttention
from supermimic.rotations import so3_exp


def state(n=3, shape=(2,)):
    def rand(*dims):
        return torch.randn(*shape, *dims) * 0.1
    return (rand(3), so3_exp(rand(3)), rand(3), rand(3), rand(29), rand(29),
            rand(n, 3), so3_exp(rand(n, 3)), rand(n, 3), rand(n, 3))


def context(dynamics, shape=(2,)):
    value = torch.zeros(*shape, dynamics.context_dim)
    value[..., 0] = 0.8
    obj = dynamics.object_context(value)
    obj[..., 0] = 10
    obj[..., 4:7] = torch.tensor([0.2, 0.3, 0.4])
    obj[..., 10:] = 1
    dynamics.pair_friction(value)[..., 0] = 0.7
    dynamics.pair_friction(value)[..., 1:] = torch.tensor([0.005, 0.0001])
    dynamics.ground_context(value)[..., 2] = 1
    return value


def geometry_bank():
    bank = EntityGeometryBank()
    points = torch.zeros(1024, 6)
    points[:, :3] = torch.linspace(-0.5, 0.5, 1024 * 3).reshape(1024, 3)
    points[:, 5] = 1
    bank.add("box", points, torch.ones(3))
    large = points.clone()
    large[:, :3] *= 2
    bank.add("large-box", large, torch.full((3,), 2.0))
    return bank


def model(n=3):
    result = EntityWorldModel(EntityDynamics(max_objects=n), geometry_bank(), width=32, layers=2, heads=4)
    # Production heads start at zero. Nonzero heads represent a fitted model
    # when testing its differentiable interaction and equivariance contracts.
    for head in (result.robot_head, result.object_head):
        torch.nn.init.normal_(head[-1].weight, std=0.02)
    return result


def permuted(S, ctx, dynamics, order):
    following = (*S[:6], *(x.index_select(-3 if i == 1 else -2, order) for i, x in enumerate(S[6:])))
    copied = ctx.clone()
    dynamics.object_context(copied).copy_(dynamics.object_context(ctx).index_select(-2, order))
    indices = torch.cat((torch.tensor([0]), order + 1, torch.tensor([len(order) + 1])))
    dynamics.pair_friction(copied).copy_(dynamics.pair_friction(ctx).index_select(-3, indices).index_select(-2, indices))
    return following, copied


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_entity_model_supports_batch_and_replay_window_axes(shape):
    torch.manual_seed(2)
    network = model()
    S, action, ctx = state(shape=shape), torch.randn(*shape, 29), context(network.dynamics, shape)
    following = network.step(S, action, ctx)
    assert network(S, action, ctx).shape == (*shape, 106)
    assert network.dynamics.pack(following).shape == (*shape, 110)
    network.update_normalizers(S, action, ctx, following)
    assert torch.isfinite(network.out_norm.std()).all()
    assert network.out_norm.std().shape == (106,)
    assert network.object_norm.count[0] == 3 * action.numel() // 29


def test_slot_permutations_preserve_robot_prediction_and_permute_object_predictions():
    torch.manual_seed(3)
    network = model()
    S, action, ctx = state(), torch.randn(2, 29), context(network.dynamics)
    # Distinct geometry and pair properties prevent passing via identical slots.
    network.dynamics.object_context(ctx)[:, 1, 12] = 2
    friction = network.dynamics.pair_friction(ctx)
    friction[:, 0, 2, 0] = 1.1
    friction[:, 2, 0, 0] = 1.1
    network.update_normalizers(S, action, ctx, state())
    order = torch.tensor([2, 0, 1])
    rearranged, new_context = permuted(S, ctx, network.dynamics, order)
    predicted = network.predict_delta(S, action, ctx)
    permuted_prediction = network.predict_delta(rearranged, action, new_context)
    torch.testing.assert_close(permuted_prediction[..., :70], predicted[..., :70], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(permuted_prediction[..., 70:].reshape(2, 3, 12),
                               predicted[..., 70:].reshape(2, 3, 12)[:, order], atol=2e-6, rtol=2e-5)
    # The statistics themselves must also be invariant to padding slot order.
    another = model()
    another.load_state_dict(network.state_dict())
    next_state = state()
    next_permuted, _ = permuted(next_state, ctx, network.dynamics, order)
    network.update_normalizers(S, action, ctx, next_state)
    another.update_normalizers(rearranged, action, new_context, next_permuted)
    torch.testing.assert_close(network.object_norm.mean, another.object_norm.mean)
    torch.testing.assert_close(network.out_norm.std(), another.out_norm.std())


def test_ghost_slots_do_not_influence_predictions_or_statistics_and_static_objects_do_not_move():
    torch.manual_seed(4)
    network = model()
    S, action, ctx = state(), torch.randn(2, 29), context(network.dynamics)
    objects = network.dynamics.object_context(ctx)
    objects[:, 1, 11] = 0  # A fixed obstacle still participates in interactions.
    objects[:, 2, 10] = 0  # An absent entity must have no effect.
    dirty_state = list(copy.deepcopy(S))
    for index in range(6, 10):
        dirty_state[index][:, 2] = 1e5
    dirty_context = ctx.clone()
    network.dynamics.object_context(dirty_context)[:, 2, :10] = 1e5
    network.dynamics.object_context(dirty_context)[:, 2, 12] = 999999  # No bank lookup for padding.
    network.dynamics.pair_friction(dirty_context)[:, 3, :, :] = 1e5
    network.dynamics.pair_friction(dirty_context)[:, :, 3, :] = 1e5
    first, second = network(S, action, ctx), network(tuple(dirty_state), action, dirty_context)
    torch.testing.assert_close(first, second)
    cloned = model()
    cloned.load_state_dict(network.state_dict())
    next_state = state()
    dirty_next = list(copy.deepcopy(next_state))
    for index in range(6, 10):
        dirty_next[index][:, 2] = 1e5
    network.update_normalizers(S, action, ctx, next_state)
    cloned.update_normalizers(tuple(dirty_state), action, dirty_context, tuple(dirty_next))
    for name, value in network.named_buffers():
        torch.testing.assert_close(value, dict(cloned.named_buffers())[name])
    assert (network.object_norm.count == 4).all()  # Fixed objects do count as inputs.
    assert (network.out_norm.object.count == 2).all()  # Only moving objects are targets.
    predicted = network.step(S, action, ctx)
    for original, following in zip(S[6:], predicted[6:]):
        torch.testing.assert_close(original[:, 1:], following[:, 1:])


@pytest.mark.parametrize("n", [1, 3])
def test_empty_scene_is_finite_without_object_statistics_or_loss(n):
    network = model(n)
    S, ctx, action = state(n), context(network.dynamics), torch.zeros(2, 29)
    network.dynamics.object_context(ctx)[..., 10] = 0
    network.update_normalizers(S, action, ctx, S)
    prediction = network.step(S, action, ctx)
    assert all(torch.isfinite(x).all() for x in prediction)
    assert (network.object_norm.count == 0).all()
    assert (network.out_norm.object.count == 0).all()
    assert torch.isfinite(network.error_per_sample(prediction, S, ctx)).all()


def test_geometry_size_and_pair_friction_affect_robot_dynamics():
    torch.manual_seed(8)
    network = model()
    S, ctx, action = state(), context(network.dynamics), torch.zeros(2, 29)
    baseline = network.predict_delta(S, action, ctx)[..., :70]
    bigger = ctx.clone()
    network.dynamics.object_context(bigger)[:, 0, 12] = 2
    assert (baseline - network.predict_delta(S, action, bigger)[..., :70]).abs().max() > 1e-6
    slippery = ctx.clone()
    network.dynamics.pair_friction(slippery)[:, 0, 1, 0] = 0.1
    network.dynamics.pair_friction(slippery)[:, 1, 0, 0] = 0.1
    assert (baseline - network.predict_delta(S, action, slippery)[..., :70]).abs().max() > 1e-6


def test_multistep_robot_prediction_backpropagates_through_objects_actions_and_geometry():
    torch.manual_seed(10)
    network = model()
    S, ctx = list(state()), context(network.dynamics)
    S[6].requires_grad_()
    actions = torch.randn(3, 2, 29, requires_grad=True)
    predicted = tuple(S)
    for action in actions:
        predicted = network.step(predicted, action, ctx)
    loss = predicted[0].square().sum() + predicted[4].square().sum()
    gradients = torch.autograd.grad(loss, (actions, S[6], network.geometry_encoder.project.weight,
                                         network.interactions[0].edge_value.weight))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-7


def test_padded_capacity_does_not_change_fitted_weights_or_predictions():
    torch.manual_seed(12)
    small, large = model(1), model(3)
    large.load_state_dict(small.state_dict())
    first, larger = state(1), list(state(3))
    larger[:6] = first[:6]
    for idx in range(6, 10):
        larger[idx][:, :1] = first[idx]
    ctx1, ctx3 = context(small.dynamics), context(large.dynamics)
    large.dynamics.object_context(ctx3)[:, 1:, 10] = 0
    action = torch.randn(2, 29)
    predicted1 = small.predict_delta(first, action, ctx1)
    predicted3 = large.predict_delta(tuple(larger), action, ctx3)
    torch.testing.assert_close(predicted1, predicted3[:, :82], atol=1e-6, rtol=1e-5)


def test_geometry_bank_rejects_position_only_clouds_and_preserves_replay_identity():
    bank = geometry_bank()
    restored = EntityGeometryBank.from_dict(bank.to_dict())
    torch.testing.assert_close(restored.points, bank.points)
    with pytest.raises(ValueError, match="xyz\\+normal"):
        bank.add("old", torch.zeros(1024, 3), torch.ones(3))
    with pytest.raises(ValueError, match="unit length"):
        bank.add("no-normals", torch.zeros(1024, 6), torch.ones(3))
    destination = EntityGeometryBank()
    destination.add("other", bank.points[2], bank.extents[2])
    remap = destination.merge(bank.to_dict())
    torch.testing.assert_close(remap, torch.tensor([0, 2, 3]))
    conflict = bank.to_dict()
    conflict["points"][1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="different content"):
        destination.merge(conflict)
    assert destination.keys == ["__none__", "other", "box", "large-box"]


def test_geometry_codes_are_recomputed_after_encoder_updates():
    network = model()
    ctx = context(network.dynamics)
    before = network.encode_geometry(ctx)
    with torch.no_grad():
        network.geometry_encoder.project.bias.add_(1)
    after = network.encode_geometry(ctx)
    torch.testing.assert_close(after - before, torch.ones_like(after))


def test_object_object_contact_conditions_influence_joint_predictions():
    torch.manual_seed(18)
    network = model()
    S, ctx, action = state(), context(network.dynamics), torch.zeros(2, 29)
    base = network.predict_delta(S, action, ctx)
    changed = ctx.clone()
    # No direct robot/object edge changes: the altered relation is between two
    # objects, and reaches the robot through the second interaction layer.
    network.dynamics.pair_friction(changed)[:, 1, 2, 0] = 1.2
    network.dynamics.pair_friction(changed)[:, 2, 1, 0] = 1.2
    updated = network.predict_delta(S, action, changed)
    assert (updated[..., :70] - base[..., :70]).abs().max() > 1e-7
    assert (updated[..., 70:] - base[..., 70:]).abs().max() > 1e-6


def test_error_averages_dynamic_objects_without_upweighting_crowded_scenes():
    network = model()
    actual, ctx = state(), context(network.dynamics)
    robot_delta = torch.zeros(2, 70)
    robot_delta[:, 0] = 0.1
    delta = torch.cat((robot_delta, torch.full((2, 36), 0.02)), -1)
    predicted = network.dynamics.integrate(actual, delta)
    crowded = network.error_per_sample(predicted, actual, ctx)
    sparse_context = ctx.clone()
    network.dynamics.object_context(sparse_context)[:, 1:, 10] = 0
    sparse = network.error_per_sample(predicted, actual, sparse_context)
    torch.testing.assert_close(crowded, sparse, atol=1e-7, rtol=1e-5)
    network.dynamics.object_context(sparse_context)[:, 0, 10] = 0
    robot_only = network.error_per_sample(predicted, actual, sparse_context)
    expected = (network.state_error(predicted, actual)[..., :70]
                / network.out_norm.std()[..., :70]).square().mean(-1)
    torch.testing.assert_close(robot_only, expected)


def test_compact_relation_values_match_explicit_messages_and_all_gradients():
    """Moving the linear edge projection after pooling preserves the operator."""
    torch.manual_seed(23)
    layer = RelationAttention(32, 4)
    tokens = torch.randn(2, 5, 32, requires_grad=True)
    edges = torch.randn(2, 5, 5, 32, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, True], [True, False, False, False, True]])
    compact = layer(tokens, edges, valid)
    q, k, v = layer.qkv(layer.norm(tokens)).reshape(2, 5, 3, 4, 8).unbind(2)
    logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(8)
    logits = logits + layer.edge_bias(edges).permute(0, 3, 1, 2)
    attention = logits.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
    explicit = layer.edge_value(edges).reshape(2, 5, 5, 4, 8) + v[:, None, :, :, :]
    message = torch.einsum("bhij,bijhd->bihd", attention, explicit).reshape(2, 5, 32)
    reference = tokens + layer.project(message)
    reference = reference + layer.ff(layer.ff_norm(reference))
    reference = torch.where(valid[..., None], reference, torch.zeros_like(reference))
    torch.testing.assert_close(compact, reference, atol=1e-6, rtol=1e-5)
    parameters = (tokens, edges, *layer.parameters())
    compact_grad = torch.autograd.grad(compact.square().mean(), parameters)
    reference_grad = torch.autograd.grad(reference.square().mean(), parameters)
    for actual, expected in zip(compact_grad, reference_grad):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
