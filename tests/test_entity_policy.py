"""Task-network contracts: entities, references, masking and differentiability."""

import pytest
import torch

from supermimic.rl.entity_policy import EntityCritic, EntityPolicy
from supermimic.rl.entity_world_model import EntityGeometryBank


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def geometry_bank():
    bank = EntityGeometryBank()
    points = torch.zeros(1024, 6)
    points[:, :3] = torch.linspace(-0.5, 0.5, 3072).reshape(1024, 3)
    points[:, 5] = 1
    bank.add("chair", points, torch.ones(3))
    bank.add("identical-chair-different-id", points, torch.ones(3))
    different = points.clone()
    different[:, :3] *= 2
    bank.add("table", different, torch.ones(3) * 2)
    return bank


def actor_data(policy, shape=(2,)):
    history = torch.randn(*shape, policy.hist, policy.obs_dim) * 0.2
    objects = history[..., 93:].reshape(*shape, policy.hist, policy.max_objects, 13)
    objects[..., 9] = 1
    objects[..., 10] = 1
    objects[..., 11] = 0
    objects[..., 0, 11] = 1
    objects[..., 12] = 1
    return history, torch.randn(*shape, policy.goal_dim)


def critic_data(critic, shape=(2,)):
    values = torch.randn(*shape, critic.in_dim) * 0.2
    _, objects, friction, plane, _, _ = critic.split_inputs(values)
    objects[..., 25] = 1
    objects[..., 35] = 0
    objects[..., 0, 35] = 1
    objects[..., 36] = 1
    objects[..., 37] = 1
    friction[..., 0] = 0.5
    friction[..., 1:] = 0.01
    plane[..., :] = torch.tensor([0.8, 0, 0, 1])
    return values


def policy(**kwargs):
    return EntityPolicy(geometry_bank(), max_objects=3, hist=2, hidden=(32, 16), **kwargs)


def critic(**kwargs):
    return EntityCritic(geometry_bank(), max_objects=3, goal_frames=3, hidden=(32, 16), **kwargs)


def test_actor_contract_has_only_observed_pose_geometry_and_labels():
    torch.manual_seed(1)
    actor = policy()
    history, goal = actor_data(actor, shape=(2, 3))
    assert actor.obs_dim == 93 + 3 * 13
    assert actor.frame_dim == 221
    assert actor.object_encoder[0].in_features == 75
    assert actor.dec[0].in_features == 221 * actor.hist + 32
    assert actor(history, goal).shape == (2, 3, 29)
    actor.update_normalizers(history, goal)
    assert actor.robot_norm.count == 12
    assert actor.object_pose_norm.count == 36
    assert actor.goal_norm.count == 6


def test_actor_permutation_and_geometry_identity_invariance():
    torch.manual_seed(2)
    actor = policy()
    history, goal = actor_data(actor)
    actor.update_normalizers(history, goal)
    expected = actor(history, goal)
    robot, objects = actor.split_observations(history)
    swapped = torch.cat((robot, objects[..., [2, 0, 1], :].flatten(-2)), -1)
    torch.testing.assert_close(actor(swapped, goal), expected, atol=2e-7, rtol=1e-5)
    renamed = history.clone()
    actor.split_observations(renamed)[1][..., 12] = 2
    torch.testing.assert_close(actor(renamed, goal), expected)


def test_actor_empty_scene_and_padding_are_inert_and_finite():
    actor = policy()
    history, goal = actor_data(actor)
    robot, objects = actor.split_observations(history)
    objects[..., 9] = 0
    expected = actor(history, goal)
    objects[..., :9] = float("nan")
    objects[..., 10:] = float("nan")
    actor.update_normalizers(history, goal)
    assert actor.object_pose_norm.count == 0
    assert torch.equal(actor.encode_scene(actor.robot_norm(robot), objects), torch.zeros(2, 2, 128))
    # Compare after normalizer updates, so only the absent raw fields differ.
    clean = history.clone()
    actor.split_observations(clean)[1][...] = 0
    torch.testing.assert_close(actor(history, goal), actor(clean, goal))
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actor(history, goal)).all()


def test_actor_reencodes_raw_history_and_backpropagates_through_scene():
    torch.manual_seed(6)
    actor = policy()
    history, goal = actor_data(actor)
    history.requires_grad_()
    before = actor(history, goal)
    before.square().sum().backward()
    assert history.grad[..., 93:102].abs().sum() > 0
    encoder_grads = [p.grad for p in actor.geometry_encoder.parameters()]
    assert all(g is None for g in encoder_grads)  # WM alone trains shared geometry.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.object_encoder.parameters())
    with torch.no_grad():
        next(iter(actor.geometry_encoder.parameters())).add_(0.1)
    after = actor(history, goal)
    assert (after - before).abs().max() > 1e-8


def test_critic_object_future_changes_value_and_preserves_input_gradients():
    torch.manual_seed(8)
    value_model = critic()
    values = critic_data(value_model)
    changed = values.clone()
    value_model.split_inputs(changed)[-1][..., 0, 1:, 0] += 2
    assert (value_model(values) - value_model(changed)).abs().max() > 1e-7
    # During actor learning critic parameters freeze, but input gradients must
    # continue through object state, physics and target trajectory encoders.
    value_model.requires_grad_(False)
    values.requires_grad_()
    value_model(values).sum().backward()
    _, object_grad, pair_grad, _, _, future_grad = value_model.split_inputs(values.grad)
    assert object_grad[..., :3].abs().sum() > 0
    assert object_grad[..., 15:25].abs().sum() > 0
    assert pair_grad[..., 1, 2, :].abs().sum() > 0
    assert future_grad[..., 0, :, :].abs().sum() > 0
    assert torch.equal(future_grad[..., 1:, :, :], torch.zeros_like(future_grad[..., 1:, :, :]))
    assert torch.isfinite(values.grad).all()


def test_critic_permutation_includes_references_and_both_friction_axes():
    torch.manual_seed(9)
    value_model = critic()
    values = critic_data(value_model)
    _, objects, friction, _, _, future = value_model.split_inputs(values)
    friction.copy_(torch.rand_like(friction))
    objects[..., 1, 37] = 3
    value_model.update_normalizers(values)
    expected = value_model(values)
    permutation = [2, 0, 1]
    entity_order = [0, 3, 1, 2, 4]
    changed = values.clone()
    _, new_objects, new_friction, _, _, new_future = value_model.split_inputs(changed)
    new_objects.copy_(objects[..., permutation, :])
    new_future.copy_(future[..., permutation, :, :])
    new_friction.copy_(friction[..., entity_order, :, :][..., :, entity_order, :])
    torch.testing.assert_close(value_model(changed), expected, atol=3e-6, rtol=1e-5)


def test_critic_masks_absent_slots_and_non_target_reference_tracks():
    torch.manual_seed(10)
    value_model = critic()
    values = critic_data(value_model)
    _, objects, _, _, _, _ = value_model.split_inputs(values)
    objects[..., 2, 36] = 0
    value_model.update_normalizers(values)
    expected = value_model(values)
    changed = values.clone()
    _, obj, pair, _, _, ref = value_model.split_inputs(changed)
    # Reference information on a non-target must never affect its value token.
    obj[..., 1, 26:35] = float("nan")
    ref[..., 1, :, :] = float("nan")
    obj[..., 2, :] = float("nan")
    obj[..., 2, 36] = 0
    ref[..., 2, :, :] = float("nan")
    pair[..., 3, :, :] = float("nan")
    pair[..., :, 3, :] = float("nan")
    torch.testing.assert_close(value_model(changed), expected)
    assert value_model.object_norm.count == 4
    assert value_model.future_norm.count == 6
    assert value_model.error_norm.count == 2
    assert value_model.friction_norm.count == 2 * (4 * 3)


def test_shared_geometry_parameters_and_metadata_lookup():
    torch.manual_seed(11)
    bank = geometry_bank()
    actor = EntityPolicy(bank, max_objects=3, hist=2, hidden=(16,))
    value_model = EntityCritic(bank, max_objects=3, goal_frames=3, hidden=(16,),
                              geometry_encoder=actor.geometry_encoder)
    assert actor.geometry_bank is value_model.geometry_bank
    actor_params = {id(p) for p in actor.geometry_encoder.parameters()}
    assert actor.geometry_encoder is value_model.geometry_encoder
    assert actor_params == {id(p) for p in value_model.geometry_encoder.parameters()}
    values = critic_data(value_model)
    expected = value_model(values)
    expected.sum().backward()
    assert all(p.grad is None for p in actor.geometry_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value_model.object_encoder.parameters())
    value_model.split_inputs(values)[1][..., 37] = 2
    torch.testing.assert_close(value_model(values), expected)
    assert value_model.net[0].in_features == 1042
    assert value_model.in_dim == value_model.current_dim + 730 + 3 * 3 * 9


def test_critic_empty_scene_keeps_ground_features_and_friction():
    torch.manual_seed(13)
    value_model = critic()
    values = critic_data(value_model)
    _, objects, friction, plane, _, _ = value_model.split_inputs(values)
    objects[..., 36] = 0
    value_model.update_normalizers(values)
    assert value_model.object_norm.count == 0
    assert value_model.future_norm.count == 0
    expected = value_model(values)
    assert torch.isfinite(expected).all()
    plane[..., 0] += 0.2
    friction[..., 0, -1, 0] += 0.2
    friction[..., -1, 0, 0] += 0.2
    assert (value_model(values) - expected).abs().max() > 1e-7


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_invalid_raw_dimensions_raise_useful_errors(kind):
    if kind == "actor":
        net = policy()
        with pytest.raises(ValueError, match="raw observation"):
            net(torch.zeros(2, 2, 93), torch.zeros(2, 730))
    else:
        net = critic()
        with pytest.raises(ValueError, match="critic input"):
            net(torch.zeros(2, 914))
