"""Coordinate, rotation and context contracts for robot/entity replay states."""

import math

import pytest
import torch

from supermimic.rl.entity_state import EntityDynamics
from supermimic.rotations import so3_exp


def sample_state(n=3, shape=(2,)):
    rand = lambda *dims: torch.randn(*shape, *dims) * .2
    return (rand(3), so3_exp(rand(3)), rand(3), rand(3), rand(29), rand(29),
            rand(n, 3), so3_exp(rand(n, 3)), rand(n, 3), rand(n, 3))


def context(dynamics, shape=(2,)):
    result = dynamics.empty_context(*shape)
    objects = dynamics.object_context(result)
    objects[..., 0] = 10
    objects[..., 4:7] = .1
    objects[..., 10:] = 1
    dynamics.pair_friction(result)[..., 0] = .7
    return result


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_packed_state_and_rotation_integration_roundtrip_in_all_batch_shapes(shape):
    torch.manual_seed(30)
    dynamics = EntityDynamics(3)
    initial = sample_state(shape=shape)
    delta = torch.randn(*shape, dynamics.output_dim) * .02
    following = dynamics.integrate(initial, delta)
    torch.testing.assert_close(dynamics.targets(initial, following), delta, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(dynamics.state_error(following, initial), delta, atol=1e-6, rtol=1e-4)
    packed = dynamics.pack(following)
    assert packed.shape == (*shape, 110)
    for actual, expected in zip(dynamics.unpack(packed), following):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    assert dynamics.error_floor("cpu").shape == (106,)


def test_directed_edges_are_in_receiver_body_frame_with_correct_rotation_columns():
    dynamics = EntityDynamics(2)
    S = list(sample_state(2, ()))
    S[0] = torch.tensor([0., 0., 1.])
    S[1] = so3_exp(torch.tensor([0., 0., math.pi / 2]))
    S[2] = torch.tensor([.5, 0., 0.])
    S[3] = torch.zeros(3)
    S[6] = torch.tensor([[1., 0., 1.], [2., 1., .4]])
    S[7] = torch.eye(3).expand(2, 3, 3)
    S[8] = torch.tensor([[1.5, 0., 0.], [0., 0., 0.]])
    S[9] = torch.zeros(2, 3)
    ctx = context(dynamics, ())
    edges, ground, is_ground = dynamics.relation_features(tuple(S), ctx)
    torch.testing.assert_close(edges[0, 1, :3], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(edges[0, 1, 3:9], torch.tensor([0., 1., -1., 0., 0., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(edges[0, 1, 9:12], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(edges[1, 0, :3], torch.tensor([-1., 0., 0.]))
    torch.testing.assert_close(edges[1, 2, :3], torch.tensor([1., 1., -.6]))
    torch.testing.assert_close(ground[0, -1, :4], torch.tensor([1., 0., 0., 1.]))
    torch.testing.assert_close(ground[-1, 0], ground[0, -1])
    torch.testing.assert_close(edges[..., -3:], dynamics.pair_friction(ctx))
    assert is_ground[-1].all() and is_ground[:, -1].all()
    assert not is_ground[:-1, :-1].any()


def test_tilted_plane_features_and_cross_entity_relations_ignore_global_xy_yaw():
    torch.manual_seed(32)
    dynamics = EntityDynamics(3)
    initial, following = sample_state(), sample_state()
    ctx = context(dynamics)
    normal = torch.tensor([.3, -.4, math.sqrt(.75)])
    dynamics.ground_context(ctx)[..., :3] = normal
    dynamics.ground_context(ctx)[..., 3] = .25
    rotation = so3_exp(torch.tensor([0., 0., 1.1]))
    shift = torch.tensor([3., -2., 0.])

    def moved(S):
        result = list(S)
        for start in (0, 6):
            result[start] = (rotation @ S[start][..., None]).squeeze(-1) + shift
            result[start + 1] = rotation @ S[start + 1]
            for index in (start + 2, start + 3):
                result[index] = (rotation @ S[index][..., None]).squeeze(-1)
        return tuple(result)

    moved_context = ctx.clone()
    moved_normal = rotation @ normal
    dynamics.ground_context(moved_context)[..., :3] = moved_normal
    dynamics.ground_context(moved_context)[..., 3] += moved_normal @ shift
    dynamics.validate_context(moved_context, geometry_count=2)
    torch.testing.assert_close(dynamics.object_features(initial), dynamics.object_features(moved(initial)),
                               atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(dynamics.ground_features(initial, ctx),
                               dynamics.ground_features(moved(initial), moved_context), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(dynamics.targets(initial, following), dynamics.targets(moved(initial), moved(following)),
                               atol=2e-6, rtol=1e-5)
    edges, ground, is_ground = dynamics.relation_features(initial, ctx)
    transformed_edges, transformed_ground, _ = dynamics.relation_features(moved(initial), moved_context)
    torch.testing.assert_close(edges[..., :-1, :-1, :], transformed_edges[..., :-1, :-1, :], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(ground[..., is_ground, :], transformed_ground[..., is_ground, :],
                               atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("invalid", ["mask", "mass", "identity", "friction", "plane"])
def test_invalid_physics_context_is_rejected_before_replay_or_training(invalid):
    dynamics = EntityDynamics(3)
    ctx = context(dynamics)
    objects = dynamics.object_context(ctx)
    if invalid == "mask":
        objects[:, 0, 10] = .5
    elif invalid == "mass":
        objects[:, 0, 0] = 0
    elif invalid == "identity":
        objects[:, 0, 12] = 0
    elif invalid == "friction":
        dynamics.pair_friction(ctx)[:, 0, 1, 0] = 1.2
    else:
        dynamics.ground_context(ctx)[:, 2] = .5
    with pytest.raises(ValueError):
        dynamics.validate_context(ctx, geometry_count=2)


def test_context_schema_views_preserve_masks_metadata_and_distinct_pair_properties():
    dynamics = EntityDynamics(8)
    ctx = context(dynamics, (2, 3))
    objects = dynamics.object_context(ctx)
    objects[..., 3:, 10:13] = 0
    pairs = dynamics.pair_friction(ctx)
    pairs[..., 0, 1, 0] = pairs[..., 1, 0, 0] = .3
    pairs[..., 1, 2, 0] = pairs[..., 2, 1, 0] = 1.2
    dynamics.validate_context(ctx, geometry_count=2)
    assert dynamics.normalize_context(ctx) is ctx
    assert (dynamics.object_context(ctx)[..., 3:, 10:13] == 0).all()
    assert (dynamics.pair_friction(ctx)[..., 0, 1, 0] == .3).all()
    assert (dynamics.pair_friction(ctx)[..., 1, 2, 0] == 1.2).all()
    assert (dynamics.ground_context(ctx)[..., 2] == 1).all()


def test_weight_transfer_ignores_only_capacity_and_requires_identical_physics_semantics():
    small, large = EntityDynamics(2), EntityDynamics(8)
    assert small.compatible_physics_schema(large.schema())
    assert large.compatible_physics_schema(small.schema())
    for field, wrong in (("control_dt", .04), ("action_dim", 28), ("quaternion", "xyzw"),
                         ("version", 999), ("geometry", "positions_without_normals"),
                         ("object_delta", "global_frame")):
        incompatible = small.schema() | {field: wrong}
        assert not large.compatible_physics_schema(incompatible)
    assert not large.compatible_physics_schema(None)
