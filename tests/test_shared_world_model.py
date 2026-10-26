"""Physical invariants and cross-task reuse of the shared dynamics contract."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.rl.shared_world_model import GeometryBank, MaskedRunningNorm, SharedDynamics, SharedWorldModel
from supermimic.rotations import so3_exp


def state(shape=(4,)):
    def rand(n):
        return torch.randn(*shape, n) * 0.2
    return (rand(3), so3_exp(rand(3)), rand(3), rand(3), rand(29), rand(29),
            rand(3), so3_exp(rand(3)), rand(3), rand(3))


def context(shape=(4,), geometry_id=1):
    value = torch.zeros(*shape, 47)
    value[..., 0] = 0.8
    value[..., 33] = 10
    value[..., 37:40] = torch.tensor([0.2, 0.3, 0.4])
    value[..., 43:46] = torch.tensor([0.7, 0.8, 1.0])
    value[..., 46] = geometry_id
    return value


def bank_with_box(key="box", size=1.0):
    bank = GeometryBank()
    # Deterministic distinct shape samples, with no stochastic preprocessing.
    points = torch.linspace(-size / 2, size / 2, 1024 * 3).reshape(1024, 3)
    bank.add(key, points, torch.full((3,), size))
    return bank


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
def test_state_pack_and_frame_local_integration_roundtrip(shape):
    torch.manual_seed(31)
    dynamics = SharedDynamics()
    initial = state(shape)
    delta = torch.randn(*shape, 82) * 0.02
    final = dynamics.integrate(initial, delta)
    torch.testing.assert_close(dynamics.targets(initial, final), delta, atol=1e-6, rtol=1e-4)
    packed = dynamics.pack(final)
    assert packed.shape == (*shape, 84)
    for restored, expected in zip(dynamics.unpack(packed), final):
        torch.testing.assert_close(restored, expected, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(dynamics.state_error(final, initial), delta, atol=1e-6, rtol=1e-4)
    assert dynamics.error_floor("cpu").shape == (82,)


def test_features_and_targets_are_invariant_to_scene_xy_and_yaw():
    torch.manual_seed(8)
    initial, final = state(), state()
    rotation = so3_exp(torch.tensor([0., 0., 1.2]))
    shift = torch.tensor([3., -7., 0.])

    def transformed(S):
        result = list(S)
        for start in (0, 6):
            result[start] = (rotation @ S[start][..., None]).squeeze(-1) + shift
            result[start + 1] = rotation @ S[start + 1]
            for i in (start + 2, start + 3):
                result[i] = (rotation @ S[i][..., None]).squeeze(-1)
        return tuple(result)

    dynamics = SharedDynamics()
    torch.testing.assert_close(dynamics.features(initial), dynamics.features(transformed(initial)),
                               atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(dynamics.targets(initial, final),
                               dynamics.targets(transformed(initial), transformed(final)),
                               atol=2e-6, rtol=1e-5)


def test_masked_statistics_exclude_absent_object_samples():
    norm = MaskedRunningNorm(2)
    norm.update(torch.tensor([[[1., 10.], [3., 1000.]]]), torch.tensor([[[1., 1.], [1., 0.]]]))
    norm.update(torch.tensor([[5., 14.]]), torch.ones(1, 2))
    torch.testing.assert_close(norm.count, torch.tensor([3., 2.]))
    torch.testing.assert_close(norm.mean, torch.tensor([3., 12.]))
    torch.testing.assert_close(norm.var, torch.tensor([8 / 3, 4.]))


def test_no_object_slots_have_no_statistics_loss_or_state_motion():
    torch.manual_seed(11)
    model = SharedWorldModel(geometry_bank=bank_with_box(), hidden=(32,))
    initial = state()
    following = model.dynamics.integrate(initial, torch.randn(4, 82) * 0.03)
    ctx = context()
    ctx[2:, 45:] = 0
    action = torch.randn(4, 29)
    model.update_normalizers(initial, action, ctx, following)
    assert (model.in_norm.count[:71] == 4).all()
    assert (model.in_norm.count[71:86] == 2).all()
    assert (model.in_norm.count[148:227] == 2).all()
    assert model.in_norm.count[-1] == 0
    assert (model.out_norm.count[:70] == 4).all()
    assert (model.out_norm.count[70:] == 2).all()
    xi = model.dynamics.normalize_context(ctx)
    inputs = model.normalized_inputs(initial, action, xi)
    assert (inputs[2:, 71:86] == 0).all()
    assert (inputs[2:, 148:227] == 0).all()
    torch.testing.assert_close(inputs[:, -1], ctx[:, 45])
    assert (model.loss_mask(ctx)[2:, 70:] == 0).all()
    predicted = model.step(initial, action, xi)
    for before, after in zip(initial[6:], predicted[6:]):
        torch.testing.assert_close(before[2:], after[2:])


def test_shared_model_inputs_support_window_batches_without_metadata_leakage():
    torch.manual_seed(14)
    bank = bank_with_box()
    # Same geometry under a second ID must produce exactly the same network input.
    bank.add("other-name", bank.points[1], bank.extents[1])
    model = SharedWorldModel(geometry_bank=bank, hidden=(16,))
    initial, action, ctx = state((2, 3)), torch.zeros(2, 3, 29), context((2, 3))
    first = model.normalized_inputs(initial, action, model.dynamics.normalize_context(ctx))
    ctx[..., 46] = 2
    second = model.normalized_inputs(initial, action, model.dynamics.normalize_context(ctx))
    assert first.shape == (2, 3, 228)
    torch.testing.assert_close(first, second)
    assert model(initial, action, model.dynamics.normalize_context(ctx)).shape == (2, 3, 82)
    model.update_normalizers(initial, action, ctx, initial)
    assert (model.out_norm.count == 6).all()


def test_object_interactions_and_shape_encoder_have_gradient_paths():
    torch.manual_seed(17)
    model = SharedWorldModel(geometry_bank=bank_with_box(), hidden=(32, 32))
    # The zero-initialized production head blocks gradients intentionally. Use a
    # nonzero head to test a fitted model's differentiable transition interface.
    torch.nn.init.normal_(model.net[-1].weight, std=0.02)
    initial = list(state())
    initial[6].requires_grad_()
    action = torch.randn(4, 29, requires_grad=True)
    xi = model.dynamics.normalize_context(context())
    predicted = model.step(tuple(initial), action, xi)
    robot_loss = predicted[0].square().sum() + predicted[4].square().sum()
    grads = torch.autograd.grad(robot_loss, (action, initial[6], model.geometry_encoder.project.weight))
    for gradient in grads:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-7


def test_geometry_bank_merge_preserves_replay_identity_and_rejects_conflicts():
    source = bank_with_box("chair", 1.0)
    restored = GeometryBank.from_dict(source.to_dict())
    assert restored.keys == source.keys
    torch.testing.assert_close(restored.points, source.points)
    destination = bank_with_box("table", 2.0)
    remap = destination.merge(source.state_dict())
    torch.testing.assert_close(remap, torch.tensor([0, 2]))
    torch.testing.assert_close(destination.points[remap], source.points)
    assert destination.add("chair", source.points[1], source.extents[1]) == 2
    with pytest.raises(ValueError, match="different content"):
        destination.merge(bank_with_box("chair", 1.5).to_dict())
    assert len(destination.keys) == 3
    destination.load_state_dict(source.state_dict())
    assert destination.keys == source.keys


def test_model_checkpoint_reuses_weights_across_different_runtime_geometry_ids():
    torch.manual_seed(22)
    source_bank = bank_with_box("chair")
    source = SharedWorldModel(geometry_bank=source_bank, hidden=(32,))
    initial, final = state(), state()
    action, ctx = torch.randn(4, 29), context()
    source.update_normalizers(initial, action, ctx, final)
    torch.nn.init.normal_(source.net[-1].weight, std=0.01)
    target_bank = bank_with_box("different-scene", 2.0)
    new_id = target_bank.merge(source_bank.to_dict())[1]
    target = SharedWorldModel(geometry_bank=target_bank, hidden=(32,))
    target.load_state_dict(source.state_dict())
    remapped = ctx.clone()
    remapped[..., 46] = new_id
    predicted = source.step(initial, action, source.dynamics.normalize_context(ctx))
    reused = target.step(initial, action, target.dynamics.normalize_context(remapped))
    for first, second in zip(predicted, reused):
        torch.testing.assert_close(first, second)
    assert not any("task" in key for key in source.dynamics.schema())


def test_compiled_transition_keeps_eager_values_and_action_gradients():
    torch.manual_seed(28)
    model = SharedWorldModel(geometry_bank=bank_with_box(), hidden=(16,))
    torch.nn.init.normal_(model.net[-1].weight, std=0.02)
    initial = state()
    action = torch.randn(4, 29, requires_grad=True)
    xi = model.dynamics.normalize_context(context())
    eager = model.step(initial, action, xi)
    # Exercise Dynamo tracing and graph breaks, without claiming an Inductor/CUDA
    # performance result from this small CPU contract test.
    compiled = torch.compile(model.step, backend="eager")(initial, action, xi)
    for before, after in zip(eager, compiled):
        torch.testing.assert_close(before, after)
    eager_grad, = torch.autograd.grad(eager[0].square().sum(), action)
    compiled_grad, = torch.autograd.grad(compiled[0].square().sum(), action)
    torch.testing.assert_close(eager_grad, compiled_grad)


def test_sim_adapter_reads_link_origin_velocities_and_offsets_world_positions():
    data = SimpleNamespace(root_link_pos_w=torch.tensor([[12., 2., 0.7]]),
                           root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                           root_link_lin_vel_w=torch.tensor([[0.1, 0.2, 0.3]]),
                           root_link_ang_vel_w=torch.tensor([[0.4, 0.5, 0.6]]),
                           joint_pos=torch.zeros(1, 29), joint_vel=torch.ones(1, 29))

    class Scene(dict):
        env_origins = torch.tensor([[10., 0., 0.]])
        @property
        def entities(self):
            return self

    scene = Scene(chair=SimpleNamespace(data=data))
    command = SimpleNamespace(robot=SimpleNamespace(data=data), scene=scene, joint_ids=list(range(29)))
    dynamics = SharedDynamics()
    result = dynamics.read_sim_state(command)
    torch.testing.assert_close(result[6], torch.tensor([[2., 2., 0.7]]))
    torch.testing.assert_close(result[8], data.root_link_lin_vel_w)
    scene.clear()
    result = dynamics.read_sim_state(command)
    assert (result[6] == 0).all()
    torch.testing.assert_close(result[7], torch.eye(3)[None])
