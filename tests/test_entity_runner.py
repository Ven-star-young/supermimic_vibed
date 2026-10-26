"""Native multi-entity learning, inference checkpoints and physics reuse."""

from contextlib import contextmanager
from dataclasses import asdict
import gc
import shutil

import pytest
import torch


@contextmanager
def entity_env(directory, device, *, objects):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.tasks.entities.assets import ObjectCfg
    from supermimic.tasks.entities.env_cfgs import make_env_cfg

    extras = (ObjectCfg(name="box", pos=(1.5, 0., .15)),) if objects else ()
    cfg = make_env_cfg(object_enabled=objects, max_objects=3, extra_objects=extras)
    cfg.scene.num_envs = 2
    cfg.auto_reset = False
    cfg.commands["motion"].motion_dir = str(directory)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        env.reset()
        yield RslRlVecEnvWrapper(env)
    finally:
        env.close()


def _small_networks(monkeypatch):
    from supermimic.rl.entity_policy import EntityPolicy, EntityCritic
    from supermimic.rl.entity_world_model import EntityWorldModel

    wm_init, actor_init, critic_init = EntityWorldModel.__init__, EntityPolicy.__init__, EntityCritic.__init__
    monkeypatch.setattr(EntityWorldModel, "__init__",
                        lambda self, **kw: wm_init(self, width=32, layers=2, heads=4, **kw))
    monkeypatch.setattr(EntityPolicy, "__init__",
                        lambda self, *args, **kw: actor_init(self, *args, hidden=(32, 32), **kw))
    monkeypatch.setattr(EntityCritic, "__init__",
                        lambda self, *args, **kw: critic_init(self, *args, hidden=(32, 32), **kw))


def test_native_entity_learning_checkpoint_and_replay_transfer(cuda_device, tmp_path, monkeypatch):
    from motion_data import load_motion
    from motion_data.collision import load_collision_parts
    from supermimic.assets import DATA_DIR
    from supermimic.rl.config import SuperMimicRunnerCfg
    from supermimic.rl.entity_replay import EntityReplay
    from supermimic.rl.runner import SuperMimicRunner
    from supermimic.rl.shared_world_model import SharedDynamics

    sources = sorted((DATA_DIR / "motions/sitting").glob("*.npz"))
    if not sources:
        pytest.skip("Real sitting reference data is not installed")
    try:
        load_collision_parts(load_motion(sources[0]))
    except FileNotFoundError:
        pytest.skip("Real chair collision cache is not installed")
    directory = tmp_path / "motions"
    directory.mkdir()
    shutil.copyfile(sources[0], directory / sources[0].name)
    _small_networks(monkeypatch)
    torch.manual_seed(57)
    cfg = asdict(SuperMimicRunnerCfg(
        buffer_steps=8, utd=1., horizon=2, wm_window=2, wm_batches=1,
        wm_batch=2, pi_batch=2, warmup_steps=0, wm_only_iters=0,
        critic_epochs=1, critic_minibatches=1, no_compile=True, amp="none",
    ))
    physics, replay, full = (tmp_path / name for name in ("physics.pt", "replay.pt", "full.pt"))
    with entity_env(directory, cuda_device, objects=True) as env:
        runner = SuperMimicRunner(env, cfg, device=cuda_device)
        initial = runner.pi.dec[-1].weight.detach().clone()
        geometry_before = [p.detach().clone() for p in runner.pi.geometry_encoder.parameters()]
        runner.learn(1)
        assert runner.current_learning_iteration == 1
        assert runner.dims.state == 71 + 13 * 3
        assert runner.dims.observation == 93 + 13 * 3
        assert runner.pi.dec[0].in_features == 2242
        assert runner.critic.net[0].in_features == 1042
        assert not torch.equal(initial, runner.pi.dec[-1].weight)
        assert any(not torch.equal(before, after) for before, after in zip(
            geometry_before, runner.pi.geometry_encoder.parameters()))
        # Three fresh real steps, two environments, two observed objects each.
        assert runner.pi.robot_norm.count == 6
        assert runner.pi.object_pose_norm.count == 12
        assert runner.critic.future_norm.count > 0
        assert all(torch.isfinite(p).all() for model in (runner.pi, runner.critic, runner.wm)
                   for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in runner.pi.parameters() if p.grad is not None)

        command = env.unwrapped.command_manager.get_term("motion")
        state = runner.dynamics.read_sim_state(command)
        normalized = runner.dynamics.normalize_context(command.xi)
        frame = command.time_steps.clone()
        goal = runner.motion.goal(frame, runner.goal_frames)
        features = runner._critic_input(state, frame, normalized, goal)
        expected_future = runner.task.critic_goal(state, frame, runner.goal_frames)
        torch.testing.assert_close(features[..., -expected_future.shape[-1]:], expected_future)
        altered = features.clone()
        future = runner.critic.split_inputs(altered)[-1]
        future[..., 0, 1:, 0] += 0.2
        assert (runner.critic(features) - runner.critic(altered)).abs().max() > 1e-7

        # Exercise the actual fitted model -> task features -> frozen critic
        # BPTT path. Freezing weights must preserve derivatives to the action.
        runner.wm.requires_grad_(False)
        runner.critic.requires_grad_(False)
        action = torch.zeros(2, 29, device=cuda_device, requires_grad=True)
        predicted = runner.wm.step(state, action, normalized)
        predicted_goal = runner.motion.goal(frame + 1, runner.goal_frames)
        objective = runner.critic(runner._critic_input(predicted, frame + 1, normalized, predicted_goal)).sum()
        action_gradient, = torch.autograd.grad(objective, action)
        assert torch.isfinite(action_gradient).all() and action_gradient.abs().sum() > 0
        runner.wm.requires_grad_(True)
        runner.critic.requires_grad_(True)

        runner.save_world_model(physics)
        runner.save_world_model_replay(replay)
        runner.save(full)
        expected_weights = {k: v.detach().cpu().clone() for k, v in runner.wm.state_dict().items()}
        observations = env.get_observations()
        with torch.no_grad():
            expected_action = runner.get_inference_policy()(observations)
        # Inference reload restores raw geometry and actor parameters without
        # constructing world model, critic, replay or training optimizers.
        inference = SuperMimicRunner(env, cfg, device=cuda_device)
        inference.load(full, load_cfg={"actor": True})
        assert inference.wm is None and inference.critic is None and inference.buffer is None
        with torch.no_grad():
            torch.testing.assert_close(inference.get_inference_policy()(observations), expected_action)
        assert inference.pi.geometry_bank.keys == runner.wm.geometry_bank.keys
        # A matching asset key must not silently replace the scene's actual
        # collision shape during inference restore.
        conflicting = torch.load(full, map_location="cpu", weights_only=False)
        conflicting["entity_geometry"]["points"][1, 0, 0] += 0.01
        conflicting_path = tmp_path / "conflicting_geometry.pt"
        torch.save(conflicting, conflicting_path)
        original_points = inference.pi.geometry_bank.points.clone()
        with pytest.raises(ValueError, match="different content"):
            inference.load(conflicting_path, load_cfg={"actor": True})
        torch.testing.assert_close(inference.pi.geometry_bank.points, original_points)
        del inference, runner, command, predicted, objective, features, altered, observations
    del env
    gc.collect()
    torch.cuda.empty_cache()

    # Reuse the same physical schema and checkpoint in a task with zero active
    # objects. Imported sitting data keeps updating object dynamics statistics.
    cfg.update(world_model_file=str(physics), world_model_replays=(str(replay),))
    with entity_env(directory, cuda_device, objects=False) as env:
        runner = SuperMimicRunner(env, cfg, device=cuda_device)
        runner._init_training_models()
        assert runner.motion.valid_mask.sum() == 0
        assert runner.shared_replay is not None
        for key, value in runner.wm.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected_weights[key], rtol=0, atol=0)
        object_count = runner.wm.out_norm.count[70:].clone()
        runner.learn(1)
        assert (runner.wm.out_norm.count[70:] > object_count).all()
        assert runner.pi.object_pose_norm.count == 0
        assert runner.critic.future_norm.count == 0
        assert all(torch.isfinite(p).all() for p in runner.wm.parameters())
        with pytest.raises(ValueError, match="mismatch"):
            runner.load(full, load_cfg={"actor": True})

        # An old single-body schema must never be silently padded/reinterpreted.
        old_model, old_replay = tmp_path / "old_model.pt", tmp_path / "old_replay.pt"
        torch.save({"schema": SharedDynamics.schema(), "wm": {}}, old_model)
        torch.save({"schema": SharedDynamics.schema()}, old_replay)
        with pytest.raises(ValueError, match="Incompatible"):
            runner.load_world_model(old_model)
        with pytest.raises(ValueError, match="Incompatible"):
            EntityReplay((old_replay,), runner.wm.geometry_bank, runner.dynamics)


def test_standalone_physics_checkpoint_reuses_weights_at_larger_capacity(tmp_path):
    """Exercise the real runner serialization boundary without allocating a sim."""
    from supermimic.rl.entity_state import EntityDynamics
    from supermimic.rl.entity_world_model import EntityGeometryBank, EntityWorldModel
    from supermimic.rl.runner import SuperMimicRunner
    from supermimic.rotations import so3_exp

    torch.manual_seed(71)
    bank = EntityGeometryBank()
    points = torch.zeros(1024, 6)
    points[:, :3] = torch.linspace(-.5, .5, 3072).reshape(1024, 3)
    points[:, 5] = 1
    bank.add("test-body", points, torch.ones(3))

    def stub(capacity):
        runner = object.__new__(SuperMimicRunner)
        runner.device = torch.device("cpu")
        runner.dynamics = EntityDynamics(capacity)
        runner.wm = EntityWorldModel(runner.dynamics, bank, width=32, layers=2, heads=4)
        runner._init_training_models = lambda: None
        return runner

    source, target = stub(2), stub(8)
    random = lambda *shape: torch.randn(2, *shape) * .1
    state = (random(3), so3_exp(random(3)), random(3), random(3), random(29), random(29),
             random(2, 3), so3_exp(random(2, 3)), random(2, 3), random(2, 3))
    action = random(29)
    context = source.dynamics.empty_context(2)
    context[..., 0] = .8
    obj = source.dynamics.object_context(context)
    obj[..., 0] = 10
    obj[..., 4:7] = .2
    obj[..., 10:] = 1
    source.dynamics.pair_friction(context)[...] = torch.tensor([.7, .005, .0001])
    following = source.dynamics.integrate(state, random(source.dynamics.output_dim))
    source.wm.update_normalizers(state, action, context, following)
    for head in (source.wm.robot_head, source.wm.object_head):
        torch.nn.init.normal_(head[-1].weight, std=.02)
    checkpoint = tmp_path / "two_objects.pt"
    source.save_world_model(checkpoint)
    target.load_world_model(checkpoint)
    for key, value in source.wm.state_dict().items():
        torch.testing.assert_close(target.wm.state_dict()[key], value, rtol=0, atol=0)

    # Keep the two actual entities unchanged while padding six empty slots.
    padded = [*state[:6]]
    for index, value in enumerate(state[6:]):
        if index == 1:
            out = torch.eye(3).expand(2, 8, 3, 3).clone()
        else:
            out = torch.zeros(2, 8, 3)
        out[:, :2] = value
        padded.append(out)
    large_context = target.dynamics.empty_context(2)
    large_context[..., :33] = context[..., :33]
    target.dynamics.object_context(large_context)[:, :2] = obj
    # Robot and both objects retain their indices; the ground moves to slot 9.
    selected = torch.tensor([0, 1, 2, 9])
    target.dynamics.pair_friction(large_context)[:, selected[:, None], selected[None, :]] = (
        source.dynamics.pair_friction(context))
    with torch.no_grad():
        expected = source.wm.step(state, action, context)
        actual = target.wm.step(tuple(padded), action, large_context)
    for first, second in zip(expected[:6], actual[:6]):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    for first, second in zip(expected[6:], actual[6:]):
        torch.testing.assert_close(first, second[:, :2], atol=2e-6, rtol=2e-5)

    # Capacity is a batching choice; physical timestep and attention-head layout
    # remain part of the semantic compatibility contract.
    incompatible = torch.load(checkpoint, weights_only=True)
    incompatible["schema"]["control_dt"] = .04
    torch.save(incompatible, tmp_path / "different_dt.pt")
    with pytest.raises(ValueError, match="Incompatible shared world-model"):
        target.load_world_model(tmp_path / "different_dt.pt")
    incompatible = torch.load(checkpoint, weights_only=True)
    incompatible["network"]["heads"] = 2
    torch.save(incompatible, tmp_path / "different_heads.pt")
    with pytest.raises(ValueError, match="network configuration"):
        target.load_world_model(tmp_path / "different_heads.pt")
