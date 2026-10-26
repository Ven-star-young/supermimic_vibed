"""Actual native training and task-independent world-model/replay transfer."""
from contextlib import contextmanager
from dataclasses import asdict
import gc
import shutil

import pytest
import torch


@contextmanager
def interaction_env(directory, device, *, objects):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.tasks.interaction.env_cfgs import make_env_cfg

    cfg = make_env_cfg(object_enabled=objects)
    cfg.scene.num_envs = 2
    cfg.auto_reset = False
    cfg.commands["motion"].motion_dir = str(directory)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        env.reset()
        yield RslRlVecEnvWrapper(env)
    finally:
        env.close()


def test_native_sitting_learning_and_transfer_to_shared_tracking(cuda_device, tmp_path, monkeypatch):
    from motion_data import load_motion
    from motion_data.collision import load_collision_parts
    from supermimic.assets import DATA_DIR
    from supermimic.rl import runner as module
    from supermimic.rl.config import SuperMimicRunnerCfg
    from supermimic.rl.shared_world_model import SharedWorldModel

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
    # Reduce only the test network width; state integration, contact physics,
    # replay, normalization and policy/world-model updates remain real.
    original_init = SharedWorldModel.__init__
    monkeypatch.setattr(SharedWorldModel, "__init__",
                        lambda self, **kw: original_init(self, hidden=(32, 32), **kw))
    cfg = asdict(SuperMimicRunnerCfg(
        buffer_steps=8, utd=1., horizon=2, wm_window=2, wm_batches=1,
        wm_batch=2, pi_batch=2, warmup_steps=0, wm_only_iters=0,
        critic_epochs=1, critic_minibatches=1, no_compile=True, amp="none",
    ))
    checkpoint, replay = tmp_path / "physics.pt", tmp_path / "sitting_replay.pt"
    with interaction_env(directory, cuda_device, objects=True) as env:
        runner = module.SuperMimicRunner(env, cfg, device=cuda_device)
        initial = runner.pi.dec[-1].weight.detach().clone()
        runner.learn(1)
        assert runner.dims.observation == 99
        assert runner.critic.net[0].in_features == 1075
        assert runner.wm.in_dim == 228 and runner.wm.net[-1].out_features == 82
        assert not torch.equal(initial, runner.pi.dec[-1].weight)
        assert all(torch.isfinite(p).all() for p in runner.wm.parameters())
        runner.save_world_model(checkpoint)
        runner.save_world_model_replay(replay)
        expected = {k: v.detach().cpu().clone() for k, v in runner.wm.state_dict().items()}
        runner.save(tmp_path / "sitting_full.pt")
        from mjlab.utils.os import get_checkpoint_path
        runner.cfg.export_world_model_replay = True
        run = tmp_path / "runs" / "run1"
        runner.save(run / "model_1.pt")
        assert (run / "world_models/model_1.pt").is_file()
        assert (run / "replays/model_1.pt").is_file()
        assert get_checkpoint_path(tmp_path / "runs", checkpoint="model_.*.pt") == run / "model_1.pt"
        del runner
    gc.collect()
    cfg.update(world_model_file=str(checkpoint), world_model_replays=(str(replay),))
    with interaction_env(directory, cuda_device, objects=False) as env:
        runner = module.SuperMimicRunner(env, cfg, device=cuda_device)
        runner._init_training_models()
        assert runner.dims.observation == 93
        for key, value in runner.wm.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
        object_count = runner.wm.out_norm.count[70:].clone()
        runner.learn(1)
        assert (runner.wm.out_norm.count[70:] > object_count).all()  # Old sitting data was used.
        assert runner.current_learning_iteration == 1
        with pytest.raises(ValueError, match="mismatch"):
            runner.load(tmp_path / "sitting_full.pt")  # Full policies remain task-specific.
