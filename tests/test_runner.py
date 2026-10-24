"""Exercise learning/checkpoint/inference/resume against the real mjlab lifecycle."""

from dataclasses import asdict
import gc

import torch


def test_native_runner_learns_saves_loads_and_refills_replay(eval_env, tmp_path, monkeypatch):
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.rl import runner as runner_module
    from supermimic.rl.config import SuperMimicRunnerCfg
    from supermimic.rl.models import WorldModel

    # Keep real model integration, gradients, optimizers and native simulation;
    # only reduce world-model width for a fast regression on a laptop GPU.
    monkeypatch.setattr(runner_module, "WorldModel", lambda **kwargs: WorldModel(hidden=(32, 32), **kwargs))
    cfg = asdict(SuperMimicRunnerCfg(
        buffer_steps=8, utd=1.0, horizon=2, wm_window=2, wm_batches=1,
        wm_batch=2, pi_batch=2, warmup_steps=0, wm_only_iters=0,
        critic_epochs=1, critic_minibatches=1, no_compile=True, amp="none",
    ))
    wrapped = RslRlVecEnvWrapper(eval_env.env)
    runner = runner_module.SuperMimicRunner(wrapped, cfg, device=eval_env.device)
    assert runner.wm is None and runner.critic is None and runner.buffer is None
    initial = runner.pi.dec[-1].weight.detach().clone()
    runner.learn(1)
    assert runner.current_learning_iteration == 1
    assert runner.buffer.size >= cfg["wm_window"]
    assert not torch.equal(runner.pi.dec[-1].weight, initial)
    assert all(torch.isfinite(parameter).all() for parameter in runner.wm.parameters())
    checkpoint = tmp_path / "model_1.pt"
    runner.save(checkpoint)
    observations = wrapped.get_observations()
    with torch.no_grad():
        expected = runner.get_inference_policy()(observations).clone()
    saved_sim_steps = runner.sim_steps
    del runner
    gc.collect()

    player = runner_module.SuperMimicRunner(wrapped, cfg, device=eval_env.device)
    player.load(checkpoint, load_cfg={"actor": True})
    assert player.wm is None and player.critic is None and player.buffer is None
    with torch.no_grad():
        actual = player.get_inference_policy()(observations)
    torch.testing.assert_close(actual, expected)
    del player
    gc.collect()

    resumed = runner_module.SuperMimicRunner(wrapped, cfg, device=eval_env.device)
    resumed.load(checkpoint)
    assert resumed.current_learning_iteration == 1
    assert resumed.buffer is None
    resumed.learn(1)
    assert resumed.current_learning_iteration == 2
    assert resumed.buffer.size >= cfg["wm_window"]
    assert resumed.sim_steps >= saved_sim_steps + eval_env.N * (cfg["wm_window"] + 1)
    assert len(resumed.opt_wm.state) and len(resumed.opt_pi.state) and len(resumed.opt_v.state)
