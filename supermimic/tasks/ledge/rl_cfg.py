"""Ledge learner defaults using the shared SuperMimic runner."""

from ...rl.config import SuperMimicRunnerCfg


def make_runner_cfg():
    return SuperMimicRunnerCfg(experiment_name="supermimic_ledge", buffer_steps=256,
                               wm_batch=1024, pi_batch=512, save_interval=200, log_every=10)
