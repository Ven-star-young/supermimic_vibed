"""Conservative per-GPU defaults for geometry-conditioned Sitting learning."""
from ...rl.config import SuperMimicRunnerCfg


def make_runner_cfg():
    return SuperMimicRunnerCfg(experiment_name="supermimic_sitting", buffer_steps=256,
                              wm_batch=128, pi_batch=128, save_interval=200, log_every=10)
