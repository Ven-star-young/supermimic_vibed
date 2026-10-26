"""Conservative memory defaults for entity interaction and geometry networks."""

from ...rl.config import SuperMimicRunnerCfg


def make_runner_cfg():
    return SuperMimicRunnerCfg(experiment_name="supermimic_g1_entities",
                              buffer_steps=256, wm_batch=128, pi_batch=128)
