"""Learner defaults for the registered G1 reference-tracking task."""

from ...rl.config import SuperMimicRunnerCfg


def make_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
    return SuperMimicRunnerCfg()
