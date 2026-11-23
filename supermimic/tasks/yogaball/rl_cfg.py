"""YogaBall learner defaults using the shared rigid-object training settings."""

from dataclasses import replace

from ..sitting.rl_cfg import make_runner_cfg as make_sitting_runner_cfg


def make_runner_cfg():
    return replace(make_sitting_runner_cfg(), experiment_name="supermimic_yogaball")
