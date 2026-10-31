"""Side-opening learner defaults, sharing the OpenDoor training algorithm."""

from dataclasses import replace

from ..open_door.rl_cfg import make_runner_cfg as make_door_runner_cfg


def make_runner_cfg():
    return replace(make_door_runner_cfg(), experiment_name="supermimic_open_door_side", wm_only_iters=0)
