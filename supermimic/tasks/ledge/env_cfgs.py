"""G1 on the approved fixed video11 scene, with native tracking managers."""

from dataclasses import dataclass, replace
from functools import partial
import os

from mjlab.envs import ManagerBasedRlEnvCfg

from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from .assets import DEFAULT_MOTION, configure_scene, make_robot_spec
from .commands import LedgeCommandCfg


@dataclass(kw_only=True)
class LedgeEnvCfg(ManagerBasedRlEnvCfg):
    def __post_init__(self):
        if set(self.scene.entities) != {"robot"}:
            raise ValueError("Ledge models one robot and a fixed scene")
        self.scene = replace(self.scene, terrain=None,
                             spec_fn=partial(configure_scene, command=self.commands["motion"]))
        self.sim = replace(self.sim, nconmax=max(512, self.sim.nconmax or 0),
                           njmax=max(2048, self.sim.njmax or 0))


def make_env_cfg(play=False):
    baseline = make_tracking_env_cfg(play)
    baseline.scene.num_envs = 1 if play else 4096
    command = LedgeCommandCfg(**vars(baseline.commands["motion"]))
    command.motion_dir = os.environ.get("SUPERMIMIC_MOTION_DIR", str(DEFAULT_MOTION))
    # Exact reference resets avoid placing a stance foot beyond the 22 cm ledge
    # or a body inside the wall before the policy can act. RSI still samples time.
    command.reset_noise = False
    baseline.commands["motion"] = command
    robot = make_g1_robot_cfg(interaction_collisions=True)
    robot.spec_fn = make_robot_spec
    baseline.scene.entities["robot"] = robot
    baseline.episode_length_s = 40.0
    return LedgeEnvCfg(**vars(baseline))
