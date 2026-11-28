"""The ordinary reference clock and reset lifecycle for a static ledge scene."""

from dataclasses import dataclass, field

import torch

from ...mdp.commands import MotionCommand, MotionCommandCfg
from .mdp import LedgeRewardCfg


@dataclass(kw_only=True)
class LedgeCommandCfg(MotionCommandCfg):
    scene_xml: str | None = None
    ledge_reward_cfg: LedgeRewardCfg = field(default_factory=LedgeRewardCfg)

    def build(self, env):
        return LedgeCommand(self, env)


class LedgeCommand(MotionCommand):
    def build_task(self):
        from .task import build_task
        return build_task(device=self.device, play=self.cfg.play, cfg=self._env.cfg)

    def _update_metrics(self):
        if not hasattr(self, "robot"):
            return
        geometry = self.mdp.geometry(self.state())
        active = self.motion.back_required[self.time_steps]
        self.metrics["back_gap_m"] = torch.where(active, geometry["back_distance_sq"].sqrt(), 0.)
        self.metrics["minimum_foot_margin_m"] = geometry["foot_margin"].amin(-1)
