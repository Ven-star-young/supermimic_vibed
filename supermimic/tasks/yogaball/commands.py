"""Synchronized G1/ball reference resets with the existing object tracking heads."""

from dataclasses import dataclass, field

from ...mdp.commands import MotionCommandCfg
from ..sitting.commands import SittingCommand
from ..sitting.events import ObjectPhysicsCfg
from ..sitting.mdp import ObjectRewardCfg
from .assets import selected_asset


@dataclass(kw_only=True)
class YogaBallCommandCfg(MotionCommandCfg):
    object_entity_name: str = "ball"
    object_physical_cfg: ObjectPhysicsCfg = field(default_factory=lambda: ObjectPhysicsCfg(
        mass=(1.0, 1.0), evaluation_mass=1.0, robot_friction=(1.0, 1.0)))
    object_reward_cfg: ObjectRewardCfg = field(default_factory=ObjectRewardCfg)

    def build(self, env):
        return YogaBallCommand(self, env)


class YogaBallCommand(SittingCommand):
    asset_loader = staticmethod(selected_asset)

    def build_task(self):
        from .task import build_task
        return build_task(device=self.device, play=self.cfg.play, cfg=self._env.cfg)
