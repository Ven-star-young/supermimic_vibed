"""Side-opening reference command with synchronized robot and hinge resets."""

from dataclasses import dataclass, field

from ..open_door.commands import DoorCommand, DoorCommandCfg
from .mdp import SideDoorRewardCfg


@dataclass(kw_only=True)
class SideDoorCommandCfg(DoorCommandCfg):
    door_reward_cfg: SideDoorRewardCfg = field(default_factory=SideDoorRewardCfg)

    def build(self, env):
        return SideDoorCommand(self, env)


class SideDoorCommand(DoorCommand):
    def build_task(self):
        from .task import build_task
        return build_task(device=self.device, play=self.cfg.play, cfg=self._env.cfg)
