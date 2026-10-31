"""Synchronized G1/hinge reference resets using the existing motion command."""

from dataclasses import dataclass, field
import math

import torch

from ...mdp.commands import MotionCommand, MotionCommandCfg
from .assets import door_context
from .mdp import DoorRewardCfg
from .events import DoorRandomizationCfg


@dataclass(kw_only=True)
class DoorCommandCfg(MotionCommandCfg):
    motion_files: tuple[str, ...] = ()
    door_xml: str | None = None
    door_mass: float = 8.0
    door_damping: float = 5.0
    door_friction: float = 0.5
    robot_door_friction: float = 0.7
    door_stiffness: float = 20.0
    door_spring_reference: float = -0.1
    door_reward_cfg: DoorRewardCfg = field(default_factory=DoorRewardCfg)
    door_randomization_cfg: DoorRandomizationCfg = field(default_factory=DoorRandomizationCfg)

    def __post_init__(self):
        values = (self.door_mass, self.door_damping, self.door_friction,
                  self.robot_door_friction, self.door_stiffness)
        if any(not math.isfinite(v) or v < 0 for v in values) or self.door_mass == 0:
            raise ValueError("Door physical parameters must be finite/nonnegative, with positive mass")
        if not math.isfinite(self.door_spring_reference):
            raise ValueError("Door spring reference must be finite")

    def build(self, env):
        return DoorCommand(self, env)


class DoorCommand(MotionCommand):
    def build_task(self):
        from .task import build_task
        return build_task(device=self.device, play=self.cfg.play, cfg=self._env.cfg)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.door = self.scene["door"]
        if tuple(self.door.joint_names) != ("door_hinge",) or self.door.is_actuated or not self.door.is_fixed_base:
            raise ValueError("OpenDoor requires exactly one passive fixed-base hinge")
        self.xi[:, 33:] = torch.as_tensor(door_context(self.motion.door_metadata, cfg), device=self.device)

    def reset_reference(self, env_ids):
        super().reset_reference(env_ids)
        theta, omega = self.motion.door_ref_state(self.time_steps[env_ids])
        self.door.write_joint_state_to_sim(theta[:, None], omega[:, None], env_ids=env_ids)
