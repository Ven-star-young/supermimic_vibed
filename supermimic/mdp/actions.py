"""Map policy actions to named mjlab joint targets."""
from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class JointPositionActionCfg:
    entity_name: str
    joint_names: tuple[str, ...]
    scale: tuple[float, ...]
    calibration_start: int = 1


class JointPositionAction:
    def __init__(self, cfg: JointPositionActionCfg, scene, device: str):
        self.cfg = cfg
        self.robot = scene[cfg.entity_name]
        if len(cfg.joint_names) != len(cfg.scale):
            raise ValueError("Action scale must contain one value per selected joint")
        self.joint_ids = torch.tensor([self.robot.joint_names.index(n) for n in cfg.joint_names], device=device)
        self.scale = torch.tensor(cfg.scale, device=device)
        self.offset = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self.action_dim = len(cfg.joint_names)

    def apply(self, action, context):
        begin = self.cfg.calibration_start
        target = self.offset + context[:, begin:begin + self.action_dim] + self.scale * action
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
