"""Native mjlab action term mapping policy actions to named joint targets."""
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


@dataclass(kw_only=True)
class JointPositionActionCfg(ActionTermCfg):
    joint_names: tuple[str, ...]
    scale: tuple[float, ...]
    calibration_start: int = 1
    command_name: str = "motion"

    def build(self, env):
        return JointPositionAction(self, env)


class JointPositionAction(ActionTerm):
    def __init__(self, cfg: JointPositionActionCfg, env):
        super().__init__(cfg, env)
        self.robot = self._entity
        if len(cfg.joint_names) != len(cfg.scale):
            raise ValueError("Action scale must contain one value per selected joint")
        self.joint_ids = torch.tensor([self.robot.joint_names.index(n) for n in cfg.joint_names], device=self.device)
        self.scale = torch.tensor(cfg.scale, device=self.device)
        self.offset = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._target = self.offset.expand(self.num_envs, -1).clone()

    @property
    def action_dim(self):
        return len(self.cfg.joint_names)

    @property
    def raw_action(self):
        return self._raw_action

    def process_actions(self, actions):
        self._raw_action.copy_(actions)
        cmd = self._env.command_manager.get_term(self.cfg.command_name)
        begin = self.cfg.calibration_start
        self._target.copy_(self.offset + cmd.xi[:, begin:begin + self.action_dim] + self.scale * actions)

    def apply_actions(self):
        self.robot.set_joint_position_target(self._target, joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_action[ids] = 0
        self._target[ids] = self.offset
