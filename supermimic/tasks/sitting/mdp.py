"""Tracking heads plus one object pose reward and privileged object features."""

from dataclasses import dataclass

import torch

from ...mdp import MDP
from ...rotations import so3_log, yaw_mat
from ...rl.shared_world_model import SharedDynamics


@dataclass(frozen=True)
class ObjectRewardCfg:
    weight: float = 1.0
    position_std: float = 0.1
    orientation_std: float = 0.4

    def __post_init__(self):
        if self.weight < 0 or self.position_std <= 0 or self.orientation_std <= 0:
            raise ValueError("Object reward requires a nonnegative weight and positive error scales")


class SittingMDP(MDP):
    def __init__(self, *args, object_enabled=True, object_reward_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_enabled = object_enabled
        self.object_reward_cfg = ObjectRewardCfg() if object_reward_cfg is None else object_reward_cfg
        if object_enabled:
            self.obs_noise = torch.cat((self.obs_noise, self.obs_noise.new_zeros(6)))
            self.obs_scale = torch.cat((self.obs_scale, self.obs_scale.new_ones(6)))

    @property
    def observation_dim(self):
        return super().observation_dim + 6 * self.object_enabled

    @property
    def critic_feature_dim(self):
        return super().critic_feature_dim + 15 + 6 + 12 + 67 + 1

    @property
    def max_step_reward(self):
        return super().max_step_reward + self.object_enabled * self.object_reward_cfg.weight * self.control_dt

    def obs(self, S, a_prev, xi, f):
        robot = super().obs(S[:6], a_prev, xi[..., :33], f)
        if not self.object_enabled:
            return robot
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        position = (Ht @ (S[6] - S[0]).unsqueeze(-1)).squeeze(-1)
        orientation = so3_log(Ht @ S[7])
        return torch.cat((robot, torch.cat((position, orientation), -1) * xi[..., 45:46]), -1)

    def reward(self, S, f, a, a_prev, terms=False):
        robot = super().reward(S[:6], f, a, a_prev, terms=terms)
        if not self.object_enabled:
            return robot
        cfg, motion = self.object_reward_cfg, self.motion
        e_position = (S[6] - motion.object_p[f]).square().sum(-1)
        e_orientation = so3_log(motion.object_R[f].transpose(-1, -2) @ S[7]).square().sum(-1)
        value = cfg.weight * self.control_dt * torch.exp(
            -e_position / cfg.position_std**2 - e_orientation / cfg.orientation_std**2
        )
        if terms:
            reward, diagnostics = robot
            return reward + value, {
                **diagnostics, "object_pose": value,
                "e_object_pos": e_position, "e_object_ori": e_orientation,
            }
        return robot + value

    def termination(self, S, f):
        return super().termination(S[:6], f)

    def soft_termination(self, S, f, tz=None, tt=None):
        return super().soft_termination(S[:6], f, tz, tt)

    def critic_features(self, S, f, xi, geometry=None):
        robot = super().critic_features(S[:6], f, xi[..., :33])
        mask = xi[..., 45:46]
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        e_position = (Ht @ (self.motion.object_p[f] - S[6]).unsqueeze(-1)).squeeze(-1)
        e_orientation = so3_log(S[7].transpose(-1, -2) @ self.motion.object_R[f])
        # The runner supplies the world model's current encoder output. None is
        # useful for tensor-only inspection; never create an unrelated encoder.
        geometry = xi.new_zeros((*xi.shape[:-1], 67)) if geometry is None else geometry.detach()
        if geometry.shape != (*xi.shape[:-1], 67):
            raise ValueError("Critic geometry must have shape (..., 67)")
        objects = torch.cat((SharedDynamics.object_features(S), e_position, e_orientation,
                             xi[..., 33:45], geometry), -1) * mask
        return torch.cat((robot, objects, mask), -1)
