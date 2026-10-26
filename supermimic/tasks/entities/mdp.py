"""Raw task observations; learned scene encoders live inside the networks."""

import torch

from ...mdp import MDP
from ...rotations import mat_to_6d, so3_log, yaw_mat
from ...rl.entity_state import EntityDynamics
from ..interaction.mdp import ObjectRewardCfg


class EntityMDP(MDP):
    def __init__(self, *args, max_objects=8, object_reward_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_objects = max_objects
        self.dynamics = EntityDynamics(max_objects)
        self.object_reward_cfg = ObjectRewardCfg() if object_reward_cfg is None else object_reward_cfg
        self.obs_noise = torch.cat((self.obs_noise, self.obs_noise.new_zeros(13 * max_objects)))
        self.obs_scale = torch.cat((self.obs_scale, self.obs_scale.new_ones(13 * max_objects)))

    @property
    def observation_dim(self):
        return super().observation_dim + 13 * self.max_objects

    @property
    def critic_feature_dim(self):
        return super().critic_feature_dim + 38 * self.max_objects + 3 * (self.max_objects + 2)**2 + 4

    @property
    def max_step_reward(self):
        present = bool(self.motion.target_mask.any())
        return super().max_step_reward + present * self.object_reward_cfg.weight * self.control_dt

    def obs(self, S, a_prev, xi, f):
        robot = super().obs(S[:6], a_prev, xi[..., :33], f)
        context = self.dynamics.object_context(xi)
        Ht = yaw_mat(S[1]).transpose(-1, -2).unsqueeze(-3)
        position = (Ht @ (S[6] - S[0].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        orientation = mat_to_6d(Ht @ S[7])
        target = self.motion.target_mask.expand_as(context[..., 10])[..., None]
        obj = torch.cat((position, orientation, context[..., 10:12], target, context[..., 12:13]), -1)
        obj = obj * context[..., 10:11]
        return torch.cat((robot, obj.flatten(-2)), -1)

    def reward(self, S, f, a, a_prev, terms=False):
        robot = super().reward(S[:6], f, a, a_prev, terms=terms)
        cfg = self.object_reward_cfg
        mask = self.motion.target_mask
        error_p = (S[6] - self.motion.object_p[f]).square().sum(-1)
        error_R = so3_log(self.motion.object_R[f].transpose(-1, -2) @ S[7]).square().sum(-1)
        value = cfg.weight * self.control_dt * (
            torch.exp(-error_p / cfg.position_std**2 - error_R / cfg.orientation_std**2) * mask
        ).sum(-1) / mask.sum().clamp_min(1)
        if terms:
            reward, diagnostics = robot
            return reward + value, {**diagnostics, "object_pose": value,
                "e_object_pos": (error_p * mask).sum(-1) / mask.sum().clamp_min(1),
                "e_object_ori": (error_R * mask).sum(-1) / mask.sum().clamp_min(1)}
        return robot + value

    def termination(self, S, f):
        return super().termination(S[:6], f)

    def soft_termination(self, S, f, tz=None, tt=None):
        return super().soft_termination(S[:6], f, tz, tt)

    def critic_features(self, S, f, xi, geometry=None):
        robot = super().critic_features(S[:6], f, xi[..., :33])
        context = self.dynamics.object_context(xi)
        Ht = yaw_mat(S[1]).transpose(-1, -2).unsqueeze(-3)
        position = (Ht @ (S[6] - S[0].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        state = torch.cat((position, mat_to_6d(Ht @ S[7]),
                           (Ht @ S[8].unsqueeze(-1)).squeeze(-1),
                           (Ht @ S[9].unsqueeze(-1)).squeeze(-1)), -1)
        target = self.motion.target_mask.expand_as(context[..., 10])[..., None]
        e_position = (Ht @ (self.motion.object_p[f] - S[6]).unsqueeze(-1)).squeeze(-1)
        e_rotation = mat_to_6d(S[7].transpose(-1, -2) @ self.motion.object_R[f])
        error = torch.cat((e_position, e_rotation), -1) * target
        obj = torch.cat((state, context[..., :10], context[..., 11:12], error,
                         target, context[..., 10:11], context[..., 12:13]), -1)
        obj = obj * context[..., 10:11]
        return torch.cat((robot, obj.flatten(-2), self.dynamics.pair_friction(xi).flatten(-3),
                          self.dynamics.ground_features(S, xi)), -1)
