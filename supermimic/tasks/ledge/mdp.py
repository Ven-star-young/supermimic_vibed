"""Tensor-only ledge rewards and failures for real and imagined transitions."""

from dataclasses import dataclass
import math

import torch

from ...mdp import MDP
from ...mdp.observations import critic_features
from ...mdp.rewards import tracking_reward


@dataclass(frozen=True)
class LedgeRewardCfg:
    back_weight: float = .35
    alignment_weight: float = .15
    support_weight: float = .35
    support_height_weight: float = .35
    slip_weight: float = .15
    stability_weight: float = .2
    illegal_weight: float = 1.0
    penetration_weight: float = 2.0
    back_distance_std: float = .05
    support_margin: float = .015
    support_margin_std: float = .03
    support_height_std: float = .04
    slip_speed_std: float = .3
    gravity_std: float = .25
    penetration_tolerance: float = .003
    penetration_scale: float = .01
    root_drop_limit: float = .19
    tilt_limit: float = math.radians(30)
    wall_penetration_limit: float = .02
    platform_penetration_limit: float = .02
    cliff_height: float = .06
    cliff_speed: float = .1
    cliff_root_drop: float = .08
    height_temperature: float = .02
    tilt_temperature: float = .025
    penetration_temperature: float = .005
    cliff_speed_temperature: float = .05

    def __post_init__(self):
        allow_zero = {"support_margin", "penetration_tolerance"}
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if not name.endswith("weight") and name not in allow_zero and value == 0:
                raise ValueError(f"{name} must be positive")
        if self.tilt_limit >= math.pi / 2:
            raise ValueError("tilt_limit must be less than pi/2 radians")
        if self.cliff_root_drop >= self.root_drop_limit:
            raise ValueError("cliff_root_drop must be smaller than root_drop_limit")
        if self.penetration_tolerance >= min(self.wall_penetration_limit,
                                             self.platform_penetration_limit):
            raise ValueError("penetration_tolerance must be smaller than the failure limits")


class LedgeMDP(MDP):
    def __init__(self, *args, geometry, ledge_reward_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.ledge_reward_cfg = ledge_reward_cfg or LedgeRewardCfg()
        self.obs_noise = torch.cat((self.obs_noise, self.obs_noise.new_zeros(12)))
        self.obs_scale = torch.cat((self.obs_scale, self.obs_scale.new_ones(12)))
        m = self.motion
        if m.back_required.shape != (m.num_frames,) or m.back_weight.shape != (m.num_frames,):
            raise ValueError("Back schedules must have one entry per reference frame")
        if m.planned_stance.shape != (m.num_frames, 2):
            raise ValueError("planned_stance must have shape (num_frames, 2)")
        if not torch.isfinite(m.back_weight).all() or ((m.back_weight < 0) | (m.back_weight > 1)).any():
            raise ValueError("back_weight must be finite and in [0, 1]")
        # This is a reference height target, not a measured contact label.
        with torch.no_grad():
            heights = []
            for frames in torch.arange(m.num_frames, device=m.root_p.device).split(256):
                heights.append(self.geometry(m.ref_state(frames))["foot_height"])
            self.reference_foot_height = torch.cat(heights)

    @property
    def observation_dim(self):
        return super().observation_dim + 12

    @property
    def critic_feature_dim(self):
        return super().critic_feature_dim + 12 + 8

    def obs(self, S, a_prev, xi, f):
        return torch.cat((super().obs(S[:6], a_prev, xi[..., :33], f),
                          self.geometry.actor_features(S)), -1)

    def reward(self, S, f, a, a_prev, terms=False):
        m, c = self.motion, self.ledge_reward_cfg
        bodies = self.kin.forward(*S[:6], all_bodies=True)
        tracked = (bodies[0][..., self.kin.tracked_idx, :],
                   bodies[1][..., self.kin.tracked_idx, :, :],
                   bodies[2][..., self.kin.tracked_idx, :],
                   bodies[3][..., self.kin.tracked_idx, :])
        robot = tracking_reward(
            S[:6], tracked, (m.root_p[f], m.root_R[f]),
            (m.body_p[f], m.body_R[f], m.body_v[f], m.body_w[f]), a, a_prev,
            self.soft_lo, self.soft_hi, self.cfg, self.control_dt, terms,
        )
        geom = self.geometry(S, body_state=bodies)
        phase = m.back_weight[f] * m.back_required[f]
        stance = m.planned_stance[f].to(geom["foot_height"].dtype)
        # Only planned support near the platform incurs edge/slip costs. The
        # separate, ungated height target prevents lifting that foot to escape.
        near_platform = torch.exp(-geom["foot_height"].square() / c.support_height_std**2)
        support = stance * near_platform
        edge_error = (c.support_margin - geom["foot_margin"]).clamp_min(0).square()
        height_error = (geom["foot_height"] - self.reference_foot_height[f]).square()
        gravity_error = (S[1][..., 2, :] - m.root_R[f][..., 2, :]).square().sum(-1)
        illegal_depth = (-geom["illegal_clearance"]).clamp_min(0).square()
        penetration = (-torch.minimum(geom["wall_clearance"], geom["platform_clearance"])
                       - c.penetration_tolerance).clamp_min(0).square()
        values = {
            "back_state": -c.back_weight * phase * (
                1 - torch.exp(-geom["back_distance_sq"] / c.back_distance_std**2)),
            "back_alignment": -c.alignment_weight * phase * (1 - geom["alignment"].clamp(-1, 1)) / 2,
            "support_margin": -c.support_weight * (
                support * edge_error / (edge_error + c.support_margin_std**2)).mean(-1),
            "support_height": -c.support_height_weight * (
                stance * (1 - torch.exp(-height_error / c.support_height_std**2))).mean(-1),
            "foot_slip": -c.slip_weight * (
                support * (1 - torch.exp(-geom["foot_speed_sq"] / c.slip_speed_std**2))).mean(-1),
            "stability": -c.stability_weight * (1 - torch.exp(-gravity_error / c.gravity_std**2)),
            "illegal_state": -c.illegal_weight * illegal_depth / (illegal_depth + c.penetration_scale**2),
            "deep_penetration": -c.penetration_weight * penetration / (penetration + c.penetration_scale**2),
        }
        values = {key: value * self.control_dt for key, value in values.items()}
        reward = (robot[0] if terms else robot) + sum(values.values())
        if terms:
            return reward, {**robot[1], **values, "e_gravity": gravity_error,
                            "e_support_height": (stance * height_error).mean(-1)}
        return reward

    def critic_features(self, S, f, xi):
        m = self.motion
        bodies = self.kin.forward(*S[:6], all_bodies=True)
        robot = critic_features(
            S[:6], bodies[0][..., self.kin.tracked_idx, :], m.root_p[f], m.root_R[f],
            m.q[f], m.body_p[f], xi[..., :33],
        )
        geom = self.geometry(S, body_state=bodies)
        context = torch.stack((geom["back_distance_sq"], geom["alignment"],
                               geom["illegal_clearance"], geom["wall_clearance"],
                               geom["platform_clearance"], m.back_weight[f] * m.back_required[f]), -1)
        return torch.cat((robot, self.geometry.actor_features(S, body_state=bodies), context,
                          m.planned_stance[f].to(robot.dtype)), -1)

    def _ledge_margins(self, S, f, tz=None, tt=None):
        c = self.ledge_reward_cfg
        geom = self.geometry(S)
        root_drop = self.motion.root_p[f][..., 2] - S[0][..., 2]
        gravity_cos = (S[1][..., 2, :] * self.motion.root_R[f][..., 2, :]).sum(-1)
        height_width = c.height_temperature if tz is None else tz
        tilt_width = c.tilt_temperature if tt is None else tt
        # A cliff failure needs both feet below the platform AND descent or
        # root-height loss. Normal swing and one foot over the edge remain valid.
        feet_margin = (geom["foot_height"].amax(-1) + c.cliff_height) / height_width
        descent_margin = (S[2][..., 2] + c.cliff_speed) / c.cliff_speed_temperature
        drop_margin = (c.cliff_root_drop - root_drop) / height_width
        cliff_margin = torch.maximum(feet_margin, torch.minimum(descent_margin, drop_margin))
        margins = torch.stack((c.root_drop_limit - root_drop,
                               gravity_cos - math.cos(c.tilt_limit),
                               geom["wall_clearance"] + c.wall_penetration_limit,
                               geom["platform_clearance"] + c.platform_penetration_limit,
                               cliff_margin), -1)
        widths = margins.new_tensor((height_width, tilt_width, c.penetration_temperature,
                                     c.penetration_temperature, 1.0))
        return margins, widths

    def termination(self, S, f):
        margins, _ = self._ledge_margins(S, f)
        return (margins <= 0).any(-1)

    def soft_termination(self, S, f, tz=None, tt=None):
        margins, widths = self._ledge_margins(S, f, tz, tt)
        u = ((widths - margins) / (2 * widths)).clamp(0, 1)
        risk = u.square() * (3 - 2 * u)
        return 1 - (1 - risk).prod(-1)
