"""One state-based reward for physical and imagined back-opening transitions."""

from dataclasses import dataclass
import math

import torch

from ...mdp import MDP


@dataclass(frozen=True)
class DoorRewardCfg:
    angle_weight: float = 1.0
    velocity_weight: float = 0.25
    angle_std: float = 0.12
    velocity_std: float = 1.0
    back_weight: float = 0.25
    illegal_weight: float = 2.0
    contact_phase_angle: float = 0.05
    contact_distance_std: float = 0.05
    penetration_scale: float = 0.01
    # Raw video references contain overlap. Start with a bounded penalty;
    # callers can additionally enable the same state-based failure boundary.
    penetration_limit: float | None = None
    hinge_margin: float = 0.05

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is None and name == "penetration_limit":
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if not name.endswith("weight") and value == 0:
                raise ValueError(f"{name} must be positive")


def smoothstep(value):
    u = value.clamp(0, 1)
    return u.square() * (3 - 2 * u)


class DoorMDP(MDP):
    def __init__(self, *args, dynamics, geometry, door_reward_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics, self.geometry = dynamics, geometry
        self.door_reward_cfg = door_reward_cfg or DoorRewardCfg()
        limit = self.door_reward_cfg.penetration_limit
        if limit is not None and limit >= min(self.motion.door_metadata["panel_half_size"]):
            raise ValueError("penetration_limit must be smaller than the door's minimum half-size")
        self.obs_noise = torch.cat((self.obs_noise, self.obs_noise.new_zeros(9)))
        self.obs_scale = torch.cat((self.obs_scale, self.obs_scale.new_ones(9)))
        self.hinge_range = tuple(float(v) for v in self.motion.door_metadata["joint_range"])

    @property
    def observation_dim(self):
        return super().observation_dim + 9

    @property
    def critic_feature_dim(self):
        return super().critic_feature_dim + 9 + 2 + 10

    @property
    def max_step_reward(self):
        c = self.door_reward_cfg
        return super().max_step_reward + (c.angle_weight + c.velocity_weight + c.back_weight) * self.control_dt

    def obs(self, S, a_prev, xi, f):
        return torch.cat((super().obs(S[:6], a_prev, xi[..., :33], f),
                          self.dynamics.door_features(S, xi)), -1)

    def reward(self, S, f, a, a_prev, terms=False):
        # Robot reward already includes dt, action-rate and joint-limit costs.
        robot = super().reward(S[:6], f, a, a_prev, terms=terms)
        c = self.door_reward_cfg
        theta, omega = S[6].squeeze(-1), S[7].squeeze(-1)
        ref_theta, ref_omega = self.motion.door_ref_state(f)
        e_angle, e_velocity = (theta - ref_theta).square(), (omega - ref_omega).square()
        geom = self.geometry(S)
        phase = smoothstep(ref_theta / c.contact_phase_angle)
        allowed = (torch.exp(-geom["back_distance_sq"] / c.contact_distance_std**2)
                   * smoothstep((geom["alignment"] - .5) / .5))
        depth_sq = (-geom["illegal_clearance"]).clamp_min(0).square()
        # Bounded cost without a flat saturated band: overlapping video
        # references must still have an escape gradient beyond one centimetre.
        illegal = (depth_sq / (depth_sq + c.penetration_scale**2)).amax(-1)
        values = {
            "door_angle": c.angle_weight * torch.exp(-e_angle / c.angle_std**2),
            "door_velocity": c.velocity_weight * torch.exp(-e_velocity / c.velocity_std**2),
            "back_state": c.back_weight * phase * allowed,
            "illegal_state": -c.illegal_weight * illegal,
        }
        values = {key: value * self.control_dt for key, value in values.items()}
        reward = (robot[0] if terms else robot) + sum(values.values())
        if terms:
            return reward, {**robot[1], **values, "e_door_angle": e_angle, "e_door_velocity": e_velocity}
        return reward

    def critic_features(self, S, f, xi):
        robot = super().critic_features(S[:6], f, xi[..., :33])
        ref_theta, ref_omega = self.motion.door_ref_state(f)
        error = torch.stack((ref_theta - S[6].squeeze(-1), ref_omega - S[7].squeeze(-1)), -1)
        return torch.cat((robot, self.dynamics.door_features(S, xi), error, xi[..., 36:46]), -1)

    def _door_margins(self, S):
        c = self.door_reward_cfg
        theta = S[6].squeeze(-1)
        lo, hi = self.hinge_range
        margins = [theta - lo + c.hinge_margin, hi + c.hinge_margin - theta]
        widths = [.025, .025]
        if c.penetration_limit is not None:
            margins.append(self.geometry(S)["illegal_clearance"].amin(-1) + c.penetration_limit)
            widths.append(c.penetration_scale)
        return torch.stack(margins, -1), theta.new_tensor(widths)

    def termination(self, S, f):
        margins, _ = self._door_margins(S)
        return super().termination(S[:6], f) | (margins <= 0).any(-1)

    def soft_termination(self, S, f, tz=None, tt=None):
        margins, widths = self._door_margins(S)
        risk = smoothstep((widths - margins) / (2 * widths))
        robot = super().soft_termination(S[:6], f, tz, tt)
        return 1 - (1 - robot) * (1 - risk).prod(-1)
