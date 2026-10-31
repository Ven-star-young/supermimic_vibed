"""Independent side-opening reward, shared by real and imagined transitions."""

from dataclasses import dataclass
import math

import torch

from ...mdp import MDP
from ..open_door.mdp import DoorMDP, smoothstep


@dataclass(frozen=True)
class SideDoorRewardCfg:
    angle_weight: float = 1.0
    velocity_weight: float = .25
    angle_std: float = .12
    velocity_std: float = 1.0
    contact_weight: float = .25
    illegal_weight: float = 2.0
    contact_distance_std: float = .05
    reference_contact_std: float = .05
    contact_penetration_weight: float = 2.0
    contact_penetration_tolerance: float = .003
    penetration_scale: float = .01
    penetration_limit: float | None = None
    hinge_margin: float = .05

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is None and name == "penetration_limit":
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if not name.endswith("weight") and name != "contact_penetration_tolerance" and value == 0:
                raise ValueError(f"{name} must be positive")


class SideDoorMDP(DoorMDP):
    def __init__(self, *args, door_reward_cfg=None, **kwargs):
        super().__init__(*args, door_reward_cfg=door_reward_cfg or SideDoorRewardCfg(), **kwargs)
        # Reference geometry supplies a smooth phase schedule, never a physical
        # contact/force label. Use the same clock as all robot and door goals.
        with torch.no_grad():
            phases = []
            for frames in torch.arange(len(self.motion.door_angle), device=self.motion.door_angle.device).split(256):
                reference = self.geometry(self.motion.full_ref_state(frames))
                phases.append(torch.exp(-reference["contact_distance_sq"]
                                        / self.door_reward_cfg.reference_contact_std**2))
            self.reference_contact_phase = torch.cat(phases)

    @property
    def max_step_reward(self):
        c = self.door_reward_cfg
        return MDP.max_step_reward.fget(self) + (c.angle_weight + c.velocity_weight + c.contact_weight) * self.control_dt

    def reward(self, S, f, a, a_prev, terms=False):
        robot = MDP.reward(self, S[:6], f, a, a_prev, terms=terms)
        c = self.door_reward_cfg
        theta, omega = S[6].squeeze(-1), S[7].squeeze(-1)
        ref_theta, ref_omega = self.motion.door_ref_state(f)
        e_angle, e_velocity = (theta - ref_theta).square(), (omega - ref_omega).square()
        geom = self.geometry(S)
        contact = (torch.exp(-geom["contact_distance_sq"] / c.contact_distance_std**2)
                   * smoothstep((geom["alignment"] - .5) / .5))

        def penetration(clearance, tolerance=0.):
            depth_sq = (-clearance - tolerance).clamp_min(0).square()
            return (depth_sq / (depth_sq + c.penetration_scale**2)).amax(-1)

        values = {
            "door_angle": c.angle_weight * torch.exp(-e_angle / c.angle_std**2),
            "door_velocity": c.velocity_weight * torch.exp(-e_velocity / c.velocity_std**2),
            "contact_state": c.contact_weight * self.reference_contact_phase[f] * contact,
            "illegal_state": -c.illegal_weight * penetration(geom["illegal_clearance"]),
            "contact_penetration": -c.contact_penetration_weight * penetration(
                geom["contact_clearance"], c.contact_penetration_tolerance),
        }
        values = {key: value * self.control_dt for key, value in values.items()}
        reward = (robot[0] if terms else robot) + sum(values.values())
        if terms:
            return reward, {**robot[1], **values, "e_door_angle": e_angle, "e_door_velocity": e_velocity}
        return reward
