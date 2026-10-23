"""Differentiable motion tracking reward terms (paper Table A.2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..motion import CONTROL_DT
from ..rotations import rot_angle_sq
from .state import aligned_reference_positions


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    # Retained for old MDP(..., cfg=RewardCfg(...)) callers. New task configs
    # provide an explicit TerminationCfg instead.
    term_height: float = 0.25
    term_tilt: float = 0.8
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_linvel: float = 1.0
    w_body_angvel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5

    def max_step_reward(self, control_dt: float = CONTROL_DT) -> float:
        """Tracking reward ceiling; regularization weights are nonpositive."""
        return (
            self.w_body_pos + self.w_body_ori + self.w_body_linvel + self.w_body_angvel
            + self.w_root_pos + self.w_root_ori
        ) * control_dt


MAX_STEP_REWARD = RewardCfg().max_step_reward()


def tracking_reward(S, body_state, reference_root, reference_bodies, a, a_prev,
                    soft_lo, soft_hi, cfg: RewardCfg, control_dt: float, terms: bool = False):
    """Reward from tensors only: no scene reads, cached simulator values or detach."""
    c = cfg
    p, R, _, _, q, _ = S
    bp, bR, bv, bw = body_state
    rp, rR = reference_root
    rbp, rbR, rbv, rbw = reference_bodies
    ref_bp, dR = aligned_reference_positions(p, R, rp, rR, rbp)
    ref_bR = dR.unsqueeze(-3) @ rbR
    e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
    e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
    e_lv = ((rbv - bv) ** 2).sum(-1).mean(-1)
    e_av = ((rbw - bw) ** 2).sum(-1).mean(-1)
    e_rp = ((rp - p) ** 2).sum(-1)
    e_ro = rot_angle_sq(rR, R)
    r_track = (
        c.w_body_pos * torch.exp(-e_pos / c.body_pos_std**2)
        + c.w_body_ori * torch.exp(-e_ori / c.body_ori_std**2)
        + c.w_body_linvel * torch.exp(-e_lv / c.body_linvel_std**2)
        + c.w_body_angvel * torch.exp(-e_av / c.body_angvel_std**2)
        + c.w_root_pos * torch.exp(-e_rp / c.root_pos_std**2)
        + c.w_root_ori * torch.exp(-e_ro / c.root_ori_std**2)
    )
    r_rate = ((a - a_prev) ** 2).sum(-1)
    r_lim = (torch.relu(q - soft_hi) + torch.relu(soft_lo - q)).sum(-1)
    r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * control_dt
    if terms:
        return r, {
            "track": r_track * control_dt,
            "e_body_pos": e_pos,
            "e_body_ori": e_ori,
            "e_root_pos": e_rp,
            "action_rate": r_rate,
            "joint_limit": r_lim,
        }
    return r
