"""Pure tensor observations shared by physical and imagined transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..rotations import mat_to_6d, yaw_mat
from .state import INV_DIM, NUM_J, XI_DIM, aligned_reference_positions, invariant_features


@dataclass
class ObservationCfg:
    include_reference_orientation: bool = True
    joint_position_noise: float = 0.01
    joint_velocity_noise: float = 0.5
    angular_velocity_noise: float = 0.2
    gravity_noise: float = 0.05
    reference_orientation_noise: float = 0.05

    @property
    def dimension(self) -> int:
        return 3 * NUM_J + 6 + (6 if self.include_reference_orientation else 0)

    def noise_tensor(self, device=None) -> torch.Tensor:
        values = [
            torch.full((NUM_J,), self.joint_position_noise),
            torch.full((NUM_J,), self.joint_velocity_noise),
            torch.full((3,), self.angular_velocity_noise),
            torch.full((3,), self.gravity_noise),
            torch.zeros(NUM_J),
        ]
        if self.include_reference_orientation:
            values.append(torch.full((6,), self.reference_orientation_noise))
        return torch.cat(values).to(device=device)

    def scale_tensor(self, device=None) -> torch.Tensor:
        # Kept for compatibility; the policy uses its learned RunningNorm.
        values = [
            torch.ones(NUM_J), torch.full((NUM_J,), 0.1), torch.full((3,), 0.25),
            torch.ones(3), torch.ones(NUM_J),
        ]
        if self.include_reference_orientation:
            values.append(torch.ones(6))
        return torch.cat(values).to(device=device)


OBS_DIM = ObservationCfg().dimension
OBS_NOISE = ObservationCfg().noise_tensor()
OBS_SCALE = ObservationCfg().scale_tensor()
CRITIC_EXTRA = NUM_J + 14 * 3 + 3 + 6 + XI_DIM


def proprioception(S, a_prev, xi, reference_R, default_q, cfg: ObservationCfg):
    """Noiseless proprioception, optionally including relative reference attitude."""
    _, R, _, w, q, qd = S
    Rt = R.transpose(-1, -2)
    dq0 = xi[..., 1 : 1 + NUM_J]
    g_b = -R[..., 2, :]
    values = [q - default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev]
    if cfg.include_reference_orientation:
        values.append(mat_to_6d(Rt @ reference_R))
    return torch.cat(values, -1)


def add_observation_noise(o, half_widths):
    return o + (2 * torch.rand_like(o) - 1) * half_widths


def critic_features(S, body_positions, reference_p, reference_R, reference_q, reference_body_p, xi):
    """Invariant state, tracking errors and physical context, excluding future goal."""
    p, R, _, _, q, _ = S
    Ht = yaw_mat(R).transpose(-1, -2)
    ref_bp, _ = aligned_reference_positions(p, R, reference_p, reference_R, reference_body_p)
    e_body = (Ht.unsqueeze(-3) @ (ref_bp - body_positions).unsqueeze(-1)).squeeze(-1).flatten(-2)
    e_root = (Ht @ (reference_p - p).unsqueeze(-1)).squeeze(-1)
    rel = mat_to_6d(R.transpose(-1, -2) @ reference_R)
    return torch.cat((invariant_features(S), reference_q - q, e_body, e_root, rel, xi), -1)


def critic_feature_dimension(num_tracked_bodies: int) -> int:
    return INV_DIM + NUM_J + num_tracked_bodies * 3 + 3 + 6 + XI_DIM
