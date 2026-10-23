"""Small, deterministic motion fixture for simulator checks without LAFAN1 data."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from supermimic.motion import MotionLib
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import mat_to_quat, so3_exp


def load_motion(kin, device: str, motion_dir: str | None = None) -> MotionLib:
    """Load real motion data, or exercise the CSV loader with a generated trajectory.

    The synthetic robot starts above the floor with nonzero translation, rotation,
    and joint velocities. This exposes quaternion order, world/local angular
    velocity, and link-origin/center-of-mass velocity conversion mistakes.
    It is a simulator plumbing check, not a motion-tracking benchmark.
    """
    if motion_dir:
        return MotionLib.from_dir(motion_dir, kin, device)
    t = torch.arange(200, dtype=torch.float32) * 0.02
    p = torch.stack((0.2 + 0.1 * t, -0.15 - 0.06 * t, 1.8 + 0.02 * t), -1)
    base = so3_exp(torch.tensor([0.17, -0.13, 0.41]))
    rotation = so3_exp(t[:, None] * torch.tensor([0.19, -0.23, 0.31])) @ base
    quaternion = mat_to_quat(rotation)
    joint_phase = torch.arange(29, dtype=torch.float32) * 0.17
    q = torch.tensor(default_joint_pos_list()) + 0.025 * torch.sin(1.7 * t[:, None] + joint_phase)
    q = q.clamp(kin.q_lo.cpu() + 0.005, kin.q_hi.cpu() - 0.005)
    # LAFAN1 CSVs store xyzw, while the state and mjlab use wxyz.
    data = torch.cat((p, quaternion[:, 1:], quaternion[:, :1], q), -1).numpy()
    with TemporaryDirectory(prefix="supermimic-smoke-") as temp:
        path = Path(temp) / "synthetic.csv"
        np.savetxt(path, data, delimiter=",")
        return MotionLib([str(path)], kin, device, src_fps=50.0)


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
