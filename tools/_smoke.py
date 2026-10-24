"""Small motion fixtures and native mjlab environments for runtime checks."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from supermimic.motion import MotionLib
from supermimic.assets.g1.g1_constants import default_joint_pos_list
from supermimic.rotations import mat_to_quat, so3_exp


@contextmanager
def motion_directory(kin=None, motion_dir: str | None = None):
    """Yield actual CSV files for the same loader used by the registered task.

    Synthetic motion is intentionally nonstationary and airborne to expose frame,
    quaternion and link-origin/COM velocity conversion errors. It is a plumbing
    fixture, not a motion-tracking benchmark. CSVs use the production 30 Hz format.
    """
    if motion_dir:
        yield str(motion_dir)
        return
    if kin is None:
        from supermimic.assets.g1.kinematics import G1Kinematics
        kin = G1Kinematics("cpu")
    t = torch.arange(200, dtype=torch.float32) / 30.0
    p = torch.stack((0.2 + 0.1 * t, -0.15 - 0.06 * t, 1.8 + 0.02 * t), -1)
    base = so3_exp(torch.tensor([0.17, -0.13, 0.41]))
    rotation = so3_exp(t[:, None] * torch.tensor([0.19, -0.23, 0.31])) @ base
    quaternion = mat_to_quat(rotation)
    joint_phase = torch.arange(29, dtype=torch.float32) * 0.17
    q = torch.tensor(default_joint_pos_list()) + 0.025 * torch.sin(1.7 * t[:, None] + joint_phase)
    q = q.clamp(kin.q_lo.cpu() + 0.005, kin.q_hi.cpu() - 0.005)
    # LAFAN1 CSVs store xyzw; the learned state and mjlab use wxyz.
    data = torch.cat((p, quaternion[:, 1:], quaternion[:, :1], q), -1).numpy()
    with TemporaryDirectory(prefix="supermimic-smoke-") as temp:
        np.savetxt(Path(temp) / "synthetic.csv", data, delimiter=",")
        yield temp


def load_motion(kin, device: str, motion_dir: str | None = None) -> MotionLib:
    with motion_directory(kin, motion_dir) as directory:
        return MotionLib.from_dir(directory, kin, device)


@contextmanager
def native_env(num_envs=4, *, device="cuda:0", motion_dir=None, play=True, auto_reset=True):
    """Construct the stock mjlab environment from the registered task factory."""
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    with motion_directory(motion_dir=motion_dir) as directory:
        cfg = make_env_cfg(play=play)
        cfg.scene.num_envs = num_envs
        cfg.commands["motion"].motion_dir = directory
        cfg.auto_reset = auto_reset
        env = ManagerBasedRlEnv(cfg, device=device)
        try:
            env.reset()
            yield env
        finally:
            env.close()


def reset_frames(env, frames, ids=None):
    if ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    env.command_manager.get_term("motion").set_reset_frames(ids, frames)
    return env.reset(env_ids=ids)


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
