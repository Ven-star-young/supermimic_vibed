"""Hinge references on the same 50 Hz control clock as the G1 clips."""

from pathlib import Path

import numpy as np
import torch

from motion_data import load_motion
from motion_data.door import door_metadata
from ...motion import MotionLib, _gradient


class DoorMotionLib(MotionLib):
    """Robot73 + (theta, 0.1 * omega) goals for one fixed door asset."""

    goal_feature_dim = 75
    door_goal_feature_dim = 2

    def __init__(self, files, kin, device):
        super().__init__(files, kin, device)
        angles, velocities, opening_frames = [], [], []
        self.door_metadata = None
        for path, count in zip(files, self.clip_len.tolist()):
            clip = load_motion(path)
            if "door_angle" not in clip:
                raise ValueError(f"{path}: OpenDoor training requires door_angle")
            metadata = door_metadata(clip)
            _validate_centered_asset(metadata, path)
            if self.door_metadata is None:
                self.door_metadata = metadata
            elif any(not np.allclose(metadata[key], value, atol=1e-6, rtol=0)
                     for key, value in self.door_metadata.items()):
                raise ValueError(f"{path}: OpenDoor clips must use the same fixed door asset and placement")
            angle = torch.as_tensor(clip["door_angle"], dtype=torch.float32)
            # The base loader owns frame counts; use its clock without a second
            # duration calculation or interpolation of source angular velocity.
            time = torch.arange(count, dtype=torch.float64) * self.control_dt * float(clip["fps"])
            i = time.floor().long().clamp(max=len(angle) - 2)
            alpha = (time - i).clamp(0, 1).float()
            angle = torch.lerp(angle[i], angle[i + 1], alpha)
            angles.append(angle)
            velocities.append(_gradient(angle, self.control_dt))
            source_opening = clip.get("door_opening_frame", 0)
            opening_frames.append(int(np.ceil(float(source_opening) / float(clip["fps"]) / self.control_dt)))
        self.door_angle = torch.cat(angles).to(device)
        self.door_omega = torch.cat(velocities).to(device)
        self.door_opening_frames = torch.tensor(opening_frames, device=device)

    @classmethod
    def from_dir(cls, path, kin, device):
        return cls(sorted(Path(path).glob("*.npz")), kin, device)

    def door_ref_state(self, f):
        """Angle and angular velocity, each with the shape of ``f``."""
        return self.door_angle[f], self.door_omega[f]

    def full_ref_state(self, f):
        return self.ref_state(f) + tuple(value.unsqueeze(-1) for value in self.door_ref_state(f))

    def door_goal(self, f, K):
        future = self.future_frames(f, K)
        return torch.stack((self.door_angle[future], .1 * self.door_omega[future]), -1).flatten(-2)

    def goal(self, f, K):
        """Interleave robot and hinge features per future frame, (B, 75K)."""
        robot = super().goal(f, K).reshape(f.shape[0], K, 73)
        door = self.door_goal(f, K).reshape(f.shape[0], K, 2)
        return torch.cat((robot, door), -1).flatten(-2)


OpenDoorMotionLib = DoorMotionLib


def _validate_centered_asset(values, path):
    # Analytic geometry follows the supplied centered asset: -Z hinge, no
    # closed-pose yaw, panel centered half its width along local -Y.
    if not np.allclose(values["joint_axis"], [0, 0, -1], atol=1e-6):
        raise ValueError(f"{path}: OpenDoor requires door_joint_axis [0, 0, -1]")
    if abs(float(values["base_yaw"])) > 1e-6:
        raise ValueError(f"{path}: OpenDoor requires door_base_yaw 0 in the centered asset frame")
    half = values["panel_half_size"]
    if not np.allclose(values["hinge_to_panel"], [0, -half[1], 0], atol=1e-6):
        raise ValueError(f"{path}: invalid door dimensions or hinge_to_panel offset")
