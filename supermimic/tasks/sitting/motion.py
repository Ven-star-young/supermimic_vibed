"""Object references on exactly the same control clock as the robot clips."""

from pathlib import Path

import torch

from motion_data import load_motion
from ...motion import MotionLib, _gradient
from ...rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat


class SittingMotionLib(MotionLib):
    object_goal_feature_dim = 9

    def __init__(self, files, kin, device):
        super().__init__(files, kin, device)
        positions, rotations, velocities, angular_velocities = [], [], [], []
        for path, count in zip(files, self.clip_len.tolist()):
            clip = load_motion(path)
            if "object_pos" not in clip:
                raise ValueError(f"{path}: sitting training requires object reference poses")
            p = torch.as_tensor(clip["object_pos"], dtype=torch.float32)
            q = torch.as_tensor(clip["object_quat"], dtype=torch.float32)
            # The robot loader owns the frame count: no independent rounding.
            time = torch.arange(count, dtype=torch.float64) * self.control_dt * float(clip["fps"])
            i = time.floor().long().clamp(max=len(p) - 2)
            alpha = (time - i).clamp(0, 1).float()
            p = torch.lerp(p[i], p[i + 1], alpha[:, None])
            R = quat_to_mat(quat_slerp(q[i], q[i + 1], alpha))
            v = _gradient(p, self.control_dt)
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * self.control_dt)
            w[0] = so3_log(R[1] @ R[0].T) / self.control_dt
            w[-1] = so3_log(R[-1] @ R[-2].T) / self.control_dt
            positions.append(p)
            rotations.append(R)
            velocities.append(v)
            angular_velocities.append(w)
        self.object_p, self.object_R, self.object_v, self.object_w = (
            torch.cat(values).to(device)
            for values in (positions, rotations, velocities, angular_velocities)
        )

    @classmethod
    def from_dir(cls, path, kin, device):
        return cls(sorted(Path(path).glob("*.npz")), kin, device)

    def object_ref_state(self, f):
        return self.object_p[f], self.object_R[f], self.object_v[f], self.object_w[f]

    def full_ref_state(self, f):
        return self.ref_state(f) + self.object_ref_state(f)

    def object_goal(self, S, f, K):
        """Future object poses in the current actual G1 heading frame, (B, 9K)."""
        future = self.future_frames(f, K)
        Ht = yaw_mat(S[1]).transpose(-1, -2).unsqueeze(-3)
        position = (Ht @ (self.object_p[future] - S[0].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        orientation = mat_to_6d(Ht @ self.object_R[future])
        return torch.cat((position, orientation), -1).flatten(-2)
