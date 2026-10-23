"""G1 tensor state and dynamics contract, independent of the simulator.

The state tuple is ``(p, R, v, w, q, qd)``. Root velocities refer to the
pelvis link origin in world coordinates. This schema deliberately models the
robot only; adding a moving object requires a corresponding dynamics schema.
"""

from __future__ import annotations

import torch

from ..rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat

NUM_J = 29
STATE_DIM = 13 + 2 * NUM_J
XI_DIM = 1 + NUM_J + 3
INV_DIM = 1 + 6 + 3 + 3 + 2 * NUM_J


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return (
        x[..., :3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13],
        x[..., 13 : 13 + NUM_J], x[..., 13 + NUM_J : STATE_DIM],
    )


def invariant_features(S) -> torch.Tensor:
    """Height, tilt, body-frame velocities and joints, without global xy/yaw."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Normalize friction, joint calibration offsets and torso COM offsets."""
    return torch.cat(
        ((xi[..., :1] - 0.75) / 0.45, xi[..., 1 : 1 + NUM_J] / 0.01,
         xi[..., 1 + NUM_J : XI_DIM] / 0.05),
        -1,
    )


def aligned_reference_positions(p, R, ref_p, ref_R, ref_body_p):
    """Re-anchor reference bodies to actual root xy/yaw, preserving reference z."""
    dR = yaw_mat(R @ ref_R.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], ref_p[..., 2:3]), -1)
    positions = anchor.unsqueeze(-2) + (
        dR.unsqueeze(-3) @ (ref_body_p - ref_p.unsqueeze(-2)).unsqueeze(-1)
    ).squeeze(-1)
    return positions, dR


class G1Dynamics:
    """State encoding and differentiable integration used by replay and learning.

    The neural network predicts 70 frame-local increments. The simulator stores
    71 values because its root orientation is packed as a quaternion.
    """

    state_dim = STATE_DIM
    action_dim = NUM_J
    context_dim = XI_DIM
    feature_dim = INV_DIM
    input_dim = INV_DIM + NUM_J + XI_DIM
    output_dim = 12 + 2 * NUM_J
    state_components = 6

    pack = staticmethod(flatten_state)
    unpack = staticmethod(unflatten_state)
    features = staticmethod(invariant_features)
    normalize_context = staticmethod(normalize_xi)

    @staticmethod
    def read_sim_state(env):
        """Read the same G1 state layout from named mjlab robot joints.

        This adapter runs only during collection. Imagined states use integrate()
        and pure tensor heads, so no simulator operation enters the BPTT graph.
        """
        data = env.robot.data
        return (
            data.root_link_pos_w.clone() - env.scene.env_origins,
            quat_to_mat(data.root_link_quat_w.clone()),
            data.root_link_lin_vel_w.clone(), data.root_link_ang_vel_w.clone(),
            data.joint_pos[:, env.joint_ids].clone(), data.joint_vel[:, env.joint_ids].clone(),
        )

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1, including rotation in so(3)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    @staticmethod
    def state_error(S_hat, S):
        """Prediction error in the true state's frame, matching target layout."""
        return G1Dynamics.targets(S, S_hat)

    @staticmethod
    def integrate(S, delta):
        p, R, v, w, q, qd = S
        dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        H = yaw_mat(R)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return p1, R1, v1, w1, q + dq, qd + dqd

    @staticmethod
    def error_floor(device):
        """Per-component lower bound for world-model error normalization."""
        return torch.cat(
            (
                torch.full((3,), 3e-3), torch.full((3,), 5e-3),
                torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2),
            )
        ).to(device)
