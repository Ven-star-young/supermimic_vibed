"""Sitting state: one G1 and one dynamic chair, with frame-local increments."""

import torch

from ...mdp.state import G1Dynamics
from ...rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat


def _rotate(rotation, vector):
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


class SittingDynamics:
    """Robot (p,R,v,w,q,qd), then chair (p,R,v,w); velocities are body-origin world velocities."""

    state_dim, action_dim, context_dim = 84, 29, 46
    feature_dim, input_dim, output_dim, state_components = 86, 224, 82, 10
    physical_context_dim = 45

    @staticmethod
    def pack(S):
        p, R, v, w = S[6:]
        return torch.cat((G1Dynamics.pack(S[:6]), p, mat_to_quat(R), v, w), -1)

    @staticmethod
    def unpack(x):
        return (*G1Dynamics.unpack(x[..., :71]), x[..., 71:74], quat_to_mat(x[..., 74:78]),
                x[..., 78:81], x[..., 81:84])

    @staticmethod
    def object_features(S):
        p, R, v, w = S[6:]
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        return torch.cat((_rotate(Ht, p - S[0]), mat_to_6d(Ht @ R),
                          _rotate(Ht, v), _rotate(Ht, w)), -1)

    @staticmethod
    def features(S):
        return torch.cat((G1Dynamics.features(S[:6]), SittingDynamics.object_features(S)), -1)

    @staticmethod
    def normalize_context(context):
        # Geometry ID at 45 is metadata and must survive physical normalization.
        return torch.cat((G1Dynamics.normalize_context(context[..., :33]),
                          (context[..., 33:34] - 10) / 5, context[..., 34:46]), -1)

    @staticmethod
    def read_sim_state(command):
        robot = G1Dynamics.read_sim_state(command)
        data = command.scene["chair"].data
        return (*robot, data.root_link_pos_w.clone() - command.scene.env_origins,
                quat_to_mat(data.root_link_quat_w.clone()),
                data.root_link_lin_vel_w.clone(), data.root_link_ang_vel_w.clone())

    @staticmethod
    def targets(S, S1):
        p, R, v, w = S[6:]
        p1, R1, v1, w1 = S1[6:]
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        return torch.cat((G1Dynamics.targets(S[:6], S1[:6]), _rotate(Ht, p1 - p),
                          so3_log(R.transpose(-1, -2) @ R1),
                          _rotate(Ht, v1 - v), _rotate(Ht, w1 - w)), -1)

    @staticmethod
    def integrate(S, delta):
        p, R, v, w = S[6:]
        dp, dtheta, dv, dw = delta[..., 70:].split(3, -1)
        H = yaw_mat(S[1])
        return (*G1Dynamics.integrate(S[:6], delta[..., :70]),
                p + _rotate(H, dp), R @ so3_exp(dtheta),
                v + _rotate(H, dv), w + _rotate(H, dw))

    @staticmethod
    def state_error(predicted, actual):
        return SittingDynamics.targets(actual, predicted)

    @staticmethod
    def error_floor(device):
        return torch.cat((G1Dynamics.error_floor(device),
                          torch.tensor([3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3,
                                       device=device)))
