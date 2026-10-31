"""G1 plus a passive hinge; panel pose is derived from the asset metadata."""

import torch

from ...mdp.state import G1Dynamics
from ...rotations import yaw_mat


class DoorDynamics:
    """State: robot ``(p,R,v,w,q,qd)``, then scalar ``theta, omega``.

    Context: robot33, hinge_xyz3, width/height/thickness, mass, damping,
    hinge_friction, ground_friction, robot_door_friction, stiffness, springref.
    The supplied asset opens about -Z; theta remains its positive joint angle.
    """

    state_dim, action_dim, context_dim = 73, 29, 46
    feature_dim, input_dim, output_dim, state_components = 80, 152, 72, 8
    physical_context_dim = 43
    physics_scale = (1., 2., .1, 10., 1., 1., 1., 1., 20., 1.)

    def __init__(self, base_yaw=0., hinge_sign=-1.):
        if hinge_sign not in (-1., 1.):
            raise ValueError("Door hinge_sign must be -1 or +1")
        self.base_yaw, self.hinge_sign = float(base_yaw), float(hinge_sign)

    @staticmethod
    def pack(S):
        return torch.cat((G1Dynamics.pack(S[:6]), S[6], S[7]), -1)

    @staticmethod
    def unpack(x):
        return (*G1Dynamics.unpack(x[..., :71]), x[..., 71:72], x[..., 72:73])

    def door_features(self, S, context):
        p, R = S[:2]
        theta, omega = S[6:]
        Ht = yaw_mat(R).transpose(-1, -2)
        relative_hinge = (Ht @ (context[..., 33:36] - p).unsqueeze(-1)).squeeze(-1)
        yaw = self.base_yaw + self.hinge_sign * theta
        normal = torch.cat((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        relative_normal = (Ht @ normal.unsqueeze(-1)).squeeze(-1)
        return torch.cat((relative_hinge, relative_normal, theta.sin(), theta.cos(), omega), -1)

    def features(self, S, context):
        return torch.cat((G1Dynamics.features(S[:6]), self.door_features(S, context)), -1)

    @staticmethod
    def normalize_context(context):
        # Keep hinge xyz in meters: relative door features consume it directly.
        scale = context.new_tensor(DoorDynamics.physics_scale)
        return torch.cat((G1Dynamics.normalize_context(context[..., :33]),
                          context[..., 33:36], context[..., 36:46] / scale), -1)

    @staticmethod
    def read_sim_state(command):
        data = command.scene["door"].data
        return (*G1Dynamics.read_sim_state(command),
                data.joint_pos[:, :1].clone(), data.joint_vel[:, :1].clone())

    @staticmethod
    def targets(S, S1):
        return torch.cat((G1Dynamics.targets(S[:6], S1[:6]), S1[6] - S[6], S1[7] - S[7]), -1)

    @staticmethod
    def integrate(S, delta):
        # Limits are physical effects to learn, not clamps that hide model error.
        return (*G1Dynamics.integrate(S[:6], delta[..., :70]),
                S[6] + delta[..., 70:71], S[7] + delta[..., 71:72])

    @staticmethod
    def state_error(predicted, actual):
        return DoorDynamics.targets(actual, predicted)

    @staticmethod
    def error_floor(device):
        return torch.cat((G1Dynamics.error_floor(device), torch.tensor([3e-3, 2e-2], device=device)))
