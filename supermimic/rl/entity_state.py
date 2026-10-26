"""Fixed per-entity physics contract; the number of active objects is masked.

Flat replay is an adapter for the existing replay buffer. Networks consume entity
axes, not slot-index-specific weights. Context contains no task targets/references.
"""
from __future__ import annotations

import torch

from ..mdp.state import G1Dynamics
from ..rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat


def rotate(R, v):
    return (R @ v.unsqueeze(-1)).squeeze(-1)


class EntityDynamics:
    action_dim = 29
    state_components = 10
    control_dt = .02

    def __init__(self, max_objects=8):
        if not isinstance(max_objects, int) or not 1 <= max_objects <= 32:
            raise ValueError("max_objects must be an integer in [1,32]")
        self.max_objects = max_objects
        self.num_entities = max_objects + 2
        self.state_dim = 71 + 13 * max_objects
        self.output_dim = 70 + 12 * max_objects
        self.object_context_start = 33
        self.pair_start = 33 + 13 * max_objects
        self.plane_start = self.pair_start + 3 * self.num_entities ** 2
        self.context_dim = self.plane_start + 4

    def schema(self):
        return dict(
            version=1, kind="g1-entity-physics", max_objects=self.max_objects,
            state_dim=self.state_dim, action_dim=29, context_dim=self.context_dim,
            output_dim=self.output_dim, control_dt=self.control_dt,
            robot="g1_29dof_rev_1_0_handless_geometry_original_inertia",
            quaternion="wxyz", velocity_origin="body", ground="unit_normal_offset_plane",
            action="scaled_joint_position_offset_with_calibration",
            object_delta="robot_heading_position_velocity;object_body_rotation",
            object_context="mass,com3,inertia_xx_yy_zz_xy_xz_yz,valid,dynamic,geometry_id",
            pair_context="robot,objects,ground;sliding,torsional,rolling",
            geometry="1024_metric_body_local_xyz_and_outward_normals",
        )

    def compatible_physics_schema(self, schema):
        """Weights share per-entity semantics even when replay capacity differs."""
        if not isinstance(schema, dict):
            return False
        capacity = {"max_objects", "state_dim", "context_dim", "output_dim"}
        return ({k: v for k, v in schema.items() if k not in capacity}
                == {k: v for k, v in self.schema().items() if k not in capacity})

    def object_context(self, context):
        return context[..., 33:self.pair_start].reshape(*context.shape[:-1], self.max_objects, 13)

    def pair_friction(self, context):
        return context[..., self.pair_start:self.plane_start].reshape(
            *context.shape[:-1], self.num_entities, self.num_entities, 3)

    def ground_context(self, context):
        return context[..., self.plane_start:self.plane_start + 4]

    def empty_context(self, *batch_shape, device=None, dtype=torch.float32):
        context = torch.zeros(*batch_shape, self.context_dim, device=device, dtype=dtype)
        self.ground_context(context)[..., 2] = 1
        return context

    @staticmethod
    def normalize_context(context):
        # Entity networks normalize each semantic block with shared statistics.
        # Preserve masks, lookup IDs and physical pair parameters here.
        return context

    def pack(self, state):
        p, R, v, w = state[6:]
        objects = torch.cat((p, mat_to_quat(R), v, w), -1)
        return torch.cat((G1Dynamics.pack(state[:6]), objects.flatten(-2)), -1)

    def unpack(self, packed):
        if packed.shape[-1] != self.state_dim:
            raise ValueError(f"Expected {self.state_dim} entity state values")
        obj = packed[..., 71:].reshape(*packed.shape[:-1], self.max_objects, 13)
        return (*G1Dynamics.unpack(packed[..., :71]), obj[..., :3], quat_to_mat(obj[..., 3:7]),
                obj[..., 7:10], obj[..., 10:13])

    def read_sim_state(self, command):
        robot = G1Dynamics.read_sim_state(command)
        zeros = robot[0].new_zeros((*robot[0].shape[:-1], self.max_objects, 3))
        p, v, w = zeros.clone(), zeros.clone(), zeros.clone()
        R = torch.eye(3, device=p.device, dtype=p.dtype).expand(
            *p.shape[:-1], 3, 3).clone()
        names = command.object_entity_names
        if len(names) > self.max_objects:
            raise ValueError("Scene has more objects than the configured entity capacity")
        for i, name in enumerate(names):
            data = command.scene[name].data
            p[..., i, :] = data.root_link_pos_w - command.scene.env_origins
            R[..., i, :, :] = quat_to_mat(data.root_link_quat_w)
            v[..., i, :] = data.root_link_lin_vel_w
            w[..., i, :] = data.root_link_ang_vel_w
        return (*robot, p, R, v, w)

    @staticmethod
    def object_features(state):
        p, R, v, w = state[6:]
        Ht = yaw_mat(state[1]).transpose(-1, -2).unsqueeze(-3)
        return torch.cat((rotate(Ht, p - state[0].unsqueeze(-2)), mat_to_6d(Ht @ R),
                          rotate(Ht, v), rotate(Ht, w)), -1)

    def ground_features(self, state, context):
        plane = self.ground_context(context)
        normal, offset = plane[..., :3], plane[..., 3:4]
        height = (state[0] * normal).sum(-1, keepdim=True) - offset
        return torch.cat((height, rotate(yaw_mat(state[1]).transpose(-1, -2), normal)), -1)

    def relation_features(self, state, context):
        """Directed edges [receiver, sender], with physical (symmetric) friction.

        Finite-body edges: local relative position/Rot6D/velocities plus friction.
        Floor edges: finite body's distance/local normal/velocities plus friction.
        Both directions use the same body's floor relation; role is in node type.
        """
        p, R, v, w = state[:4]
        op, oR, ov, ow = state[6:]
        plane = self.ground_context(context)
        normal = plane[..., :3]
        floor_p = p - ((normal * p).sum(-1, keepdim=True) - plane[..., 3:4]) * normal
        identity = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        positions = torch.cat((p.unsqueeze(-2), op, floor_p.unsqueeze(-2)), -2)
        rotations = torch.cat((R.unsqueeze(-3), oR, identity.unsqueeze(-3)), -3)
        velocities = torch.cat((v.unsqueeze(-2), ov, torch.zeros_like(v).unsqueeze(-2)), -2)
        angular = torch.cat((w.unsqueeze(-2), ow, torch.zeros_like(w).unsqueeze(-2)), -2)
        Rt = rotations.transpose(-1, -2)
        local = Rt.unsqueeze(-3)
        dp = rotate(local, positions.unsqueeze(-3) - positions.unsqueeze(-2))
        dR = local @ rotations.unsqueeze(-4)
        dv = rotate(local, velocities.unsqueeze(-3) - velocities.unsqueeze(-2))
        dw = rotate(local, angular.unsqueeze(-3) - angular.unsqueeze(-2))
        friction = self.pair_friction(context)
        finite = torch.cat((dp, mat_to_6d(dR), dv, dw, friction), -1)
        distance = (positions * normal.unsqueeze(-2)).sum(-1, keepdim=True) - plane[..., 3:4].unsqueeze(-2)
        body_floor = torch.cat((distance, rotate(Rt, normal.unsqueeze(-2)),
                                rotate(Rt, velocities), rotate(Rt, angular)), -1)
        slots = torch.arange(self.num_entities, device=p.device)
        receiver_floor = slots[:, None] == self.num_entities - 1
        sender_floor = slots[None, :] == self.num_entities - 1
        to_ground = body_floor.unsqueeze(-2).expand(*body_floor.shape[:-2], self.num_entities, self.num_entities, 10)
        from_ground = body_floor.unsqueeze(-3).expand_as(to_ground)
        ground = torch.cat((torch.where(receiver_floor[..., None], from_ground, to_ground), friction), -1)
        # A plane has no meaningful own body orientation. Its self-edge must
        # not leak world-axis orientation into otherwise heading-local inputs.
        ground = torch.where((receiver_floor & sender_floor)[..., None], torch.zeros_like(ground), ground)
        return finite, ground, receiver_floor | sender_floor

    @staticmethod
    def targets(state, following):
        p, R, v, w = state[6:]
        p1, R1, v1, w1 = following[6:]
        Ht = yaw_mat(state[1]).transpose(-1, -2).unsqueeze(-3)
        delta = torch.cat((rotate(Ht, p1 - p), so3_log(R.transpose(-1, -2) @ R1),
                           rotate(Ht, v1 - v), rotate(Ht, w1 - w)), -1)
        return torch.cat((G1Dynamics.targets(state[:6], following[:6]), delta.flatten(-2)), -1)

    def integrate(self, state, delta):
        p, R, v, w = state[6:]
        obj = delta[..., 70:].reshape(*delta.shape[:-1], self.max_objects, 12)
        dp, dtheta, dv, dw = obj.split(3, -1)
        H = yaw_mat(state[1]).unsqueeze(-3)
        return (*G1Dynamics.integrate(state[:6], delta[..., :70]),
                p + rotate(H, dp), R @ so3_exp(dtheta), v + rotate(H, dv), w + rotate(H, dw))

    def state_error(self, predicted, actual):
        return self.targets(actual, predicted)

    def error_floor(self, device):
        obj = torch.tensor([3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3, device=device)
        return torch.cat((G1Dynamics.error_floor(device), obj.repeat(self.max_objects)))

    def validate_context(self, context, geometry_count=None):
        if context.shape[-1] != self.context_dim or not torch.isfinite(context).all():
            raise ValueError("Invalid entity context shape or nonfinite values")
        obj = self.object_context(context)
        valid, dynamic, ids = obj[..., 10], obj[..., 11], obj[..., 12]
        if not (((valid == 0) | (valid == 1)) & ((dynamic == 0) | (dynamic == 1))).all():
            raise ValueError("Entity masks must be binary")
        if (dynamic > valid).any() or not torch.equal(ids, ids.round()) or (ids < 0).any():
            raise ValueError("Invalid entity mask or geometry ID")
        if not torch.equal(ids > 0, valid.bool()) or (geometry_count is not None and (ids >= geometry_count).any()):
            raise ValueError("Geometry ID and entity presence disagree")
        if ((obj[..., 0] <= 0) & dynamic.bool()).any():
            raise ValueError("Dynamic object mass must be positive")
        friction = self.pair_friction(context)
        if (friction < 0).any() or not torch.allclose(friction, friction.transpose(-3, -2)):
            raise ValueError("Pair friction must be nonnegative and symmetric")
        if not torch.allclose(self.ground_context(context)[..., :3].norm(dim=-1), torch.ones_like(valid[..., 0]), atol=1e-5):
            raise ValueError("Ground plane normal must be unit length")
