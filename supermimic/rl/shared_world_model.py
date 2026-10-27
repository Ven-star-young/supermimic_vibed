"""Task-independent G1 + one rigid-body dynamics, geometry and masked statistics.

Replay stores 84 state values and 47 context values. The last two context
values are presence and geometry-bank ID; the ID is lookup metadata and never
enters the dynamics MLP. All positions/velocities refer to body origins.
"""

from __future__ import annotations

import torch
from torch import nn

from supermimic.mdp.state import G1Dynamics
from supermimic.rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat
from .models import mlp, run_mlp
from . import distributed


def _rotate(rotation, vector):
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


class SharedDynamics:
    """Robot tuple ``(p,R,v,w,q,qd)`` followed by object ``(p,R,v,w)``."""

    state_dim, action_dim, context_dim = 84, 29, 47
    feature_dim, input_dim, output_dim, state_components = 86, 228, 82, 10

    @staticmethod
    def schema():
        return {
            "version": 1, "robot": "g1_29dof", "object_slots": 1,
            "state_dim": 84, "action_dim": 29, "context_dim": 47,
            "feature_dim": 86, "input_dim": 228, "output_dim": 82,
            "control_dt": 0.02, "quaternion": "wxyz", "velocity_origin": "body",
            "action": "scaled_joint_position_offset_with_calibration",
            "object_delta_frame": "robot_heading;rotation=object_body",
            "physical_context": "robot33,mass,com3,inertia_xx_yy_zz_xy_xz_yz,mu_robot,mu_ground",
            "context_metadata": ["presence", "geometry_id"],
            "geometry": {"points": 1024, "latent": 64, "frame": "object_body_meters"},
        }

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
        return torch.cat((G1Dynamics.features(S[:6]), SharedDynamics.object_features(S)), -1)

    @staticmethod
    def normalize_context(context):
        obj = torch.cat(((context[..., 33:34] - 10) / 5, context[..., 34:45]), -1)
        return torch.cat((G1Dynamics.normalize_context(context[..., :33]),
                          obj * context[..., 45:46], context[..., 45:47]), -1)

    @staticmethod
    def read_sim_state(command):
        robot = G1Dynamics.read_sim_state(command)
        if "chair" not in command.scene.entities:
            zero = torch.zeros_like(robot[0])
            identity = torch.eye(3, device=zero.device, dtype=zero.dtype).expand_as(robot[1]).clone()
            return (*robot, zero.clone(), identity, zero.clone(), zero.clone())
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
        return SharedDynamics.targets(actual, predicted)

    @staticmethod
    def error_floor(device):
        return torch.cat((G1Dynamics.error_floor(device),
                          torch.tensor([3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3,
                                       device=device)))


class GeometryBank:
    """Deduplicated metric body-local point clouds; slot zero means no object.

    This is replay metadata, not a learned parameter. Save it separately from
    network weights, so loading a shared model into another task can retain
    that task's geometry IDs. Replays must travel with their original bank.
    """

    num_points = 1024
    empty_key = "__none__"

    def __init__(self):
        self.keys = [self.empty_key]
        self.points = torch.zeros(1, self.num_points, 3)
        self.extents = torch.zeros(1, 3)
        self._device_cache = {}

    def add(self, key, points, extents):
        key = str(key)
        points = torch.as_tensor(points, dtype=torch.float32).detach().cpu().clone()
        extents = torch.as_tensor(extents, dtype=torch.float32).detach().cpu().clone()
        if points.shape != (self.num_points, 3) or extents.shape != (3,):
            raise ValueError("Geometry requires [1024,3] points and [3] extents")
        if not torch.isfinite(points).all() or not torch.isfinite(extents).all() or (extents < 0).any():
            raise ValueError("Geometry must be finite, with nonnegative extents")
        if key in self.keys:
            index = self.keys.index(key)
            if not torch.equal(self.points[index], points) or not torch.equal(self.extents[index], extents):
                raise ValueError(f"Geometry key {key!r} already has different content")
            return index
        if extents.max() <= 0:
            raise ValueError("An object must have nonzero extent")
        self.keys.append(str(key))
        self.points = torch.cat((self.points, points[None]))
        self.extents = torch.cat((self.extents, extents[None]))
        self._device_cache.clear()
        return len(self.keys) - 1

    def tensors(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (self.points.to(device), self.extents.to(device))
        return self._device_cache[key]

    def to_dict(self):
        return {"keys": list(self.keys), "points": self.points.clone(), "extents": self.extents.clone()}

    state_dict = to_dict

    def load_state_dict(self, data):
        restored = self.from_dict(data)
        self.keys, self.points, self.extents = restored.keys, restored.points, restored.extents
        self._device_cache.clear()

    def merge(self, data):
        """Import another replay's geometries and return its ID -> this ID map."""
        other = self.from_dict(data)
        # Check conflicts before mutating this bank.
        for key, points, extents in zip(other.keys, other.points, other.extents):
            if key in self.keys:
                index = self.keys.index(key)
                if not torch.equal(self.points[index], points) or not torch.equal(self.extents[index], extents):
                    raise ValueError(f"Geometry key {key!r} already has different content")
        return torch.tensor([self.add(key, points, extents)
                             for key, points, extents in zip(other.keys, other.points, other.extents)],
                            dtype=torch.long)

    def to(self, device):
        self.tensors(device)
        return self

    @classmethod
    def from_dict(cls, data):
        bank = cls()
        keys, points, extents = data["keys"], data["points"], data["extents"]
        if (not keys or keys[0] != cls.empty_key or len(set(keys)) != len(keys)
                or len(keys) != len(points) or len(keys) != len(extents)):
            raise ValueError("Invalid geometry bank IDs")
        for key, cloud, size in zip(keys, points, extents):
            bank.add(key, cloud, size)
        return bank


class GeometryEncoder(nn.Module):
    """PointNet over isotropically scaled points, retaining the body origin."""

    def __init__(self):
        super().__init__()
        self.points = mlp(3, [64], 128, act=nn.SiLU)
        self.project = nn.Linear(128, 64)

    def forward(self, points, extents):
        scale = extents.amax(-1, keepdim=True).clamp_min(1e-6)
        shape = self.project(self.points(points / scale[..., None]).amax(-2))
        return torch.cat((shape, extents), -1)


class MaskedRunningNorm(nn.Module):
    """Independent per-feature counts exclude absent object slots from statistics."""

    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(dim))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x, mask=None):
        if mask is not None:
            mask = torch.broadcast_to(mask, x.shape).reshape(-1, x.shape[-1]).float()
        x = x.reshape(-1, x.shape[-1]).float()
        mask = torch.ones_like(x) if mask is None else mask
        mask = mask * (self.count < self.max_count)
        n = mask.sum(0)
        if distributed.active():
            n, mean, var = distributed.global_moments(x, mask)
        else:
            mean = (x * mask).sum(0) / n.clamp_min(1)
            var = ((x - mean).square() * mask).sum(0) / n.clamp_min(1)
        total = (self.count + n).clamp_min(1)
        delta = mean - self.mean
        new_var = (self.var * self.count + var * n + delta.square() * self.count * n / total) / total
        self.mean.add_(delta * n / total)
        self.var.copy_(torch.where(n > 0, new_var, self.var))
        self.count.add_(n)

    def forward(self, x):
        return (x - self.mean) / self.std()

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


class SharedWorldModel(nn.Module):
    """One checkpoint predicts robot/object transitions across reward definitions."""

    def __init__(self, dynamics=None, geometry_bank=None, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.dynamics = dynamics if dynamics is not None else SharedDynamics()
        self.geometry_bank = geometry_bank if geometry_bank is not None else GeometryBank()
        self.geometry_encoder = GeometryEncoder()
        self.in_dim = self.dynamics.input_dim
        self.net = mlp(self.in_dim, list(hidden), self.dynamics.output_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = MaskedRunningNorm(self.in_dim)
        self.out_norm = MaskedRunningNorm(self.dynamics.output_dim, eps=1e-4)

    def encode_geometry(self, context):
        """67 features; geometry ID and presence are unchanged by context normalization."""
        indices, inverse = context[..., 46].long().reshape(-1).unique(return_inverse=True)
        points, extents = self.geometry_bank.tensors(context.device)
        encoded = self.geometry_encoder(points.index_select(0, indices), extents.index_select(0, indices))
        return encoded[inverse].reshape(*context.shape[:-1], 67) * context[..., 45:46]

    @staticmethod
    def _input_mask(context, include_presence=True):
        mask = context[..., 45:46]
        one = torch.ones_like(mask)
        return torch.cat((one.expand(*one.shape[:-1], 71), mask.expand(*mask.shape[:-1], 15),
                          one.expand(*one.shape[:-1], 62), mask.expand(*mask.shape[:-1], 79),
                          one if include_presence else torch.zeros_like(one)), -1)

    @staticmethod
    def loss_mask(context):
        one = torch.ones_like(context[..., :1])
        return torch.cat((one.expand(*one.shape[:-1], 70), context[..., 45:46].expand(*one.shape[:-1], 12)), -1)

    def inputs(self, S, action, normalized_context):
        return torch.cat((self.dynamics.features(S), action, normalized_context[..., :45],
                          self.encode_geometry(normalized_context), normalized_context[..., 45:46]), -1)

    def normalized_inputs(self, S, action, normalized_context):
        features = self.in_norm(self.inputs(S, action, normalized_context))
        # Keep the binary presence channel literal; absent slots must remain zero
        # even after fitting statistics to populated scenes.
        return torch.cat((features[..., :-1] * self._input_mask(normalized_context)[..., :-1],
                          normalized_context[..., 45:46]), -1)

    def forward(self, S, action, normalized_context):
        """Normalized regression output; use loss_mask for transition losses."""
        return run_mlp(self.net, self.normalized_inputs(S, action, normalized_context))

    def predict_delta(self, S, action, normalized_context, normalized=False):
        y = self(S, action, normalized_context)
        return y if normalized else (y * self.out_norm.std() + self.out_norm.mean) * self.loss_mask(normalized_context)

    def step(self, S, action, normalized_context):
        return self.dynamics.integrate(S, self.predict_delta(S, action, normalized_context))

    def targets(self, S, S1):
        return self.dynamics.targets(S, S1)

    def state_error(self, predicted, actual):
        return self.dynamics.state_error(predicted, actual)

    @torch.no_grad()
    def update_normalizers(self, S, action, context, S1):
        normalized = self.dynamics.normalize_context(context)
        features = self.inputs(S, action, normalized)
        # Flatten here because update also supports arbitrary replay-window batch axes.
        mask = self._input_mask(context, include_presence=False)
        self.in_norm.update(features.reshape(-1, self.in_dim), mask.reshape(-1, self.in_dim))
        target = self.targets(S, S1)
        self.out_norm.update(target.reshape(-1, self.dynamics.output_dim),
                             self.loss_mask(context).reshape(-1, self.dynamics.output_dim))
