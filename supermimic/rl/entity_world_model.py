"""Task-independent, permutation-equivariant dynamics for G1 and rigid entities.

The geometry bank contains replay metadata (metric positions and unit normals),
not trainable embeddings. All object slots share encoders and normalization
statistics; padding and fixed bodies never contribute a dynamics target.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from supermimic.mdp.state import G1Dynamics
from .models import mlp, run_mlp


class EntityGeometryBank:
    """Deduplicated body-local clouds; ID zero is the absent-geometry sentinel."""

    num_points = 1024
    empty_key = "__none__"
    format_version = 2

    def __init__(self):
        self.keys = [self.empty_key]
        self.points = torch.zeros(1, self.num_points, 6)
        self.extents = torch.zeros(1, 3)
        self._device_cache = {}

    def add(self, key, points, extents):
        key = str(key)
        points = torch.as_tensor(points, dtype=torch.float32).detach().cpu().clone()
        extents = torch.as_tensor(extents, dtype=torch.float32).detach().cpu().clone()
        if points.shape != (self.num_points, 6) or extents.shape != (3,):
            raise ValueError("Entity geometry requires [1024,6] xyz+normal samples and [3] extents")
        if (not torch.isfinite(points).all() or not torch.isfinite(extents).all()
                or (extents < 0).any()):
            raise ValueError("Geometry must be finite, with nonnegative extents")
        if key in self.keys:
            index = self.keys.index(key)
            if not torch.equal(self.points[index], points) or not torch.equal(self.extents[index], extents):
                raise ValueError(f"Geometry key {key!r} already has different content")
            return index
        if extents.max() <= 0:
            raise ValueError("An object must have nonzero extent")
        if not torch.allclose(points[:, 3:].norm(dim=-1), torch.ones(self.num_points), atol=1e-3):
            raise ValueError("Geometry normals must have unit length")
        self.keys.append(key)
        self.points = torch.cat((self.points, points[None]))
        self.extents = torch.cat((self.extents, extents[None]))
        self._device_cache.clear()
        return len(self.keys) - 1

    def tensors(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (self.points.to(device), self.extents.to(device))
        return self._device_cache[key]

    def to(self, device):
        self.tensors(device)
        return self

    def to_dict(self):
        return {"format_version": self.format_version, "keys": list(self.keys),
                "points": self.points.clone(), "extents": self.extents.clone()}

    state_dict = to_dict

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != cls.format_version:
            raise ValueError("Entity geometry bank requires version 2 xyz+normal samples")
        keys, points, extents = data["keys"], data["points"], data["extents"]
        if (not keys or keys[0] != cls.empty_key or len(set(keys)) != len(keys)
                or len(keys) != len(points) or len(keys) != len(extents)):
            raise ValueError("Invalid entity geometry bank IDs")
        bank = cls()
        for key, cloud, size in zip(keys, points, extents):
            bank.add(key, cloud, size)
        return bank

    def load_state_dict(self, data):
        restored = self.from_dict(data)
        self.keys, self.points, self.extents = restored.keys, restored.points, restored.extents
        self._device_cache.clear()

    def merge(self, data):
        """Import a replay bank atomically and return its ID → this bank's ID map."""
        other = self.from_dict(data)
        for key, cloud, size in zip(other.keys, other.points, other.extents):
            if key in self.keys:
                index = self.keys.index(key)
                if not torch.equal(self.points[index], cloud) or not torch.equal(self.extents[index], size):
                    raise ValueError(f"Geometry key {key!r} already has different content")
        return torch.tensor([self.add(key, cloud, size)
                             for key, cloud, size in zip(other.keys, other.points, other.extents)],
                            dtype=torch.long)


class EntityGeometryEncoder(nn.Module):
    """PointNet using meters and unit normals, retaining physical object size."""

    def __init__(self):
        super().__init__()
        self.points = mlp(6, [64], 128, act=nn.SiLU)
        # Explicit metric extents also preserve size when point sampling is sparse.
        self.project = nn.Linear(131, 64)

    def forward(self, points, extents):
        pooled = run_mlp(self.points, points).amax(-2)
        return run_mlp(self.project, torch.cat((pooled, extents), -1))


class EntityRunningNorm(nn.Module):
    """Feature statistics pooled over valid entities, never over slot identities."""

    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(dim))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x, mask=None):
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).float()
        valid = torch.ones_like(x, dtype=torch.bool) if mask is None else torch.broadcast_to(
            mask, original_shape).reshape_as(x).bool()
        valid = valid & (self.count < self.max_count)
        x = torch.where(valid, x, torch.zeros_like(x))
        n = valid.sum(0).float()
        total = (self.count + n).clamp_min(1)
        mean = x.sum(0) / n.clamp_min(1)
        variance = torch.where(valid, (x - mean).square(), torch.zeros_like(x)).sum(0) / n.clamp_min(1)
        delta = mean - self.mean
        updated_variance = (self.var * self.count + variance * n
                            + delta.square() * self.count * n / total) / total
        self.mean.add_(delta * n / total)
        self.var.copy_(torch.where(n > 0, updated_variance, self.var))
        self.count.add_(n)

    def std(self):
        return (self.var + self.eps ** 2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


class EntityOutputNorm(nn.Module):
    """Flat runner interface backed by robot70 + shared object12 statistics."""

    def __init__(self, max_objects):
        super().__init__()
        self.max_objects = max_objects
        self.robot = EntityRunningNorm(70, eps=1e-4)
        self.object = EntityRunningNorm(12, eps=1e-4)

    def _flatten(self, robot, obj):
        return torch.cat((robot, obj.repeat(self.max_objects)), -1)

    @property
    def mean(self):
        return self._flatten(self.robot.mean, self.object.mean)

    @property
    def count(self):
        return self._flatten(self.robot.count, self.object.count)

    def std(self):
        return self._flatten(self.robot.std(), self.object.std())

    def forward(self, x):
        return (x - self.mean) / self.std()

    @torch.no_grad()
    def update(self, x, mask=None):
        self.robot.update(x[..., :70], None if mask is None else mask[..., :70])
        self.object.update(x[..., 70:].reshape(*x.shape[:-1], self.max_objects, 12),
                           None if mask is None else mask[..., 70:].reshape(
                               *mask.shape[:-1], self.max_objects, 12))


class RelationAttention(nn.Module):
    """Attention with relation-conditioned message values and masked receivers."""

    def __init__(self, width, heads, edge_width=32):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.heads, self.head_dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Linear(edge_width, heads)
        self.edge_value = nn.Linear(edge_width, width)
        self.project = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = mlp(width, [2 * width], width, act=nn.SiLU)

    def forward(self, tokens, edges, valid):
        batch, length, width = tokens.shape
        q, k, v = run_mlp(self.qkv, self.norm(tokens)).reshape(
            batch, length, 3, self.heads, self.head_dim).unbind(2)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_dim)
        logits = logits + run_mlp(self.edge_bias, edges).permute(0, 3, 1, 2)
        # Robot and ground are always valid, including in an empty scene.
        attention = logits.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
        # A linear relation projection commutes with the weighted sum. Aggregate
        # the 32-D edges first, avoiding B x L x L x width message tensors in
        # every layer and every differentiable rollout step.
        message = torch.einsum("bhij,bjhd->bihd", attention, v)
        pooled_edges = torch.einsum("bhij,bijc->bihc", attention, edges)
        weights = self.edge_value.weight.reshape(self.heads, self.head_dim, edges.shape[-1])
        message = message + torch.einsum("bihc,hdc->bihd", pooled_edges, weights)
        message = message + self.edge_value.bias.reshape(self.heads, self.head_dim)
        message = message.reshape(batch, length, width)
        tokens = tokens + run_mlp(self.project, message)
        tokens = tokens + run_mlp(self.ff, self.ff_norm(tokens))
        return torch.where(valid[..., None], tokens, torch.zeros_like(tokens))


class EntityWorldModel(nn.Module):
    """One set of weights handles masked sets of interacting rigid entities."""

    def __init__(self, dynamics=None, geometry_bank=None, width=256, layers=4, heads=4):
        super().__init__()
        if dynamics is None:
            from .entity_state import EntityDynamics
            dynamics = EntityDynamics()
        self.dynamics = dynamics
        self.max_objects = dynamics.max_objects
        self.width, self.layers, self.heads = width, layers, heads
        self.geometry_bank = geometry_bank if geometry_bank is not None else EntityGeometryBank()
        self.geometry_encoder = EntityGeometryEncoder()
        self.robot_norm = EntityRunningNorm(133)
        # Geometry embeddings are learned; do not accumulate stale embedding statistics.
        self.object_norm = EntityRunningNorm(25)
        self.ground_norm = EntityRunningNorm(4)
        self.edge_norm = EntityRunningNorm(18)
        self.ground_edge_norm = EntityRunningNorm(13)
        self.out_norm = EntityOutputNorm(self.max_objects)
        self.robot_encoder = mlp(133, [width], width, act=nn.SiLU)
        self.object_encoder = mlp(90, [width], width, act=nn.SiLU)
        self.ground_encoder = mlp(4, [width], width, act=nn.SiLU)
        self.edge_encoder = mlp(18, [64], 32, act=nn.SiLU)
        self.ground_edge_encoder = mlp(13, [64], 32, act=nn.SiLU)
        self.interactions = nn.ModuleList(RelationAttention(width, heads) for _ in range(layers))
        self.robot_head = mlp(width, [width], 70, act=nn.SiLU)
        self.object_head = mlp(width, [width], 12, act=nn.SiLU)
        for head in (self.robot_head, self.object_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def network_schema(self):
        """Version weights independently of runtime object capacity or task."""
        return {"version": 1, "architecture": "entity_relation_attention",
                "width": self.width, "layers": self.layers, "heads": self.heads,
                "robot_input": 133, "object_input": 90, "ground_input": 4,
                "edge_input": 18, "ground_edge_input": 13, "edge_latent": 32,
                "geometry": {"format_version": EntityGeometryBank.format_version,
                             "points": 1024, "channels": 6, "latent": 64,
                             "coordinates": "body_local_meters_and_unit_normals"},
                "normalization": "robot133_object25_ground4_edge18_groundedge13_output70plus12shared",
                "messages": "relative_value_and_attention_bias",
                "outputs": {"robot": 70, "shared_object": 12}}

    def valid_entities(self, context):
        objects = self.dynamics.object_context(context)[..., 10] > 0.5
        one = torch.ones_like(context[..., :1], dtype=torch.bool)
        return torch.cat((one, objects, one), -1)

    def encode_geometry(self, context):
        obj = self.dynamics.object_context(context)
        valid = obj[..., 10] > 0.5
        ids = torch.where(valid, obj[..., 12], torch.zeros_like(obj[..., 12])).long()
        # Deduplicate raw cloud lookup; recompute learned codes on every call.
        unique, inverse = ids.reshape(-1).unique(return_inverse=True)
        if unique.numel() == 0:
            return context.new_zeros(*ids.shape, 64)
        points, extents = self.geometry_bank.tensors(context.device)
        encoded = self.geometry_encoder(points.index_select(0, unique), extents.index_select(0, unique))
        encoded = encoded[inverse].reshape(*ids.shape, 64)
        return torch.where((valid & (ids != 0))[..., None], encoded, torch.zeros_like(encoded))

    def inputs(self, S, action, context):
        """Raw robot133, object25, ground4 and relations before shared statistics."""
        obj = self.dynamics.object_context(context)
        robot = torch.cat((G1Dynamics.features(S[:6]), action,
                           G1Dynamics.normalize_context(context[..., :33])), -1)
        physical = torch.cat((self.dynamics.object_features(S), obj[..., :10]), -1)
        physical = torch.where((obj[..., 10] > 0.5)[..., None], physical, torch.zeros_like(physical))
        edges, ground_edges, is_ground = self.dynamics.relation_features(S, context)
        valid = self.valid_entities(context)
        pairs = valid[..., :, None] & valid[..., None, :]
        edges = torch.where((pairs & ~is_ground)[..., None], edges, torch.zeros_like(edges))
        ground_edges = torch.where((pairs & is_ground)[..., None], ground_edges, torch.zeros_like(ground_edges))
        return robot, physical, self.dynamics.ground_features(S, context), edges, ground_edges, is_ground

    def forward(self, S, action, context):
        robot, physical, ground, edges, ground_edges, is_ground = self.inputs(S, action, context)
        leading = action.shape[:-1]
        valid = self.valid_entities(context)
        obj = self.dynamics.object_context(context)
        physical = self.object_norm(physical)
        objects = torch.cat((physical[..., :15], self.encode_geometry(context),
                             physical[..., 15:], obj[..., 11:12]), -1)
        objects = torch.where(valid[..., 1:-1, None], objects, torch.zeros_like(objects))
        tokens = torch.cat((run_mlp(self.robot_encoder, self.robot_norm(robot)).unsqueeze(-2),
                            run_mlp(self.object_encoder, objects),
                            run_mlp(self.ground_encoder, self.ground_norm(ground)).unsqueeze(-2)), -2)
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        encoded_edges = torch.where(is_ground[..., None],
                                    run_mlp(self.ground_edge_encoder, self.ground_edge_norm(ground_edges)),
                                    run_mlp(self.edge_encoder, self.edge_norm(edges)))
        pair_valid = valid[..., :, None] & valid[..., None, :]
        encoded_edges = torch.where(pair_valid[..., None], encoded_edges, torch.zeros_like(encoded_edges))
        length, width = tokens.shape[-2:]
        tokens = tokens.reshape(-1, length, width)
        encoded_edges = encoded_edges.reshape(-1, length, length, 32)
        for interaction in self.interactions:
            tokens = interaction(tokens, encoded_edges, valid.reshape(-1, length))
        robot_output = run_mlp(self.robot_head, tokens[:, 0]).reshape(*leading, 70)
        object_output = run_mlp(self.object_head, tokens[:, 1:-1]).reshape(*leading, 12 * self.max_objects)
        output = torch.cat((robot_output, object_output), -1)
        return torch.where(self.loss_mask(context).bool(), output, torch.zeros_like(output))

    def loss_mask(self, context):
        obj = self.dynamics.object_context(context)
        moving = (obj[..., 10] > 0.5) & (obj[..., 11] > 0.5)
        return torch.cat((torch.ones_like(context[..., :1]).expand(*context.shape[:-1], 70),
                          moving[..., None].expand(*moving.shape, 12).reshape(
                              *context.shape[:-1], 12 * self.max_objects).to(context.dtype)), -1)

    def predict_delta(self, S, action, context, normalized=False):
        prediction = self(S, action, context)
        if normalized:
            return prediction
        delta = prediction * self.out_norm.std() + self.out_norm.mean
        return torch.where(self.loss_mask(context).bool(), delta, torch.zeros_like(delta))

    def step(self, S, action, context):
        return self.dynamics.integrate(S, self.predict_delta(S, action, context))

    def targets(self, S, S1):
        return self.dynamics.targets(S, S1)

    def state_error(self, predicted, actual):
        return self.dynamics.state_error(predicted, actual)

    def error_per_sample(self, predicted, actual, context, error_floor=None):
        """Equal robot/object-group weight; object count cannot dominate a scene."""
        error = self.state_error(predicted, actual)
        scale = self.out_norm.std()
        if error_floor is not None:
            scale = scale.clamp_min(error_floor)
        error = torch.where(self.loss_mask(context).bool(), error / scale, torch.zeros_like(error))
        robot = error[..., :70].square().mean(-1)
        obj = error[..., 70:].reshape(*error.shape[:-1], self.max_objects, 12).square().mean(-1)
        moving = self.loss_mask(context)[..., 70:].reshape(*error.shape[:-1], self.max_objects, 12)[..., 0]
        count = moving.sum(-1)
        objects = (obj * moving).sum(-1) / count.clamp_min(1)
        return (robot + objects) / (1 + (count > 0).to(robot.dtype))

    @torch.no_grad()
    def update_normalizers(self, S, action, context, S1):
        robot, physical, ground, edges, ground_edges, is_ground = self.inputs(S, action, context)
        valid = self.valid_entities(context)
        pair_valid = valid[..., :, None] & valid[..., None, :]
        self.robot_norm.update(robot)
        self.object_norm.update(physical, valid[..., 1:-1, None])
        self.ground_norm.update(ground)
        self.edge_norm.update(edges, (pair_valid & ~is_ground)[..., None])
        self.ground_edge_norm.update(ground_edges, (pair_valid & is_ground)[..., None])
        self.out_norm.update(self.targets(S, S1), self.loss_mask(context))
