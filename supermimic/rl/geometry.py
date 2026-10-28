"""Metric point clouds and a shared PointNet; geometry IDs are lookup metadata."""

import torch
from torch import nn

from .models import mlp, run_mlp


class GeometryBank:
    """Deduplicated local xyz + normal samples; every ID names a real asset."""

    num_points = 1024
    format_version = 3

    def __init__(self):
        self.keys = []
        self.points = torch.empty(0, self.num_points, 6)
        self.extents = torch.empty(0, 3)
        self._device_cache = {}

    def add(self, key, points, extents):
        key = str(key)
        points = torch.as_tensor(points, dtype=torch.float32).detach().cpu().clone()
        extents = torch.as_tensor(extents, dtype=torch.float32).detach().cpu().clone()
        if points.shape != (self.num_points, 6) or extents.shape != (3,):
            raise ValueError("Geometry requires [1024,6] xyz+normal samples and [3] extents")
        if (not torch.isfinite(points).all() or not torch.isfinite(extents).all()
                or (extents < 0).any() or extents.max() <= 0):
            raise ValueError("Geometry must be finite and have nonzero, nonnegative extents")
        if not torch.allclose(points[:, 3:].norm(dim=-1), torch.ones(self.num_points), atol=1e-3):
            raise ValueError("Geometry normals must have unit length")
        if key in self.keys:
            index = self.keys.index(key)
            if not torch.equal(self.points[index], points) or not torch.equal(self.extents[index], extents):
                raise ValueError(f"Geometry key {key!r} already has different content")
            return index
        self.keys.append(key)
        self.points = torch.cat((self.points, points[None]))
        self.extents = torch.cat((self.extents, extents[None]))
        self._device_cache.clear()
        return len(self.keys) - 1

    def tensors(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = self.points.to(device), self.extents.to(device)
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
            raise ValueError("Geometry bank requires version 3: xyz+normals without absent slots")
        keys, points, extents = data["keys"], data["points"], data["extents"]
        if len(set(keys)) != len(keys) or len(keys) != len(points) or len(keys) != len(extents):
            raise ValueError("Invalid geometry bank IDs")
        bank = cls()
        for key, cloud, size in zip(keys, points, extents):
            bank.add(key, cloud, size)
        return bank

    def load_state_dict(self, data):
        restored = self.from_dict(data)
        self.keys, self.points, self.extents = restored.keys, restored.points, restored.extents
        self._device_cache.clear()


class GeometryEncoder(nn.Module):
    """1024 metric xyz+normal samples and metric extents become 64 features."""

    def __init__(self):
        super().__init__()
        self.points = mlp(6, [64], 128, act=nn.SiLU)
        self.project = nn.Linear(131, 64)

    def forward(self, points, extents):
        pooled = run_mlp(self.points, points).amax(-2)
        return run_mlp(self.project, torch.cat((pooled, extents), -1))


def encode_geometry(bank, encoder, geometry_ids):
    """Encode each requested asset once with current weights, without embedding caches."""
    indices, inverse = geometry_ids.long().reshape(-1).unique(return_inverse=True)
    points, extents = bank.tensors(geometry_ids.device)
    encoded = encoder(points.index_select(0, indices), extents.index_select(0, indices))
    return encoded[inverse].reshape(*geometry_ids.shape, 64)
