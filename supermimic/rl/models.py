"""Small neural-network components shared by task-owned models."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import distributed


TACTILE_CHANNELS = 112
TACTILE_LATENT_DIM = 64
TACTILE_REGION_SIZES = (48, 40, 8, 8, 4, 4)


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        if distributed.active():
            counts, m, v = distributed.global_moments(x)
            n = counts[0]
        else:
            n = x.shape[0]
            if n == 0:
                return
            m, v = x.mean(0), x.var(0, unbiased=False)
        if n == 0:
            return
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None or x.device.type != "cuda":
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class TactileEncoder(nn.Module):
    """Six-region spatial attention and ten-frame temporal attention to a 64-D latent.

    Counts are raw uint8 or floating 0–255 values in the frozen channel order;
    validity is a separate boolean tensor. Invalid counts are removed before
    normalization, and each region's validity mask is part of its MLP input.
    """

    def __init__(self, history_length: int = 10):
        super().__init__()
        if history_length != 10:
            raise ValueError("TactileEncoder requires history_length=10")
        self.history_length = history_length
        self.region_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * size, 64), nn.SiLU(), nn.Linear(64, 64))
            for size in TACTILE_REGION_SIZES
        ])
        self.queries = nn.Parameter(torch.empty(6, 64))
        self.spatial_attention = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)
        self.spatial_norm = nn.LayerNorm(64)
        self.time_embeddings = nn.Parameter(torch.empty(history_length, 64))
        self.temporal = nn.TransformerEncoderLayer(
            64, 4, dim_feedforward=128, dropout=0.0, activation="gelu", batch_first=True,
        )
        self.projection = nn.Linear(6 * 64, TACTILE_LATENT_DIM)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.time_embeddings, std=0.02)

    def forward(self, counts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if (counts.ndim != 3 or counts.shape[1:] != (self.history_length, TACTILE_CHANNELS)
                or valid.shape != counts.shape or valid.dtype != torch.bool
                or (counts.dtype != torch.uint8 and not counts.is_floating_point())):
            raise ValueError("Expected raw counts and bool validity [B,10,112]")
        values = torch.where(valid, counts, 0).to(self.queries.dtype) / 255.0
        validity = valid.to(values.dtype)
        regions, start = [], 0
        for size, mlp in zip(TACTILE_REGION_SIZES, self.region_mlps, strict=True):
            inputs = torch.cat((values[..., start:start + size], validity[..., start:start + size]), dim=-1)
            regions.append(mlp(inputs))
            start += size
        batch = counts.shape[0]
        regional = torch.stack(regions, dim=-2).reshape(batch * self.history_length, 6, 64)
        queries = self.queries.unsqueeze(0).expand(regional.shape[0], -1, -1)
        spatial, _ = self.spatial_attention(queries, regional, regional, need_weights=False)
        spatial = self.spatial_norm(spatial).reshape(batch, self.history_length, 6, 64)
        temporal_input = spatial.transpose(1, 2).reshape(batch * 6, self.history_length, 64)
        temporal = self.temporal(temporal_input + self.time_embeddings)
        tokens = temporal[:, -1].reshape(batch, 6, 64) + spatial[:, -1]
        return self.projection(tokens.flatten(1))


class TensorDictPolicy(nn.Module):
    """Expose the learned policy through mjlab's native observation interface."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observations) -> torch.Tensor:
        history = observations["policy_history"]
        history = history.reshape(history.shape[0], self.policy.hist, -1)
        geometry_ids = observations.get("geometry_id")
        if geometry_ids is not None:
            geometry_ids = geometry_ids.squeeze(-1).long()
        return self.policy(history, observations["goal"], geometry_ids)
