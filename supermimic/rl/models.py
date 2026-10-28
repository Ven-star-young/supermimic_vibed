"""Small neural-network components shared by task-owned models."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import distributed


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
