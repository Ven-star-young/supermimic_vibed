"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import distributed

from supermimic.mdp import G1Dynamics


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
            count, m, v = distributed.global_moments(x)
            n = count[0]
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


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(self, hidden=(2048, 2048, 2048), dynamics=None):
        super().__init__()
        self.dynamics = dynamics if dynamics is not None else G1Dynamics()
        self.in_dim = self.dynamics.input_dim
        self.net = mlp(self.in_dim, list(hidden), self.dynamics.output_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.dynamics.output_dim, eps=1e-4)

    def inputs(self, S, a, xi_n):
        return torch.cat((self.dynamics.features(S), a, xi_n), -1)

    def targets(self, S, S1):
        """Task-specific regression targets without hard-coding a robot state layout."""
        return self.dynamics.targets(S, S1)

    def step(self, S, a, xi_n):
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        delta = y * self.out_norm.std() + self.out_norm.mean
        return self.dynamics.integrate(S, delta)

    def state_error(self, S_hat, S):
        return self.dynamics.state_error(S_hat, S)


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32, action_dim: int = 29):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], action_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


class TensorDictPolicy(nn.Module):
    """Expose the learned policy through mjlab's native observation interface."""

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations) -> torch.Tensor:
        history = observations["policy_history"]
        history = history.reshape(history.shape[0], self.policy.hist, -1)
        return self.policy(history, observations["goal"])
