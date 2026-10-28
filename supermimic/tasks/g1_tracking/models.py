"""Robot tracking Actor, dynamics MLP and value network."""
import torch
from torch import nn

from ...mdp import G1Dynamics
from ...rl.models import RunningNorm, mlp, run_mlp


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

    def encode_geometry(self, context):
        return None

    @torch.no_grad()
    def update_normalizers(self, S, a, context, S1):
        self.in_norm.update(self.inputs(S, a, self.dynamics.normalize_context(context)))
        self.out_norm.update(self.targets(S, S1))

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

    @torch.no_grad()
    def update_normalizers(self, history, goal):
        self.obs_norm.update(history)
        self.goal_norm.update(goal)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, geometry_ids=None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    @torch.no_grad()
    def update_normalizers(self, x):
        self.norm.update(x)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
