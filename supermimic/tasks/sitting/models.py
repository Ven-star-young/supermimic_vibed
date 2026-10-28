"""Sitting-specific Actor, differentiable dynamics and privileged Critic."""

import torch
from torch import nn

from ...rl.geometry import GeometryEncoder, encode_geometry
from ...rl.models import RunningNorm, mlp, run_mlp
from .dynamics import SittingDynamics


class SittingPolicy(nn.Module):
    """10x102 dynamic history + 32 robot-goal latent + 64 static geometry -> 29 actions."""

    def __init__(self, geometry_bank, *, geometry_encoder=None, history_length=10, goal_frames=10,
                 hidden=(2048, 1024, 512, 256)):
        super().__init__()
        self.hist = history_length
        self.geometry_bank = geometry_bank
        self.geometry_encoder = geometry_encoder if geometry_encoder is not None else GeometryEncoder()
        self.enc = mlp(73 * goal_frames, list(hidden), 32)
        self.dec = mlp(102 * history_length + 32 + 64, list(hidden), 29)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(102)
        self.goal_norm = RunningNorm(73 * goal_frames)

    def forward(self, history, goal, geometry_ids=None):
        if geometry_ids is None:
            if len(self.geometry_bank.keys) != 1:
                raise ValueError("Sitting policy requires geometry IDs when the bank has multiple chairs")
            geometry_ids = torch.zeros(history.shape[:-2], dtype=torch.long, device=history.device)
        if geometry_ids.shape != history.shape[:-2]:
            raise ValueError("Geometry IDs must match the policy batch dimensions")
        with torch.no_grad():
            geometry = encode_geometry(self.geometry_bank, self.geometry_encoder, geometry_ids)
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        observations = self.obs_norm(history).flatten(-2)
        return run_mlp(self.dec, torch.cat((observations, z, geometry), -1))

    @torch.no_grad()
    def update_normalizers(self, history, goal):
        self.obs_norm.update(history[..., -1, :])
        self.goal_norm.update(goal)


class SittingWorldModel(nn.Module):
    """160 physical inputs + 64 geometry -> 82 frame-local state increments."""

    def __init__(self, geometry_bank, *, geometry_encoder=None, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.dynamics = SittingDynamics()
        self.geometry_bank = geometry_bank
        self.geometry_encoder = geometry_encoder if geometry_encoder is not None else GeometryEncoder()
        self.in_dim = 224
        self.net = mlp(224, list(hidden), 82, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(160)
        self.out_norm = RunningNorm(82, eps=1e-4)

    def encode_geometry(self, context):
        return encode_geometry(self.geometry_bank, self.geometry_encoder, context[..., 45].long())

    def numeric_inputs(self, S, action, normalized_context):
        return torch.cat((self.dynamics.features(S), action, normalized_context[..., :45]), -1)

    def inputs(self, S, action, normalized_context):
        return torch.cat((self.numeric_inputs(S, action, normalized_context),
                          self.encode_geometry(normalized_context)), -1)

    def forward(self, S, action, normalized_context):
        numeric = self.in_norm(self.numeric_inputs(S, action, normalized_context))
        return run_mlp(self.net, torch.cat((numeric, self.encode_geometry(normalized_context)), -1))

    def step(self, S, action, normalized_context):
        delta = self(S, action, normalized_context) * self.out_norm.std() + self.out_norm.mean
        return self.dynamics.integrate(S, delta)

    def targets(self, S, S1):
        return self.dynamics.targets(S, S1)

    def state_error(self, predicted, actual):
        return self.dynamics.state_error(predicted, actual)

    @torch.no_grad()
    def update_normalizers(self, S, action, raw_context, S1):
        context = self.dynamics.normalize_context(raw_context)
        self.in_norm.update(self.numeric_inputs(S, action, context))
        self.out_norm.update(self.targets(S, S1))


class SittingCritic(nn.Module):
    """Current privileged state and future references; geometry bypasses running statistics."""

    geometry_start = 217
    geometry_end = 281

    def __init__(self, goal_frames=10):
        super().__init__()
        self.in_dim = 281 + (73 + 9) * goal_frames
        self.net = mlp(self.in_dim, [256, 128], 1)
        self.norm = RunningNorm(self.in_dim - 64)

    def numeric_inputs(self, features):
        return torch.cat((features[..., :self.geometry_start], features[..., self.geometry_end:]), -1)

    def forward(self, features):
        numeric = self.norm(self.numeric_inputs(features))
        inputs = torch.cat((numeric[..., :self.geometry_start],
                            features[..., self.geometry_start:self.geometry_end],
                            numeric[..., self.geometry_start:]), -1)
        return run_mlp(self.net, inputs).squeeze(-1)

    @torch.no_grad()
    def update_normalizers(self, features):
        self.norm.update(self.numeric_inputs(features))
