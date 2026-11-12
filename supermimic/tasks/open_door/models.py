"""Use the shared tracking networks with OpenDoor's tensor dimensions."""

import torch
from torch import nn

from ...rl.models import TACTILE_CHANNELS, TACTILE_LATENT_DIM, TactileEncoder, run_mlp
from ..g1_tracking.models import Critic as TrackingCritic
from ..g1_tracking.models import Policy as TrackingPolicy
from ..g1_tracking.models import WorldModel as TrackingWorldModel
from .dynamics import DoorDynamics


class Policy(TrackingPolicy):
    """State history, optionally followed by tactile counts and validity.

    Experts observe 102 state features; proprio-only students observe the first
    93. Tactile frames append 112 raw 0–255 counts and 112 validity values.
    Only the state prefix uses running normalization.
    """

    def __init__(self, history_length=10, goal_frames=10, *, use_tactile: bool = False,
                 proprio_only: bool = False):
        if not isinstance(use_tactile, bool):
            raise ValueError("use_tactile must be boolean")
        if not isinstance(proprio_only, bool):
            raise ValueError("proprio_only must be boolean")
        state_observation_dim = 93 if proprio_only else 102
        super().__init__(obs_dim=state_observation_dim, goal_dim=75 * goal_frames, hist=history_length)
        self.proprio_only = proprio_only
        self.state_observation_dim = state_observation_dim
        self.use_tactile = use_tactile
        self.tactile_encoder = TactileEncoder(history_length) if use_tactile else None
        if use_tactile:
            self.dec[0] = nn.Linear(self.dec[0].in_features + TACTILE_LATENT_DIM, self.dec[0].out_features)

    def _tactile_history(self, history: torch.Tensor) -> torch.Tensor:
        frame_dim = self.state_observation_dim + 2 * TACTILE_CHANNELS
        if history.ndim != 3 or history.shape[1:] != (self.hist, frame_dim):
            raise ValueError(f"Expected tactile history [B,{self.hist},{frame_dim}], got {tuple(history.shape)}")
        return history

    @torch.no_grad()
    def update_normalizers(self, history, goal):
        if self.use_tactile:
            frame_dim = self.state_observation_dim + 2 * TACTILE_CHANNELS
            if (history.ndim != 3 or history.shape[-1] != frame_dim
                    or history.shape[0] != goal.shape[0]):
                raise ValueError(f"Expected tactile normalization frames [B,T,{frame_dim}], got {tuple(history.shape)}")
            history = history[..., :self.state_observation_dim]
        super().update_normalizers(history, goal)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, geometry_ids=None) -> torch.Tensor:
        if not self.use_tactile:
            return super().forward(obs_hist, goal, geometry_ids)
        history = self._tactile_history(obs_hist)
        if (goal.ndim != 2 or goal.shape[-1] != self.goal_norm.mean.numel()
                or goal.shape[0] != history.shape[0]):
            raise ValueError(f"Expected goal [B,{self.goal_norm.mean.numel()}], got {tuple(goal.shape)}")
        counts = history[..., self.state_observation_dim:self.state_observation_dim + TACTILE_CHANNELS]
        valid = history[..., self.state_observation_dim + TACTILE_CHANNELS:].bool()
        tactile = self.tactile_encoder(counts, valid)
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        state = self.obs_norm(history[..., :self.state_observation_dim]).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, state, tactile), dim=-1))


class WorldModel(TrackingWorldModel):
    """80 state features + 29 actions + 43 context values -> 72 increments."""

    def __init__(self, dynamics=None, hidden=(2048, 2048, 2048)):
        super().__init__(hidden=hidden, dynamics=DoorDynamics() if dynamics is None else dynamics)

    def inputs(self, S, action, normalized_context):
        # Hinge xyz already enters through relative features, so avoid adding
        # absolute coordinates as a second input to the dynamics MLP.
        return torch.cat((self.dynamics.features(S, normalized_context), action,
                          normalized_context[..., :33], normalized_context[..., 36:46]), -1)


class Critic(TrackingCritic):
    """205 current privileged features plus 75 channels per future frame."""

    def __init__(self, goal_frames=10):
        self.in_dim = 205 + 75 * goal_frames
        super().__init__(self.in_dim)
