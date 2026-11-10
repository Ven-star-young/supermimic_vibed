"""OpenDoor student with proprioception, motion goals, and channel tactile history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...rl.models import run_mlp
from ...tasks.g1_tracking.models import Policy as _BasePolicy


HISTORY_LENGTH = 10
PROPRIO_DIM = 93
DOOR_DIM = 9
OBS_DIM = PROPRIO_DIM + DOOR_DIM
GOAL_DIM = 75 * HISTORY_LENGTH
ACTION_DIM = 29
STUDENT_INPUT_DIM = PROPRIO_DIM * HISTORY_LENGTH
TACTILE_CHANNELS = 112
TACTILE_LATENT_DIM = 64
TACTILE_REGION_SIZES = (48, 40, 8, 8, 4, 4)


def _as_history(value: torch.Tensor, *, frame_dim: int, history_length: int) -> torch.Tensor:
    if value.ndim == 2:
        expected = history_length * frame_dim
        if value.shape[-1] != expected:
            raise ValueError(f"Expected flattened history (..., {expected}), got {tuple(value.shape)}")
        return value.reshape(value.shape[0], history_length, frame_dim)
    if value.ndim == 3 and value.shape[-2:] == (history_length, frame_dim):
        return value
    raise ValueError(
        f"Expected history [B,{history_length},{frame_dim}] or [B,{history_length * frame_dim}], "
        f"got {tuple(value.shape)}"
    )


def split_open_door_history(
    policy_history: torch.Tensor,
    *,
    history_length: int = HISTORY_LENGTH,
    expert_observation_dim: int = OBS_DIM,
    proprio_dim: int = PROPRIO_DIM,
) -> torch.Tensor:
    """Extract the observable proprioceptive prefix from OpenDoor history."""

    history = _as_history(
        policy_history,
        frame_dim=expert_observation_dim,
        history_length=history_length,
    )
    return history[..., :proprio_dim]


class TactileHistory:
    """Oldest-to-newest channel frames; reset worlds repeat their first observation."""

    def __init__(self, num_envs: int, device: torch.device | str, history_length: int = HISTORY_LENGTH):
        if num_envs < 1 or history_length != HISTORY_LENGTH:
            raise ValueError("Tactile history requires positive num_envs and history_length=10")
        self.counts = torch.zeros((num_envs, history_length, TACTILE_CHANNELS), dtype=torch.uint8, device=device)
        self.valid = torch.zeros_like(self.counts, dtype=torch.bool)
        self._pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._step_id: int | None = None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._pending.fill_(True)
        else:
            self._pending[env_ids] = True

    @torch.no_grad()
    def update(self, counts: torch.Tensor, valid: torch.Tensor, step_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Append once per step ID; repeated reads only backfill pending reset worlds.

        Inputs are uint8/bool ``[num_envs,112]`` on the history device. Returned
        buffers are reused; clone them when retaining observations across updates.
        """
        shape = (self.counts.shape[0], TACTILE_CHANNELS)
        if (counts.shape != shape or valid.shape != shape or counts.dtype != torch.uint8
                or valid.dtype != torch.bool or counts.device != self.counts.device
                or valid.device != self.counts.device or type(step_id) is not int):
            raise ValueError("Expected uint8 counts and bool validity [num_envs,112] on the history device, and int step_id")
        counts = counts.masked_fill(~valid, 0)
        if step_id != self._step_id:
            self.counts[:, :-1] = self.counts[:, 1:].clone()
            self.valid[:, :-1] = self.valid[:, 1:].clone()
            self.counts[:, -1] = counts
            self.valid[:, -1] = valid
        pending = self._pending[:, None, None]
        self.counts.copy_(torch.where(pending, counts[:, None], self.counts))
        self.valid.copy_(torch.where(pending, valid[:, None], self.valid))
        self._pending.zero_()
        self._step_id = step_id
        return self.counts, self.valid


class TactileEncoder(nn.Module):
    """Six-region spatial attention and ten-frame temporal attention to a 64-D latent.

    Counts are raw uint8 or floating 0–255 values in the frozen channel order;
    validity is a separate boolean tensor. Invalid counts are removed before
    normalization, and each region's validity mask is part of its MLP input.
    """

    def __init__(self, history_length: int = HISTORY_LENGTH):
        super().__init__()
        if history_length != HISTORY_LENGTH:
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


class OpenDoorStudentPolicy(_BasePolicy):
    """OpenDoor goal/actor widths with a tactile latent replacing direct door state.

    Input:
        ``proprio_history``: ``[B, 10, 93]`` (or flattened ``[B, 930]``)
        ``goal``: ``[B, 750]``
        ``tactile_history``, ``tactile_valid``: raw counts/bool ``[B, 10, 112]``
    Output:
        normalized 29-D joint action consumed by the existing action manager.
    """

    def __init__(self, history_length: int = HISTORY_LENGTH, goal_frames: int = HISTORY_LENGTH):
        if history_length != HISTORY_LENGTH:
            raise ValueError("The first OpenDoor student contract requires history_length=10")
        super().__init__(
            obs_dim=PROPRIO_DIM,
            goal_dim=75 * goal_frames,
            hist=history_length,
            action_dim=ACTION_DIM,
        )
        self.tactile_encoder = TactileEncoder(history_length)
        self.dec[0] = nn.Linear(self.dec[0].in_features + TACTILE_LATENT_DIM, self.dec[0].out_features)

    @torch.no_grad()
    def initialize_goal_from_expert(self, expert: nn.Module) -> None:
        """Copy compatible goal normalization/encoder weights from an expert."""

        for name in ("enc", "goal_norm"):
            source, target = getattr(expert, name, None), getattr(self, name, None)
            if source is None or target is None:
                raise TypeError(f"Expert lacks compatible {name} module")
            target.load_state_dict(source.state_dict())

    def forward(self, proprio_history: torch.Tensor, goal: torch.Tensor,
                tactile_history: torch.Tensor, tactile_valid: torch.Tensor):
        history = _as_history(
            proprio_history,
            frame_dim=PROPRIO_DIM,
            history_length=self.hist,
        )
        if (goal.ndim != 2 or goal.shape[-1] != self.goal_norm.mean.numel()
                or goal.shape[0] != history.shape[0] or tactile_history.shape[0] != history.shape[0]):
            raise ValueError(
                f"Expected goal [B,{self.goal_norm.mean.numel()}], got {tuple(goal.shape)}"
            )
        goal_latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        proprio = self.obs_norm(history).flatten(-2)
        tactile = self.tactile_encoder(tactile_history, tactile_valid)
        return run_mlp(self.dec, torch.cat((goal_latent, proprio, tactile), dim=-1))


class FrozenOpenDoorExpert(nn.Module):
    """Adapter exposing a full OpenDoor expert as a DAgger labeler."""

    def __init__(self, expert: nn.Module, *, history_length: int = HISTORY_LENGTH):
        super().__init__()
        self.expert = expert.eval()
        self.history_length = history_length
        for parameter in self.expert.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, policy_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        history = _as_history(
            policy_history,
            frame_dim=OBS_DIM,
            history_length=self.history_length,
        )
        return self.expert(history, goal)

    def load_checkpoint(self, checkpoint: dict[str, Any], *, strict: bool = True) -> None:
        """Load the existing SuperMimic checkpoint's policy, including norms."""

        if "pi" not in checkpoint:
            raise ValueError("OpenDoor checkpoint must contain the 'pi' state dict")
        self.expert.load_state_dict(checkpoint["pi"], strict=strict)
        self.expert.eval()


def load_frozen_open_door_expert(path: str | Path, *, device: torch.device | str = "cpu") -> FrozenOpenDoorExpert:
    """Construct the current OpenDoor policy and load a full checkpoint."""

    from ...tasks.open_door.models import Policy

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy_config = checkpoint.get("policy_config", {})
    history_length = policy_config.get("history_length", HISTORY_LENGTH)
    goal_frames = policy_config.get("goal_frames", HISTORY_LENGTH)
    if history_length != HISTORY_LENGTH or goal_frames != HISTORY_LENGTH:
        raise ValueError(
            "The first student contract requires the current OpenDoor history_length=10 and goal_frames=10"
        )
    expert = Policy(history_length=history_length, goal_frames=goal_frames).to(device)
    wrapper = FrozenOpenDoorExpert(expert)
    wrapper.load_checkpoint(checkpoint)
    return wrapper
