"""OpenDoor student with proprioception, motion goals, and channel tactile history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...rl.models import (
    TACTILE_CHANNELS,
    TACTILE_LATENT_DIM,
    TACTILE_REGION_SIZES,
    TactileEncoder,
    run_mlp,
)
from ...tasks.g1_tracking.models import Policy as _BasePolicy
from ..sitting.models import DEPTH_HEIGHT, DEPTH_WIDTH, DEPTH_LATENT_DIM, DepthEncoder


HISTORY_LENGTH = 10
PROPRIO_DIM = 93
DOOR_DIM = 9
OBS_DIM = PROPRIO_DIM + DOOR_DIM
GOAL_DIM = 75 * HISTORY_LENGTH
ACTION_DIM = 29
STUDENT_INPUT_DIM = PROPRIO_DIM * HISTORY_LENGTH


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


class OpenDoorStudentPolicy(_BasePolicy):
    """OpenDoor student with optional tactile and depth latents instead of direct door state.

    Input:
        ``proprio_history``: ``[B, 10, 93]`` (or flattened ``[B, 930]``)
        ``goal``: ``[B, 750]``
        ``tactile_history``: raw counts ``[B, 10, 112]``.
        ``tactile_valid``: preprocessing mask; a network feature only for legacy checkpoints.
        ``depth``: current float32 metric depth ``[B,1,120,160]``, as in Sitting.
    Output:
        normalized 29-D joint action consumed by the existing action manager.
    """

    def __init__(self, history_length: int = HISTORY_LENGTH, goal_frames: int = HISTORY_LENGTH,
                 *, use_tactile: bool = True, tactile_include_valid: bool = True,
                 use_depth: bool = False):
        if history_length != HISTORY_LENGTH:
            raise ValueError("The first OpenDoor student contract requires history_length=10")
        if not isinstance(use_tactile, bool):
            raise ValueError("use_tactile must be boolean")
        if not isinstance(use_depth, bool):
            raise ValueError("use_depth must be boolean")
        super().__init__(
            obs_dim=PROPRIO_DIM,
            goal_dim=75 * goal_frames,
            hist=history_length,
            action_dim=ACTION_DIM,
        )
        self.use_tactile = use_tactile
        self.use_depth = use_depth
        self.goal_initialized_from_teacher = False
        self.tactile_encoder = (TactileEncoder(history_length, include_valid=tactile_include_valid)
                                if use_tactile else None)
        self.depth_encoder = DepthEncoder() if use_depth else None
        extra = (TACTILE_LATENT_DIM if use_tactile else 0) + (DEPTH_LATENT_DIM if use_depth else 0)
        if extra:
            self.dec[0] = nn.Linear(self.dec[0].in_features + extra, self.dec[0].out_features)

    @torch.no_grad()
    def initialize_goal_from_expert(self, expert: nn.Module) -> None:
        """Copy compatible goal normalization/encoder weights from an expert."""

        for name in ("enc", "goal_norm"):
            source, target = getattr(expert, name, None), getattr(self, name, None)
            if source is None or target is None:
                raise TypeError(f"Expert lacks compatible {name} module")
            target.load_state_dict(source.state_dict())
        self.goal_initialized_from_teacher = True

    def forward(self, proprio_history: torch.Tensor, goal: torch.Tensor,
                tactile_history: torch.Tensor | None = None, tactile_valid: torch.Tensor | None = None,
                *, depth: torch.Tensor | None = None):
        history = _as_history(
            proprio_history,
            frame_dim=PROPRIO_DIM,
            history_length=self.hist,
        )
        if (goal.ndim != 2 or goal.shape[-1] != self.goal_norm.mean.numel()
                or goal.shape[0] != history.shape[0]):
            raise ValueError(
                f"Expected goal [B,{self.goal_norm.mean.numel()}], got {tuple(goal.shape)}"
            )
        goal_latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        proprio = self.obs_norm(history).flatten(-2)
        features = [goal_latent, proprio]
        if self.use_tactile:
            if tactile_history is None or tactile_history.shape[0] != history.shape[0]:
                raise ValueError("Tactile students require counts for every observation")
            features.append(self.tactile_encoder(tactile_history, tactile_valid))
        if self.use_depth:
            if (depth is None or depth.shape != (history.shape[0], 1, DEPTH_HEIGHT, DEPTH_WIDTH)
                    or depth.dtype != torch.float32):
                raise ValueError("Depth students require float32 metric depth [B,1,120,160]")
            features.append(self.depth_encoder(depth))
        elif depth is not None:
            raise ValueError("Student configured without depth does not accept depth input")
        return run_mlp(self.dec, torch.cat(features, dim=-1))


class FrozenOpenDoorExpert(nn.Module):
    """Adapter exposing a full OpenDoor expert as a DAgger labeler."""

    def __init__(self, expert: nn.Module, *, history_length: int = HISTORY_LENGTH):
        super().__init__()
        self.expert = expert.eval()
        self.history_length = history_length
        self.observation_dim = OBS_DIM + (2 * TACTILE_CHANNELS if getattr(expert, "use_tactile", False) else 0)
        for parameter in self.expert.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, policy_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        history = _as_history(
            policy_history,
            frame_dim=self.observation_dim,
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
    expert = Policy(history_length=history_length, goal_frames=goal_frames,
                    use_tactile="tactile" in checkpoint.get("task", {})).to(device)
    wrapper = FrozenOpenDoorExpert(expert)
    wrapper.load_checkpoint(checkpoint)
    return wrapper


def load_open_door_student(checkpoint: dict[str, Any],
                           *, device: torch.device | str = "cpu") -> OpenDoorStudentPolicy:
    """Validate the tactile/depth input contract and load a frozen DAgger actor."""
    from ...camera import HDMI_CAMERA_METADATA

    saved = checkpoint.get("config", {})
    use_tactile = saved.get("use_tactile", True)
    use_depth = saved.get("use_depth", False)
    include_valid = saved.get("tactile_include_valid", True)
    if any(type(flag) is not bool for flag in (use_tactile, use_depth, include_valid)):
        raise ValueError("Student modality settings must be boolean")
    kind = ("open_door_tactile_depth_student" if use_tactile else "open_door_depth_student") if use_depth else (
        "open_door_tactile_student" if use_tactile else "open_door_proprio_student")
    counts_only = use_tactile and not include_valid
    expected = {"history_length": 10, "goal_frames": 10, "proprio_dim": 93,
                "tactile_channels": 112 if use_tactile else 0,
                "tactile_latent_dim": 64 if use_tactile else 0, "action_dim": 29,
                "history_order": "oldest_to_newest",
                "tactile_units": "uint8_counts" if use_tactile else None}
    if counts_only:
        expected["tactile_include_valid"] = False
    if use_depth:
        expected.update(use_depth=True, goal_dim=GOAL_DIM,
                        depth_shape=[1, DEPTH_HEIGHT, DEPTH_WIDTH], depth_latent_dim=DEPTH_LATENT_DIM,
                        depth_units="metres", tactile_include_valid=include_valid if use_tactile else False)
        if checkpoint.get("camera_metadata") != HDMI_CAMERA_METADATA:
            raise ValueError("OpenDoor student camera metadata differs from the HDMI depth configuration")
    elif checkpoint.get("camera_metadata") is not None:
        raise ValueError("Student without depth must not contain camera metadata")
    schema = 3 if use_depth else (2 if counts_only else 1)
    if (checkpoint.get("kind") != kind or checkpoint.get("schema_version") != schema
            or checkpoint.get("policy_config") != expected):
        raise ValueError("Unsupported OpenDoor student input schema or modality configuration")
    policy = OpenDoorStudentPolicy(use_tactile=use_tactile, use_depth=use_depth,
                                   tactile_include_valid=include_valid).to(device)
    policy.load_state_dict(checkpoint["student"], strict=True)
    policy.goal_initialized_from_teacher = checkpoint.get("goal_encoder", {}).get("initialized_from_teacher", False)
    return policy.eval().requires_grad_(False)
