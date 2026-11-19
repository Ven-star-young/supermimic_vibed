"""Sitting student with proprioception, robot motion goals, and one depth image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...rl.models import run_mlp
from ...tasks.g1_tracking.models import Policy as _BasePolicy


HISTORY_LENGTH = 10
PROPRIO_DIM = 93
OBS_DIM = 102
GOAL_DIM = 73 * HISTORY_LENGTH
ACTION_DIM = 29
DEPTH_HEIGHT = 120
DEPTH_WIDTH = 160
DEPTH_LATENT_DIM = 64
STUDENT_PROPRIO_INPUT_DIM = PROPRIO_DIM * HISTORY_LENGTH + 32
STUDENT_INPUT_DIM = STUDENT_PROPRIO_INPUT_DIM + DEPTH_LATENT_DIM


def _as_history(value: torch.Tensor, *, frame_dim: int) -> torch.Tensor:
    if value.ndim == 2 and value.shape[-1] == HISTORY_LENGTH * frame_dim:
        return value.reshape(value.shape[0], HISTORY_LENGTH, frame_dim)
    if value.ndim == 3 and value.shape[-2:] == (HISTORY_LENGTH, frame_dim):
        return value
    raise ValueError(f"Expected history [B,10,{frame_dim}] or [B,{10 * frame_dim}], got {tuple(value.shape)}")


def split_sitting_history(policy_history: torch.Tensor) -> torch.Tensor:
    """Keep the robot's 93 observable channels; drop the nine chair-state channels."""
    return _as_history(policy_history, frame_dim=OBS_DIM)[..., :PROPRIO_DIM]


class DepthEncoder(nn.Sequential):
    """HDMI convolution sizes for a metric [B,1,120,160] depth image."""

    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2), nn.Mish(),
            nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2), nn.Mish(),
            nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2), nn.Mish(),
            nn.Flatten(), nn.Linear(8 * 15 * 20, DEPTH_LATENT_DIM), nn.LayerNorm(DEPTH_LATENT_DIM),
        )


class SittingStudentPolicy(_BasePolicy):
    """930 proprioception + 32 goal, optionally augmented with 64 depth features.

    Inputs are ten oldest-to-newest proprioceptive frames, the existing 730-D
    robot goal, and the current float32 depth image clipped to 0.1--4 metres.
    Image values bypass running normalization; the CNN is trained with the actor.
    """

    def __init__(self, *, use_depth: bool = True):
        if not isinstance(use_depth, bool):
            raise ValueError("use_depth must be boolean")
        super().__init__(obs_dim=PROPRIO_DIM, goal_dim=GOAL_DIM,
                         hist=HISTORY_LENGTH, action_dim=ACTION_DIM)
        self.use_depth = use_depth
        self.goal_initialized_from_teacher = False
        self.depth_encoder = DepthEncoder() if use_depth else None
        if use_depth:
            self.dec[0] = nn.Linear(STUDENT_INPUT_DIM, self.dec[0].out_features)

    @torch.no_grad()
    def initialize_goal_from_expert(self, expert: nn.Module) -> None:
        for name in ("enc", "goal_norm"):
            getattr(self, name).load_state_dict(getattr(expert, name).state_dict())
        self.goal_initialized_from_teacher = True

    def forward(self, proprio_history: torch.Tensor, goal: torch.Tensor,
                depth: torch.Tensor | None = None):
        history = _as_history(proprio_history, frame_dim=PROPRIO_DIM)
        batch = history.shape[0]
        if goal.shape != (batch, GOAL_DIM):
            raise ValueError(f"Expected goal [B,{GOAL_DIM}], got {tuple(goal.shape)}")
        goal_latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        proprio = self.obs_norm(history).flatten(-2)
        features = [goal_latent, proprio]
        if self.use_depth:
            if (depth is None or depth.shape != (batch, 1, DEPTH_HEIGHT, DEPTH_WIDTH)
                    or depth.dtype != torch.float32):
                raise ValueError("Depth students require float32 metric depth [B,1,120,160]")
            features.append(self.depth_encoder(depth))
        elif depth is not None:
            raise ValueError("Proprioception students do not accept depth input")
        return run_mlp(self.dec, torch.cat(features, dim=-1))


class FrozenSittingExpert(nn.Module):
    """Frozen existing sitting actor; geometry IDs are only passed to the teacher."""

    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert.eval()
        for parameter in self.expert.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, policy_history: torch.Tensor, goal: torch.Tensor,
                geometry_ids: torch.Tensor) -> torch.Tensor:
        history = _as_history(policy_history, frame_dim=OBS_DIM)
        if geometry_ids.shape != (history.shape[0],):
            raise ValueError("Expected teacher geometry IDs [B]")
        return self.expert(history, goal, geometry_ids)

    def load_checkpoint(self, checkpoint: dict[str, Any], *, strict: bool = True) -> None:
        if "pi" not in checkpoint:
            raise ValueError("Sitting checkpoint must contain the 'pi' state dict")
        policy_config = checkpoint.get("policy_config", {})
        if any(policy_config.get(name, 10) != 10 for name in ("history_length", "goal_frames")):
            raise ValueError("Sitting distillation requires history_length=10 and goal_frames=10")
        self.expert.load_state_dict(checkpoint["pi"], strict=strict)
        self.expert.eval()


def load_frozen_sitting_expert(path: str | Path, geometry_bank,
                              *, device: torch.device | str = "cpu") -> FrozenSittingExpert:
    """Use the environment's chair geometry bank and the checkpoint's learned actor."""
    from ...tasks.sitting.models import SittingPolicy

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    wrapper = FrozenSittingExpert(SittingPolicy(geometry_bank).to(device))
    wrapper.load_checkpoint(checkpoint)
    return wrapper


def load_sitting_student(checkpoint: dict[str, Any],
                         *, device: torch.device | str = "cpu") -> SittingStudentPolicy:
    """Validate the inference contract, then load a frozen BC/DAgger student."""
    from ...camera import HDMI_CAMERA_METADATA

    kind = checkpoint.get("kind")
    if kind not in ("sitting_depth_student", "sitting_proprio_student") or checkpoint.get("schema_version") != 1:
        raise ValueError("Expected a sitting_depth_student or sitting_proprio_student checkpoint with schema_version=1")
    use_depth = kind == "sitting_depth_student"
    if use_depth and checkpoint.get("camera_metadata") != HDMI_CAMERA_METADATA:
        raise ValueError("Sitting student camera metadata differs from the HDMI depth configuration")
    if not use_depth and checkpoint.get("camera_metadata") is not None:
        raise ValueError("Proprioception student must not contain camera metadata")
    policy_config = checkpoint.get("policy_config", {})
    declared_use_depth = policy_config.get("use_depth", use_depth)
    if declared_use_depth != use_depth:
        raise ValueError("Sitting student use_depth metadata does not match checkpoint kind")
    expected = {"history_length": HISTORY_LENGTH, "goal_frames": HISTORY_LENGTH,
                "proprio_dim": PROPRIO_DIM, "goal_dim": GOAL_DIM,
                "action_dim": ACTION_DIM, "history_order": "oldest_to_newest"}
    if use_depth:
        expected.update(depth_shape=[1, DEPTH_HEIGHT, DEPTH_WIDTH], depth_latent_dim=DEPTH_LATENT_DIM,
                        depth_units="metres")
    for name, value in expected.items():
        if policy_config.get(name) != value:
            raise ValueError(f"Sitting student requires policy_config.{name}={value!r}")
    if "student" not in checkpoint:
        raise ValueError("Sitting student checkpoint lacks the 'student' state dict")
    policy = SittingStudentPolicy(use_depth=use_depth).to(device)
    policy.load_state_dict(checkpoint["student"], strict=True)
    policy.goal_initialized_from_teacher = checkpoint.get("goal_encoder", {}).get("initialized_from_teacher", False)
    return policy.eval().requires_grad_(False)
