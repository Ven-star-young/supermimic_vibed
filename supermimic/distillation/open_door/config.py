"""Configuration for OpenDoor proprioception/tactile BC and DAgger."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpenDoorDaggerConfig:
    """Small, explicit DAgger settings.

    The environment rollout is intentionally supplied by the caller.  This
    keeps the data-collection code independent of mjlab while enforcing the
    action and observation contract used by the OpenDoor task.
    """

    history_length: int = 10
    proprio_dim: int = 93
    expert_observation_dim: int = 102
    goal_dim: int = 750
    action_dim: int = 29
    expert_checkpoint: str = "logs/wm_transfer/back_model_10000.pt"
    output_dir: str = "logs/open_door_dagger_tactile"
    motion_dir: str = "data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz"
    num_envs: int = 16
    tactile_layout: str = "original_outer"
    tactile_contact_distance_m: float = 0.004
    tactile_counts_full_scale_depth_m: float = 0.010
    tactile_counts_exponent: float = 1.0
    device: str = "cuda"
    seed: int = 0
    rounds: int = 10
    bc_rollout_steps: int = 2000
    bc_train_epochs: int = 4
    bc_batches: int = 8
    rollout_steps: int = 2000
    updates_per_round: int = 4
    save_every: int = 1
    capacity: int = 200_000
    batch_size: int = 1024
    train_epochs: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_decay_rounds: int = 10

    def __post_init__(self):
        if (self.history_length, self.proprio_dim, self.expert_observation_dim,
                self.goal_dim, self.action_dim) != (10, 93, 102, 750, 29):
            raise ValueError("OpenDoor distillation requires history10/proprio93/expert102/goal750/action29")
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.tactile_layout not in ("original_outer", "inner_arms_v1"):
            raise ValueError("Unknown tactile layout")
        from ...tactile import G1TactileSensorCfg
        G1TactileSensorCfg(name="tactile", contact_distance_m=self.tactile_contact_distance_m,
            counts_full_scale_depth_m=self.tactile_counts_full_scale_depth_m,
            counts_exponent=self.tactile_counts_exponent)
        if (self.rounds < 1 or self.bc_rollout_steps < 1 or self.bc_train_epochs < 1
                or self.bc_batches < 1 or self.rollout_steps < 1
                or self.updates_per_round < 1 or self.save_every < 1):
            raise ValueError("BC/DAgger rounds, rollout steps, updates, and save_every must be positive")
        if self.capacity < 1 or self.batch_size < 1 or self.train_epochs < 1:
            raise ValueError("capacity, batch_size, and train_epochs must be positive")
        if (not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.max_grad_norm))
                or self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0):
            raise ValueError("optimizer settings must be positive/nonnegative")
        if not 0 <= self.beta_start <= 1 or not 0 <= self.beta_end <= 1:
            raise ValueError("beta values must lie in [0, 1]")
        if self.beta_decay_rounds < 1:
            raise ValueError("beta_decay_rounds must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OpenDoorDaggerConfig":
        """Build a config while rejecting misspelled training fields."""

        if not isinstance(values, dict):
            raise TypeError("DAgger config must be a JSON object")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown OpenDoor DAgger config fields: {', '.join(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "OpenDoorDaggerConfig":
        path = Path(path)
        with path.open() as stream:
            values = json.load(stream)
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
