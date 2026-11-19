"""Configuration for sitting proprioception/depth BC and DAgger."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SittingDaggerConfig:
    history_length: int = 10
    proprio_dim: int = 93
    expert_observation_dim: int = 102
    goal_dim: int = 730
    action_dim: int = 29
    expert_checkpoint: str = "logs/cuhk_sitting_deployment/model_10000.pt"
    split: str = "logs/cuhk_sitting_deployment/split.json"
    data_root: str = "data/motions/sitting"
    collision_cache_dir: str = "data/collisions/chairs"
    collision_profile: str = "default"
    output_dir: str = "logs/sitting_dagger_depth"
    num_envs: int = 83
    use_depth: bool = True
    initialize_goal_from_teacher: bool = True
    device: str = "cuda:0"
    seed: int = 0
    rounds: int = 16
    bc_rollout_steps: int = 128
    bc_train_epochs: int = 4
    bc_batches: int = 64
    rollout_steps: int = 64
    updates_per_round: int = 64
    save_every: int = 1
    capacity: int = 16_384
    max_dataset_gib: float = 2.0
    batch_size: int = 64
    train_epochs: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_decay_rounds: int = 8

    def __post_init__(self):
        if (self.history_length, self.proprio_dim, self.expert_observation_dim,
                self.goal_dim, self.action_dim) != (10, 93, 102, 730, 29):
            raise ValueError("Sitting distillation requires history10/proprio93/expert102/goal730/action29")
        if not isinstance(self.initialize_goal_from_teacher, bool):
            raise ValueError("initialize_goal_from_teacher must be boolean")
        if not isinstance(self.use_depth, bool):
            raise ValueError("use_depth must be boolean")
        for name in ("num_envs", "rounds", "bc_rollout_steps", "bc_train_epochs", "bc_batches",
                     "rollout_steps", "updates_per_round", "save_every", "capacity", "batch_size",
                     "train_epochs", "beta_decay_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not all(math.isfinite(value) for value in
                    (self.learning_rate, self.weight_decay, self.max_grad_norm, self.max_dataset_gib))
                or self.learning_rate <= 0 or self.weight_decay < 0
                or self.max_grad_norm <= 0 or self.max_dataset_gib <= 0):
            raise ValueError("Optimizer settings and dataset memory budget must be positive/nonnegative")
        if not 0 <= self.beta_start <= 1 or not 0 <= self.beta_end <= 1:
            raise ValueError("beta values must lie in [0, 1]")
        if self.dataset_bytes > self.max_dataset_gib * 1024**3:
            raise ValueError(f"Dataset requires {self.dataset_bytes / 1024**3:.2f} GiB, "
                             f"exceeding max_dataset_gib={self.max_dataset_gib}")

    @property
    def dataset_bytes(self) -> int:
        """CPU ring storage, excluding sampled batches and optimizer/model memory."""
        proprio_goal_action = 4 * (10 * 93 + 730 + 29) + 1
        return self.capacity * (proprio_goal_action + (2 * 120 * 160 if self.use_depth else 0))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SittingDaggerConfig":
        if not isinstance(values, dict):
            raise TypeError("DAgger config must be a JSON object")
        unknown = sorted(set(values) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown Sitting DAgger config fields: {', '.join(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "SittingDaggerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
