"""Training defaults (paper Table A.5) and command-line configuration.

Task packages may construct a modified ``TrainConfig`` without importing the runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    task: str = "g1_tracking"
    num_envs: int = 4096
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_every: int = 50
    motion_dir: str = str(REPO / "data" / "lafan1_g1")
    log_dir: str = str(REPO / "logs" / "run")
    seed: int = 0
    resume: str | None = None
    no_compile: bool = False
    amp: str = "bf16"
    device: str = "cuda:0"


def build_train_parser(defaults: TrainConfig | None = None) -> argparse.ArgumentParser:
    defaults = defaults or TrainConfig()
    parser = argparse.ArgumentParser(description="Train SuperMimic through a registered task.")
    help_text = {
        "utd": "updates per vectorized simulator step",
        "warmup_steps": "sim steps before any update",
        "wm_only_iters": "world-model-only iterations at the start",
        "no_compile": "disable torch.compile for debugging and small smoke runs",
        "amp": "autocast dtype for MLP matmuls",
    }
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        options = {"default": value, "help": help_text.get(field.name)}
        if isinstance(value, bool):
            options["action"] = "store_true"
        else:
            options["type"] = type(value) if value is not None else str
        if field.name == "amp":
            options["choices"] = ["none", "bf16"]
        parser.add_argument("--" + field.name, **options)
    return parser
