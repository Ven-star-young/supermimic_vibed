"""Sitting BC/DAgger with ten proprioceptive frames and one depth image."""

from .config import SittingDaggerConfig
from .dataset import DaggerBatch, SittingDaggerDataset
from .dagger import DaggerCoordinator, DaggerStep, mix_actions
from .models import (
    ACTION_DIM,
    DEPTH_HEIGHT,
    DEPTH_LATENT_DIM,
    DEPTH_WIDTH,
    GOAL_DIM,
    HISTORY_LENGTH,
    OBS_DIM,
    PROPRIO_DIM,
    STUDENT_INPUT_DIM,
    DepthEncoder,
    FrozenSittingExpert,
    SittingStudentPolicy,
    load_frozen_sitting_expert,
    load_sitting_student,
    split_sitting_history,
)

__all__ = [
    "ACTION_DIM", "DEPTH_HEIGHT", "DEPTH_LATENT_DIM", "DEPTH_WIDTH", "GOAL_DIM",
    "HISTORY_LENGTH", "OBS_DIM", "PROPRIO_DIM", "STUDENT_INPUT_DIM", "DaggerBatch",
    "DaggerCoordinator", "DaggerStep", "DepthEncoder", "FrozenSittingExpert",
    "SittingDaggerConfig", "SittingDaggerDataset", "SittingStudentPolicy",
    "load_frozen_sitting_expert", "load_sitting_student", "mix_actions", "split_sitting_history",
]
