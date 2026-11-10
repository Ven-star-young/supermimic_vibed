"""OpenDoor BC/DAgger with ten-frame proprioception and tactile observations."""

from .config import OpenDoorDaggerConfig
from .dataset import DaggerBatch, OpenDoorDaggerDataset
from .dagger import DaggerCoordinator, DaggerStep, mix_actions
from .models import (
    ACTION_DIM,
    GOAL_DIM,
    HISTORY_LENGTH,
    OBS_DIM,
    PROPRIO_DIM,
    STUDENT_INPUT_DIM,
    TACTILE_CHANNELS,
    TACTILE_LATENT_DIM,
    TACTILE_REGION_SIZES,
    FrozenOpenDoorExpert,
    OpenDoorStudentPolicy,
    TactileEncoder,
    TactileHistory,
    load_frozen_open_door_expert,
    split_open_door_history,
)

__all__ = [
    "ACTION_DIM",
    "GOAL_DIM",
    "HISTORY_LENGTH",
    "OBS_DIM",
    "PROPRIO_DIM",
    "STUDENT_INPUT_DIM",
    "TACTILE_CHANNELS",
    "TACTILE_LATENT_DIM",
    "TACTILE_REGION_SIZES",
    "DaggerBatch",
    "DaggerCoordinator",
    "DaggerStep",
    "FrozenOpenDoorExpert",
    "OpenDoorDaggerConfig",
    "OpenDoorDaggerDataset",
    "OpenDoorStudentPolicy",
    "TactileEncoder",
    "TactileHistory",
    "load_frozen_open_door_expert",
    "mix_actions",
    "split_open_door_history",
]
