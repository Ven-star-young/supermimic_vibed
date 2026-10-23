"""Compatibility exports for the native mjlab tracking environment."""
from .envs.tracking_env import TrackingSim
from .tasks.g1_tracking.env_cfgs import make_sim as make_sim_cfg

DECIMATION = 4
PHYSICS_DT = 0.005
__all__ = ["TrackingSim", "make_sim_cfg", "DECIMATION", "PHYSICS_DT"]
