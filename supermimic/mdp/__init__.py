"""Analytic MDP heads used by both physical and imagined transitions.

Legacy exports are retained while individual terms live in focused modules.
"""

from .core import MDP
from .observations import CRITIC_EXTRA, OBS_DIM, OBS_NOISE, OBS_SCALE, ObservationCfg
from .rewards import MAX_STEP_REWARD, RewardCfg
from .state import (
    INV_DIM, NUM_J, STATE_DIM, XI_DIM, G1Dynamics, flatten_state, invariant_features,
    normalize_xi, unflatten_state,
)
from .terminations import TerminationCfg

__all__ = [
    "MDP", "ObservationCfg", "RewardCfg", "TerminationCfg", "G1Dynamics",
    "STATE_DIM", "OBS_DIM", "XI_DIM", "NUM_J", "INV_DIM", "CRITIC_EXTRA",
    "OBS_NOISE", "OBS_SCALE", "MAX_STEP_REWARD", "flatten_state", "unflatten_state",
    "invariant_features", "normalize_xi",
]
