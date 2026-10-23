"""Composition of native mjlab scene/events and task-specific tensor MDP."""
from dataclasses import dataclass

from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg

from ..mdp.actions import JointPositionActionCfg
from ..mdp.commands import MotionCommandCfg
from ..mdp import ObservationCfg, RewardCfg, TerminationCfg


@dataclass(kw_only=True)
class TrackingEnvCfg:
    scene: SceneCfg
    sim: SimulationCfg
    actions: JointPositionActionCfg
    commands: MotionCommandCfg
    events: dict[str, EventTermCfg]
    observations: ObservationCfg
    rewards: RewardCfg
    terminations: TerminationCfg
    decimation: int = 4
    episode_length_s: float = 10.0
    reset_noise: bool = True
    context_dim: int = 33

    @property
    def step_dt(self):
        return self.sim.mujoco.timestep * self.decimation
