"""HDMI-style factories for the G1 motion-tracking task.

Scene/events use native mjlab configuration. Observation/reward/termination
configuration belongs to pure tensor MDP terms so the same heads run in BPTT.
"""
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from ...assets.g1.g1_constants import G1_JOINT_NAMES, action_scale, make_g1_robot_cfg
from ...envs.config import TrackingEnvCfg
from ...mdp import ObservationCfg, RewardCfg, TerminationCfg
from ...mdp.actions import JointPositionActionCfg
from ...mdp.commands import MotionCommandCfg
from ...mdp.events import (
    PhysicalContextCfg, ReferenceResetCfg, randomize_joint_offsets,
    randomize_physics, reset_robot_to_reference,
)


def make_scene(play=False):
    return SceneCfg(num_envs=32 if play else 4096, env_spacing=0.0,
                    terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": make_g1_robot_cfg()})


def make_sim():
    return SimulationCfg(njmax=500, nconmax=128,
                         mujoco=MujocoCfg(timestep=0.005, integrator="implicitfast", cone="pyramidal"))


def make_actions():
    return JointPositionActionCfg(entity_name="robot", joint_names=tuple(G1_JOINT_NAMES),
                                  scale=tuple(action_scale(name) for name in G1_JOINT_NAMES))


def make_commands(play=False):
    return MotionCommandCfg(sampling_mode="start" if play else "adaptive")


def make_observations():
    return ObservationCfg()


def make_rewards():
    return RewardCfg()


def make_terminations():
    return TerminationCfg()


def make_events():
    return {
        "physical_context": EventTermCfg(func=randomize_physics, mode="startup", params={"cfg": PhysicalContextCfg()}),
        "scene_reset": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        "joint_offsets": EventTermCfg(func=randomize_joint_offsets, mode="reset", params={"cfg": PhysicalContextCfg()}),
        "reference_state": EventTermCfg(func=reset_robot_to_reference, mode="reset", params={"cfg": ReferenceResetCfg()}),
    }


def make_env_cfg(play=False):
    return TrackingEnvCfg(scene=make_scene(play), sim=make_sim(), actions=make_actions(),
                          commands=make_commands(play), events=make_events(), observations=make_observations(),
                          rewards=make_rewards(), terminations=make_terminations(), reset_noise=not play)
