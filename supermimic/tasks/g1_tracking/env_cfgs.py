"""Native mjlab scene and manager configuration for G1 motion tracking."""
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from ...assets import DATA_DIR
from ...assets.g1.g1_constants import G1_JOINT_NAMES, action_scale, make_g1_robot_cfg
from ...mdp.actions import JointPositionActionCfg
from ...mdp.commands import MotionCommandCfg
from ...mdp.events import randomize_joint_offsets, randomize_physics
from ...mdp import native


def make_scene(play=False):
    return SceneCfg(num_envs=32 if play else 4096, env_spacing=0.0,
                    terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": make_g1_robot_cfg()})


def make_sim():
    return SimulationCfg(njmax=500, nconmax=128,
                         mujoco=MujocoCfg(timestep=0.005, integrator="implicitfast", cone="pyramidal"))


def make_actions():
    return {"joint_position": JointPositionActionCfg(
        entity_name="robot", joint_names=tuple(G1_JOINT_NAMES),
        scale=tuple(action_scale(name) for name in G1_JOINT_NAMES))}


def make_commands(play=False):
    return {"motion": MotionCommandCfg(
        motion_dir=os.environ.get("SUPERMIMIC_MOTION_DIR", str(DATA_DIR / "lafan1_g1")),
        play=play, reset_noise=not play,
                                        sampling_mode="start" if play else "adaptive")}


def make_observations():
    return {
        "policy_history": ObservationGroupCfg(terms={"history": ObservationTermCfg(func=native.PolicyHistory)}),
        "goal": ObservationGroupCfg(terms={"reference": ObservationTermCfg(func=native.goal)}),
        "state": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=native.state)}),
        "context": ObservationGroupCfg(terms={"context": ObservationTermCfg(func=native.context)}),
        "frame": ObservationGroupCfg(terms={"frame": ObservationTermCfg(func=native.frame)}),
        "step_count": ObservationGroupCfg(terms={"step_count": ObservationTermCfg(func=native.step_count)}),
        "previous_action": ObservationGroupCfg(terms={"action": ObservationTermCfg(func=native.previous_action)}),
    }


def make_rewards():
    # The analytic MDP head already applies the 0.02 s reward scaling. The same
    # function is differentiated in imagination; keep one source of truth.
    return {"tracking": RewardTermCfg(func=native.tracking_reward, weight=1.0)}


def make_terminations(play=False):
    terms = {
        "tracking_failure": TerminationTermCfg(func=native.tracking_failure),
        "motion_complete": TerminationTermCfg(func=native.motion_complete, time_out=True),
    }
    if not play:
        terms["time_out"] = TerminationTermCfg(func=time_out, time_out=True)
    return terms


def make_events():
    return {
        "physical_context": EventTermCfg(func=randomize_physics, mode="startup"),
        "scene_reset": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        "joint_offsets": EventTermCfg(func=randomize_joint_offsets, mode="reset"),
    }


def make_env_cfg(play=False):
    return ManagerBasedRlEnvCfg(
        scene=make_scene(play), sim=make_sim(), decimation=4, episode_length_s=10.0,
        seed=0, auto_reset=play, scale_rewards_by_dt=False,
        actions=make_actions(), commands=make_commands(play), events=make_events(),
        observations=make_observations(), rewards=make_rewards(), terminations=make_terminations(play),
        viewer=ViewerConfig(entity_name="robot", body_name="torso_link",
                            origin_type=ViewerConfig.OriginType.ASSET_BODY, distance=3.0),
    )
