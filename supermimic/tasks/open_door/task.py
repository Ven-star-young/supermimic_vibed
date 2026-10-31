"""OpenDoor reference-tracking task assembly and checkpoint schema."""

import re
from dataclasses import asdict, replace

import torch

from motion_data import load_motion
from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ..base import TrackingTask
from ..g1_tracking.task import validate_env_cfg
from .dynamics import DoorDynamics
from .mdp import DoorMDP
from .models import Critic, Policy, WorldModel
from .motion import DoorMotionLib
from .geometry import DoorGeometry
from .assets import select_motion_files


class OpenDoorTask(TrackingTask):
    task_name = "g1_open_door"
    geometry_cls = DoorGeometry
    mdp_cls = DoorMDP

    @staticmethod
    def select_motion_files(command):
        files = select_motion_files(command)
        for path in files:
            region = load_motion(path).get("door_contact_region")
            # Legacy back references predate the explicit contact-region label.
            if region is not None and (region.shape != () or region.item() != "back"):
                raise ValueError(f"{path}: back-opening task cannot load a different door_contact_region")
        return files

    def make_policy(self, history_length, goal_frames, geometry_bank=None):
        return Policy(history_length=history_length, goal_frames=goal_frames)

    def make_world_model(self, geometry_bank=None, geometry_encoder=None):
        return WorldModel(dynamics=self.dynamics)

    def make_critic(self, goal_frames):
        return Critic(goal_frames)

    def checkpoint_metadata(self):
        return {
            "name": self.name, "schema_version": 2,
            "dynamics_schema": "open-door-g1-hinge-v1",
            "dimensions": asdict(self.dimensions),
            "state": "g1_p3_quat_wxyz4_v3_w3_q29_qd29,hinge_theta_omega",
            "context": "robot33,hinge_xyz3,door_dims3,door_physics7",
            "actor_observation": "robot93,door_hinge_relative3,door_normal3,sin_cos_theta,omega",
            "actor_goal": "robot73+door_theta+0.1*door_omega=75/frame",
            "geometry": None,
            "door_asset": {key: value.tolist() for key, value in self.motion.door_metadata.items()},
            "door_physics": {key: getattr(self.env_cfg.commands["motion"], key) for key in (
                "door_mass", "door_damping", "door_friction", "robot_door_friction",
                "door_stiffness", "door_spring_reference")},
            "door_randomization": asdict(self.env_cfg.commands["motion"].door_randomization_cfg),
            "model": {"actor_hidden": [2048, 1024, 512, 256], "goal_latent": 32,
                      "world_model_hidden": [2048, 2048, 2048], "critic_hidden": [256, 128]},
            "control_dt": self.mdp.control_dt,
            "door_reward": asdict(self.mdp.door_reward_cfg),
            "robot_reward": asdict(self.mdp.cfg),
            "observation": asdict(self.mdp.observation_cfg),
            "termination": asdict(self.mdp.termination_cfg),
        }

    def validate_checkpoint(self, checkpoint):
        from .events import DoorRandomizationCfg

        metadata = checkpoint.get("task")
        if metadata and "door_randomization" in metadata:
            # JSON stores tuples as lists; preserve strict value comparison while
            # accepting either representation of a validated sampling range.
            try:
                values = metadata["door_randomization"]
                if set(values) != set(asdict(self.env_cfg.commands["motion"].door_randomization_cfg)):
                    raise ValueError("Door randomization fields do not match the task schema")
                randomization = DoorRandomizationCfg(**values)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid OpenDoor checkpoint door_randomization") from exc
            checkpoint = {**checkpoint, "task": {
                **metadata, "door_randomization": asdict(randomization),
            }}
        super().validate_checkpoint(checkpoint)


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None,
               task_cls=OpenDoorTask):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    validate_env_cfg(cfg)
    command = cfg.commands["motion"]
    kin = G1Kinematics(device) if kin is None else kin
    if motion is None:
        selected = replace(command, motion_dir=motion_dir) if motion_dir is not None else command
        motion = DoorMotionLib(task_cls.select_motion_files(selected), kin, device)
    joints = cfg.scene.entities["robot"].init_state.joint_pos
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in joints.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    metadata = motion.door_metadata
    dynamics = DoorDynamics(float(metadata["base_yaw"]), float(metadata["joint_axis"][2]))
    geometry = task_cls.geometry_cls(kin, metadata, device)
    mdp = task_cls.mdp_cls(
        kin, motion, torch.tensor(defaults, device=device), cfg=command.reward_cfg,
        observation_cfg=command.observation_cfg, termination_cfg=command.termination_cfg,
        control_dt=cfg.sim.mujoco.timestep * cfg.decimation,
        door_reward_cfg=command.door_reward_cfg,
        dynamics=dynamics, geometry=geometry,
    )
    return task_cls(task_cls.task_name, motion, mdp, dynamics, cfg,
                    modeled_entities=("robot", "door"))
