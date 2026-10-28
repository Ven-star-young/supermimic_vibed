"""Own the Sitting state, models, reference semantics and checkpoint contract."""

import re
from dataclasses import asdict, replace

import torch

from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ..base import TrackingTask
from ..g1_tracking.task import validate_env_cfg
from .dynamics import SittingDynamics
from .mdp import SittingMDP
from .models import SittingCritic, SittingPolicy, SittingWorldModel
from .motion import SittingMotionLib


class SittingTask(TrackingTask):
    def make_policy(self, history_length, goal_frames, geometry_bank):
        return SittingPolicy(geometry_bank, history_length=history_length, goal_frames=goal_frames)

    def make_world_model(self, geometry_bank, geometry_encoder):
        return SittingWorldModel(geometry_bank, geometry_encoder=geometry_encoder)

    def make_critic(self, goal_frames):
        return SittingCritic(goal_frames)

    @staticmethod
    def geometry_ids(context):
        return context[..., 45].long()

    def critic_goal_dim(self, frames):
        return self.goal_dim(frames) + self.motion.object_goal_feature_dim * frames

    def critic_goal(self, S, f, frames):
        """Extra critic reference; robot reference is resolved separately from f."""
        return self.motion.object_goal(S, f, frames)

    def critic_input(self, state, frame, normalized_context, goal, goal_frames, geometry=None):
        if geometry is None:
            raise ValueError("Sitting critic requires the shared geometry encoder output")
        features = self.mdp.critic_features(state, frame, normalized_context, geometry=geometry)
        return torch.cat((features, goal, self.critic_goal(state, frame, goal_frames)), -1)

    def checkpoint_metadata(self):
        return {
            "name": self.name, "schema_version": 4,
            "dynamics_schema": "sitting-g1-chair-v1",
            "dimensions": asdict(self.dimensions), "critic_object_goal_per_frame": 9,
            "state": "g1_p3_quat_wxyz4_v3_w3_q29_qd29,chair_p3_quat_wxyz4_v3_w3",
            "velocity_origin": "body_origin_world",
            "context": "robot33,mass,com3,inertia_xx_yy_zz_xy_xz_yz,mu_robot,mu_ground,geometry_id",
            "actor_observation": "robot93,chair_position3_rot6d6_in_actual_robot_heading",
            "actor_goal": "robot73:dp_rot6d_in_current_reference_heading,v_w_in_future_body,q,0.1*qd",
            "critic_object_goal": "position3_rot6d6_in_current_actual_robot_heading",
            "chair_delta": "position_velocity_in_actual_robot_heading;rotation_in_chair_body",
            "geometry": {"bank_version": 3, "points": 1024, "channels": 6, "latent": 64,
                         "frame": "chair_body_meters", "optimizer": "world_model_only"},
            "model": {"actor_hidden": [2048, 1024, 512, 256], "goal_latent": 32,
                      "world_model_hidden": [2048, 2048, 2048], "critic_hidden": [256, 128]},
            "control_dt": self.mdp.control_dt,
            "object_reward": asdict(self.mdp.object_reward_cfg),
            "robot_reward": asdict(self.mdp.cfg),
            "observation": asdict(self.mdp.observation_cfg),
            "termination": asdict(self.mdp.termination_cfg),
        }


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None, files=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    validate_env_cfg(cfg)
    command = cfg.commands["motion"]
    kin = G1Kinematics(device) if kin is None else kin
    if motion is None:
        if files is None:
            from .assets import select_motion_files
            files = select_motion_files(command if motion_dir is None else replace(command, motion_dir=motion_dir))
        motion = SittingMotionLib(files, kin, device)
    joint_positions = cfg.scene.entities["robot"].init_state.joint_pos
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in joint_positions.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    mdp = SittingMDP(
        kin, motion, torch.tensor(defaults, device=device), cfg=command.reward_cfg,
        observation_cfg=command.observation_cfg, termination_cfg=command.termination_cfg,
        control_dt=cfg.sim.mujoco.timestep * cfg.decimation,
        object_reward_cfg=getattr(command, "object_reward_cfg", None),
    )
    return SittingTask(
        "g1_sitting", motion, mdp, SittingDynamics(), cfg, modeled_entities=("robot", "chair"),
    )
