"""Bind task-specific objectives to the common robot/object dynamics contract."""

import re
from dataclasses import asdict, replace

import torch

from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ...rl.shared_world_model import SharedDynamics
from ..base import TrackingTask
from ..g1_tracking.task import validate_env_cfg
from .mdp import InteractionMDP
from .motion import InteractionMotionLib


class InteractionTask(TrackingTask):
    def critic_goal_dim(self, frames):
        return self.goal_dim(frames) + self.motion.object_goal_feature_dim * frames

    def critic_goal(self, S, f, frames):
        """Extra critic reference; robot reference is resolved separately from f."""
        return self.motion.object_goal(S, f, frames)

    def checkpoint_metadata(self):
        return {
            "name": self.name, "schema_version": 3,
            "dynamics_schema": "g1-rigid-object-v1",
            "dimensions": asdict(self.dimensions), "critic_object_goal_per_frame": 6,
        }


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None, files=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    validate_env_cfg(cfg)
    command = cfg.commands["motion"]
    enabled = command.object_enabled
    kin = G1Kinematics(device) if kin is None else kin
    if motion is None:
        if files is None:
            from .assets import select_motion_files
            files = select_motion_files(command if motion_dir is None else replace(command, motion_dir=motion_dir))
        motion = InteractionMotionLib(files, kin, device, object_enabled=enabled)
    if motion.object_enabled != enabled:
        raise ValueError("Motion object references and scene object_enabled must agree")
    joint_positions = cfg.scene.entities["robot"].init_state.joint_pos
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in joint_positions.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    mdp = InteractionMDP(
        kin, motion, torch.tensor(defaults, device=device), cfg=command.reward_cfg,
        observation_cfg=command.observation_cfg, termination_cfg=command.termination_cfg,
        control_dt=cfg.sim.mujoco.timestep * cfg.decimation, object_enabled=enabled,
        object_reward_cfg=getattr(command, "object_reward_cfg", None),
    )
    return InteractionTask(
        "g1_sitting" if enabled else "g1_shared_tracking", motion, mdp, SharedDynamics(), cfg,
        modeled_entities=("robot", "chair") if enabled else ("robot",),
    )
