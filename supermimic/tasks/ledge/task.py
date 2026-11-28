"""Ledge tensor heads, reference phases and strict checkpoint identity."""

from dataclasses import asdict, replace
import re

import numpy as np
import torch

from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ...motion import MotionLib
from ..base import TrackingTask
from ..g1_tracking.task import validate_env_cfg
from .assets import selected_asset, selected_paths
from .dynamics import LedgeDynamics
from .geometry import LedgeGeometry
from .mdp import LedgeMDP


class LedgeTask(TrackingTask):
    def checkpoint_metadata(self):
        command = self.env_cfg.commands["motion"]
        return {
            "name": self.name, "schema_version": 1,
            "dynamics_schema": "g1-fixed-ledge-v1",
            "dimensions": asdict(self.dimensions),
            "state": "g1_p3_quat_wxyz4_v3_w3_q29_qd29",
            "context": "robot_friction1_joint_offsets29_com_offsets3",
            "scene": {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in self.mdp.geometry_metadata.items()},
            "actor_observation": "robot93,scene_xyz3,yaw_sin_cos2,foot_margin2,foot_height2,scene_velocity3",
            "actor_goal": "robot73/frame",
            "contact_phase": "source-planned-back-linear-and-stance-nearest-on-control-clock-v1",
            "model": {"actor_hidden": [2048, 1024, 512, 256], "goal_latent": 32,
                      "world_model_hidden": [2048, 2048, 2048], "critic_hidden": [256, 128]},
            "control_dt": self.mdp.control_dt,
            "ledge_reward": asdict(self.mdp.ledge_reward_cfg),
            "robot_reward": asdict(self.mdp.cfg),
            "observation": asdict(self.mdp.observation_cfg),
            "robot_physics": asdict(command.physical_cfg),
            "external_force": False,
        }


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    if motion_dir is not None:
        cfg = replace(cfg, commands={**cfg.commands, "motion": replace(
            cfg.commands["motion"], motion_dir=str(motion_dir))})
    validate_env_cfg(cfg)
    command = cfg.commands["motion"]
    clip, metadata = selected_asset(command)
    path, _, _ = selected_paths(command)
    kin = G1Kinematics(device) if kin is None else kin
    motion = MotionLib([str(path)], kin, device) if motion is None else motion
    count = len(clip["root_pos"])
    source_time = np.arange(motion.num_frames, dtype=np.float64) * motion.control_dt * float(clip["fps"])
    expected = int(np.floor((count - 1) / float(clip["fps"]) / motion.control_dt + 1e-9)) + 1
    if len(motion.names) != 1 or motion.num_frames != expected:
        raise ValueError("Ledge motion and planned phases must share the same single reference clock")
    nearest = np.rint(source_time).astype(int).clip(0, count - 1)
    motion.planned_stance = torch.tensor(clip["planned_stance"][nearest], device=device)
    motion.back_required = torch.tensor(clip["back_required"][nearest], device=device)
    motion.back_weight = torch.tensor(np.interp(source_time, np.arange(count),
                                              clip["back_alignment_weight"]),
                                      dtype=torch.float32, device=device)
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in cfg.scene.entities["robot"].init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    geometry = LedgeGeometry(kin, metadata, device)
    mdp = LedgeMDP(kin, motion, torch.tensor(defaults, device=device), geometry=geometry,
                   ledge_reward_cfg=command.ledge_reward_cfg, cfg=command.reward_cfg,
                   observation_cfg=command.observation_cfg, termination_cfg=command.termination_cfg,
                   control_dt=cfg.sim.mujoco.timestep * cfg.decimation)
    mdp.geometry_metadata = metadata
    return LedgeTask("g1_ledge", motion, mdp, LedgeDynamics(), cfg)
