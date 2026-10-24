"""Build G1 tensor heads when the native motion command initializes."""
import re

import torch

from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ...mdp import G1Dynamics, MDP
from ...motion import MotionLib
from ..base import TrackingTask


def validate_env_cfg(cfg):
    action = cfg.actions["joint_position"]
    if action.joint_names != tuple(G1_JOINT_NAMES):
        raise ValueError("G1 action joint names/order must match the reference and dynamics schema")
    if action.entity_name != "robot" or action.calibration_start != 1:
        raise ValueError("G1 requires entity 'robot' and context [friction, 29 joint offsets, 3 COM offsets]")
    physical = cfg.commands["motion"].physical_cfg
    if physical.entity_name != "robot" or physical.calibration_start != 1 or physical.com_start != 30:
        raise ValueError("G1 physical context layout must match [friction, 29 joint offsets, 3 COM offsets]")
    if "robot" not in cfg.scene.entities or cfg.scene.entities["robot"].init_state.joint_pos is None:
        raise ValueError("G1 task requires explicit default joint positions in its robot entity config")


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    validate_env_cfg(cfg)
    command = cfg.commands["motion"]
    kin = G1Kinematics(device) if kin is None else kin
    motion = MotionLib.from_dir(motion_dir or command.motion_dir, kin, device) if motion is None else motion
    joint_positions = cfg.scene.entities["robot"].init_state.joint_pos
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in joint_positions.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    mdp = MDP(kin, motion, torch.tensor(defaults, device=device), cfg=command.reward_cfg,
              observation_cfg=command.observation_cfg, termination_cfg=command.termination_cfg,
              control_dt=cfg.sim.mujoco.timestep * cfg.decimation)
    return TrackingTask("g1_tracking", motion, mdp, G1Dynamics(), cfg)
