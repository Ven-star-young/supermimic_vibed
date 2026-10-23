"""Assemble the concrete G1 reference-tracking task used by training and evaluation."""
import re
import torch

from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ...mdp import G1Dynamics, MDP
from ...motion import MotionLib
from ..base import TrackingTask
from .env_cfgs import make_env_cfg


def validate_env_cfg(cfg):
    if cfg.actions.joint_names != tuple(G1_JOINT_NAMES):
        raise ValueError("G1 action joint names/order must match the reference and dynamics schema")
    if cfg.actions.entity_name != "robot" or cfg.context_dim != 33 or cfg.actions.calibration_start != 1:
        raise ValueError("G1 requires entity 'robot' and context [friction, 29 joint offsets, 3 COM offsets]")
    if "robot" not in cfg.scene.entities or cfg.scene.entities["robot"].init_state.joint_pos is None:
        raise ValueError("G1 task requires explicit default joint positions in its robot entity config")


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None):
    cfg = make_env_cfg(play) if cfg is None else cfg
    validate_env_cfg(cfg)
    kin = G1Kinematics(device) if kin is None else kin
    motion = MotionLib.from_dir(motion_dir, kin, device) if motion is None else motion
    joint_positions = cfg.scene.entities["robot"].init_state.joint_pos
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in joint_positions.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    mdp = MDP(kin, motion, torch.tensor(defaults, device=device), cfg=cfg.rewards,
              observation_cfg=cfg.observations, termination_cfg=cfg.terminations, control_dt=cfg.step_dt)
    task = TrackingTask("g1_tracking", motion, mdp, G1Dynamics(), cfg, play=play, validate_env_cfg=validate_env_cfg)
    return task
