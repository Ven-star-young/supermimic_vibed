"""mjlab events for physical context and reference-state initialization.

Physical context is stored on the motion command and recorded into replay.
Reference initialization is invoked by that command after reset events, so the
sampled frame and the written robot state always agree.
"""
from dataclasses import dataclass

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from ..rotations import mat_to_quat, so3_exp


@dataclass(kw_only=True)
class PhysicalContextCfg:
    entity_name: str = "robot"
    torso_body: str = "torso_link"
    friction: tuple[float, float] = (0.3, 1.2)
    evaluation_friction: float = 1.0
    com_ranges: tuple[tuple[float, float], ...] = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
    joint_offset: tuple[float, float] = (-0.01, 0.01)
    calibration_start: int = 1
    com_start: int = 30


@dataclass(kw_only=True)
class ReferenceResetCfg:
    position: tuple[float, ...] = (0.05, 0.05, 0.01)
    rotation: tuple[float, ...] = (0.1, 0.1, 0.2)
    linear_velocity: tuple[float, ...] = (0.5, 0.5, 0.2)
    angular_velocity: tuple[float, ...] = (0.52, 0.52, 0.78)
    joint_position: float = 0.1


def default_model_value(sim, field, env_ids, item):
    """Select immutable model defaults from uniform or per-world variant scenes."""
    values = sim.get_default_field(field)
    if field in getattr(sim, "per_world_default_fields", ()):
        return values[env_ids, item]
    return values[item]


@requires_model_fields("geom_friction", "body_ipos", recompute=RecomputeLevel.set_const)
def randomize_physics(env, env_ids=None, command_name: str = "motion"):
    cmd = env.command_manager.get_term(command_name)
    cfg = cmd.cfg.physical_cfg
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[cfg.entity_name]
    if cmd.cfg.play:
        cmd.xi[ids, 0] = cfg.evaluation_friction
    else:
        cmd.xi[ids, 0] = torch.empty(len(ids), device=env.device).uniform_(*cfg.friction)
        for axis, limits in enumerate(cfg.com_ranges):
            cmd.xi[ids, cfg.com_start + axis] = torch.empty(len(ids), device=env.device).uniform_(*limits)
    geoms = robot.indexing.geom_ids
    env.sim.model.geom_friction[ids[:, None], geoms[None], 0] = cmd.xi[ids, :1]
    planes = torch.as_tensor(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE,
                             device=env.device).nonzero().flatten()
    env.sim.model.geom_friction[ids[:, None], planes[None], 0] = cfg.evaluation_friction if cmd.cfg.play else cfg.friction[0]
    torso = int(robot.indexing.body_ids[robot.body_names.index(cfg.torso_body)])
    env.sim.model.body_ipos[ids, torso] = default_model_value(env.sim, "body_ipos", ids, torso) + cmd.xi[ids, cfg.com_start:cfg.com_start + 3]


def randomize_joint_offsets(env, env_ids, command_name: str = "motion"):
    cmd = env.command_manager.get_term(command_name)
    cfg = cmd.cfg.physical_cfg
    begin, count = cfg.calibration_start, cmd.dynamics.action_dim
    cmd.xi[env_ids, begin:begin + count] = (
        0.0 if cmd.cfg.play else torch.empty(len(env_ids), count, device=env.device).uniform_(*cfg.joint_offset)
    )


def reset_robot_to_reference(env, env_ids, cfg: ReferenceResetCfg, *, command=None):
    cmd = env.command_manager.get_term("motion") if command is None else command
    frames = cmd.time_steps[env_ids]
    p, R, v, w, q, qd = (value.clone() for value in cmd.motion.ref_state(frames))
    if cmd.cfg.reset_noise and not cmd.cfg.play:
        def perturb(value, bounds):
            return value + (2 * torch.rand_like(value) - 1) * torch.tensor(bounds, device=env.device)
        p = perturb(p, cfg.position)
        rotation = perturb(torch.zeros(len(env_ids), 3, device=env.device), cfg.rotation)
        R = so3_exp(rotation) @ R
        v, w = perturb(v, cfg.linear_velocity), perturb(w, cfg.angular_velocity)
        q = q + torch.empty_like(q).uniform_(-cfg.joint_position, cfg.joint_position)
    q = q.clamp(cmd.mdp.kin.q_lo, cmd.mdp.kin.q_hi)
    pose = torch.cat((p + env.scene.env_origins[env_ids], mat_to_quat(R)), -1)
    cmd.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    cmd.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
    cmd.robot.write_joint_state_to_sim(q, qd, joint_ids=cmd.joint_ids, env_ids=env_ids)
    cmd.robot.set_joint_position_target(q, joint_ids=cmd.joint_ids, env_ids=env_ids)
