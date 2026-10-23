"""mjlab events for physical context and reference-state initialization.

These run only in the simulator. Differentiable rewards/observations live in
separate tensor modules and consume the recorded physical context.
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


@requires_model_fields("geom_friction", "body_ipos", recompute=RecomputeLevel.set_const)
def randomize_physics(env, env_ids, cfg: PhysicalContextCfg):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[cfg.entity_name]
    if env.eval_mode:
        env.xi[ids, 0] = cfg.evaluation_friction
    else:
        env.xi[ids, 0] = torch.empty(len(ids), device=env.device).uniform_(*cfg.friction)
        for axis, limits in enumerate(cfg.com_ranges):
            env.xi[ids, cfg.com_start + axis] = torch.empty(len(ids), device=env.device).uniform_(*limits)
    geoms = robot.indexing.geom_ids
    env.sim.model.geom_friction[ids[:, None], geoms[None], 0] = env.xi[ids, :1]
    planes = torch.as_tensor(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=env.device).nonzero().flatten()
    env.sim.model.geom_friction[ids[:, None], planes[None], 0] = cfg.evaluation_friction if env.eval_mode else cfg.friction[0]
    torso = int(robot.indexing.body_ids[robot.body_names.index(cfg.torso_body)])
    env.sim.model.body_ipos[ids, torso] = env.sim.get_default_field("body_ipos")[torso] + env.xi[ids, cfg.com_start:cfg.com_start + 3]


def randomize_joint_offsets(env, env_ids, cfg: PhysicalContextCfg):
    begin, count = cfg.calibration_start, env.action.action_dim
    env.xi[env_ids, begin:begin + count] = (
        0.0 if env.eval_mode else torch.empty(len(env_ids), count, device=env.device).uniform_(*cfg.joint_offset)
    )


def reset_robot_to_reference(env, env_ids, cfg: ReferenceResetCfg):
    frames = env.command.time_steps[env_ids]
    p, R, v, w, q, qd = (value.clone() for value in env.motion.ref_state(frames))
    if env.reset_noise:
        def perturb(value, bounds):
            return value + (2 * torch.rand_like(value) - 1) * torch.tensor(bounds, device=env.device)
        p = perturb(p, cfg.position)
        rotation = perturb(torch.zeros(len(env_ids), 3, device=env.device), cfg.rotation)
        R = so3_exp(rotation) @ R
        v, w = perturb(v, cfg.linear_velocity), perturb(w, cfg.angular_velocity)
        q = q + torch.empty_like(q).uniform_(-cfg.joint_position, cfg.joint_position)
    q = q.clamp(env.mdp.kin.q_lo, env.mdp.kin.q_hi)
    pose = torch.cat((p + env.scene.env_origins[env_ids], mat_to_quat(R)), -1)
    env.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    env.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
    env.robot.write_joint_state_to_sim(q, qd, joint_ids=env.joint_ids, env_ids=env_ids)
    env.robot.set_joint_position_target(q, joint_ids=env.joint_ids, env_ids=env_ids)
