"""Per-episode object mass and robot-object friction, recorded in Replay."""

from dataclasses import dataclass

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from ...rotations import quat_to_mat
from ...mdp.events import default_model_value


@dataclass(kw_only=True)
class ObjectPhysicsCfg:
    mass: tuple[float, float] = (5.0, 15.0)
    evaluation_mass: float = 10.0
    robot_friction: tuple[float, float] = (0.3, 1.2)
    evaluation_robot_friction: float = 1.0
    ground_friction: float = 1.0

    def __post_init__(self):
        if not 0 < self.mass[0] <= self.mass[1] or self.evaluation_mass <= 0:
            raise ValueError("Object masses must be positive with an ordered training range")
        if (not 0 <= self.robot_friction[0] <= self.robot_friction[1]
                or self.evaluation_robot_friction < 0 or self.ground_friction < 0):
            raise ValueError("Object friction must be nonnegative with an ordered training range")


@requires_model_fields("body_mass", "body_inertia", "geom_friction", recompute=RecomputeLevel.set_const)
def randomize_object_physics(env, env_ids=None, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    cfg, sim = cmd.cfg.object_physical_cfg, env.sim
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    chair = env.scene["chair"]
    body = chair.indexing.root_body_id
    sample = lambda limits: torch.empty(len(ids), device=env.device).uniform_(*limits)
    mass = (torch.full((len(ids),), cfg.evaluation_mass, device=env.device)
            if cmd.cfg.play else sample(cfg.mass))
    friction = (torch.full_like(mass, cfg.evaluation_robot_friction)
                if cmd.cfg.play else sample(cfg.robot_friction))
    scale = mass / default_model_value(sim, "body_mass", ids, body)
    principal = default_model_value(sim, "body_inertia", ids, body) * scale[:, None]
    sim.model.body_mass[ids, body] = mass
    sim.model.body_inertia[ids, body] = principal
    geoms = chair.indexing.geom_ids
    contact = geoms[torch.as_tensor(sim.mj_model.geom_contype[geoms.cpu().numpy()] == 2, device=env.device)]
    sim.model.geom_friction[ids[:, None], contact[None], 0] = friction[:, None]
    rotation = quat_to_mat(default_model_value(sim, "body_iquat", ids, body))
    inertia = rotation @ torch.diag_embed(principal) @ rotation.transpose(-1, -2)
    cmd.xi[ids, 33] = mass
    cmd.xi[ids, 34:37] = default_model_value(sim, "body_ipos", ids, body)
    cmd.xi[ids, 37:43] = inertia[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    cmd.xi[ids, 43] = friction
    cmd.xi[ids, 44] = cfg.ground_friction
