"""Per-episode door physics, shared by the simulator and Replay context."""

from dataclasses import dataclass
import math

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from ...mdp.events import default_model_value


@dataclass(kw_only=True)
class DoorRandomizationCfg:
    enabled: bool = True
    mass: tuple[float, float] = (5.0, 15.0)
    damping: tuple[float, float] = (2.5, 7.5)
    hinge_friction: tuple[float, float] = (0.25, 0.75)
    stiffness: tuple[float, float] = (10.0, 30.0)
    robot_friction: tuple[float, float] = (0.3, 1.2)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("Door randomization enabled must be boolean")
        for name in ("mass", "damping", "hinge_friction", "stiffness", "robot_friction"):
            bounds = tuple(getattr(self, name))
            if (len(bounds) != 2 or any(not math.isfinite(v) for v in bounds)
                    or not 0 <= bounds[0] <= bounds[1] or (name == "mass" and bounds[0] == 0)):
                raise ValueError(f"Door {name} needs finite ordered nonnegative bounds (mass must be positive)")
            setattr(self, name, bounds)


@requires_model_fields("body_mass", "body_inertia", "dof_damping", "dof_frictionloss",
                       "jnt_stiffness", "geom_friction", recompute=RecomputeLevel.set_const)
def randomize_door_physics(env, env_ids=None, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    cfg, sim = cmd.cfg, env.sim
    randomization = cfg.door_randomization_cfg
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    door = env.scene["door"]
    joint = int(door.indexing.joint_ids[0])
    # A fixed-base entity has a mocap parent; the hinged child owns the mass.
    body = int(sim.mj_model.jnt_bodyid[joint])
    dof = int(sim.mj_model.jnt_dofadr[joint])
    panel = int(sim.mj_model.geom("door/door_panel").id)

    def sample(name, nominal):
        values = torch.empty(len(ids), device=env.device)
        if cfg.play or not randomization.enabled:
            return values.fill_(nominal)
        return values.uniform_(*getattr(randomization, name))

    mass = sample("mass", cfg.door_mass)
    damping = sample("damping", cfg.door_damping)
    friction = sample("hinge_friction", cfg.door_friction)
    stiffness = sample("stiffness", cfg.door_stiffness)
    contact_friction = sample("robot_friction", cfg.robot_door_friction)
    # Scale from immutable asset defaults, so repeated resets cannot compound
    # inertia changes. Shape and COM stay fixed: mass determines the inertia.
    ratio = mass / default_model_value(sim, "body_mass", ids, body)
    sim.model.body_mass[ids, body] = mass
    sim.model.body_inertia[ids, body] = default_model_value(sim, "body_inertia", ids, body) * ratio[:, None]
    sim.model.dof_damping[ids, dof] = damping
    sim.model.dof_frictionloss[ids, dof] = friction
    sim.model.jnt_stiffness[ids, joint] = stiffness
    sim.model.geom_friction[ids, panel, 0] = contact_friction
    # Keep ground friction (42), spring reference (45) and static geometry intact.
    cmd.xi[ids, 39] = mass
    cmd.xi[ids, 40] = damping
    cmd.xi[ids, 41] = friction
    cmd.xi[ids, 43] = contact_friction
    cmd.xi[ids, 44] = stiffness
