"""Per-episode entity physics and exact effective contact-pair context."""

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from ...rotations import quat_to_mat
from ..interaction.events import ObjectPhysicsCfg


@requires_model_fields("body_mass", "body_inertia", "geom_friction", recompute=RecomputeLevel.set_const)
def randomize_entity_physics(env, env_ids=None, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    cfg, sim, dynamics = cmd.cfg.object_physical_cfg, env.sim, cmd.dynamics
    obj_context = dynamics.object_context(cmd.xi)
    pairs = dynamics.pair_friction(cmd.xi)
    pairs[ids] = 0
    sample = lambda limits: torch.empty(len(ids), device=env.device).uniform_(*limits)
    plane_ids = torch.as_tensor(sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE,
                                device=env.device).nonzero().flatten()
    if len(plane_ids) != 1:
        raise ValueError("Entity task v1 requires exactly one plane ground")
    ground_friction = sim.model.geom_friction[ids, plane_ids[0]]
    robot_geoms = env.scene["robot"].indexing.geom_ids
    robot_friction = sim.model.geom_friction[ids, robot_geoms[0]]
    ground_index = dynamics.num_entities - 1
    pairs[ids, 0, ground_index] = pairs[ids, ground_index, 0] = torch.maximum(robot_friction, ground_friction)
    contacts = []
    for index, name in enumerate(cmd.object_entity_names):
        target = cmd.cfg.object_enabled and index == 0
        extra = None if target else cmd.cfg.extra_objects[index - int(cmd.cfg.object_enabled)]
        entity = env.scene[name]
        body = entity.indexing.root_body_id
        dynamic = target or extra.dynamic
        randomized = not cmd.cfg.play and dynamic and (target or extra.randomize_physics)
        fixed_mass = cfg.evaluation_mass if target else extra.mass
        fixed_mu = cfg.evaluation_robot_friction if target else extra.friction
        mass = sample(cfg.mass) if randomized else torch.full((len(ids),), fixed_mass, device=env.device)
        friction = sample(cfg.robot_friction) if randomized else torch.full_like(mass, fixed_mu)
        principal = sim.get_default_field("body_inertia")[body] * (
            mass / sim.get_default_field("body_mass")[body])[:, None]
        sim.model.body_mass[ids, body] = mass
        sim.model.body_inertia[ids, body] = principal
        geom_ids = entity.indexing.geom_ids
        main = geom_ids[torch.as_tensor(sim.mj_model.geom_contype[geom_ids.cpu().numpy()] == 2,
                                        device=env.device)]
        floor_copy = geom_ids[torch.as_tensor([
            sim.mj_model.geom(int(geom)).name.endswith("_ground") for geom in geom_ids
        ], device=env.device)]
        sim.model.geom_friction[ids[:, None], main[None], 0] = friction[:, None]
        # Main copy has priority 1, robot has priority 0: all friction components
        # are taken from the object. Object-object copies have equal priorities.
        main_friction = sim.model.geom_friction[ids, main[0]]
        contacts.append(main_friction)
        pairs[ids, 0, index + 1] = pairs[ids, index + 1, 0] = main_friction
        floor_friction = torch.maximum(sim.model.geom_friction[ids, floor_copy[0]], ground_friction)
        pairs[ids, index + 1, ground_index] = pairs[ids, ground_index, index + 1] = floor_friction
        rotation = quat_to_mat(torch.as_tensor(sim.mj_model.body_iquat[body], device=env.device, dtype=torch.float32))
        inertia = rotation @ torch.diag_embed(principal) @ rotation.T
        obj_context[ids, index, 0] = mass
        obj_context[ids, index, 1:4] = sim.get_default_field("body_ipos")[body]
        obj_context[ids, index, 4:10] = inertia[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    for i in range(len(contacts)):
        for j in range(i + 1, len(contacts)):
            pairs[ids, i + 1, j + 1] = pairs[ids, j + 1, i + 1] = torch.maximum(contacts[i], contacts[j])
