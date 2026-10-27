"""Native mixed chair slots must preserve friction roles and per-world inertia."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.tasks.entities import variants


def source_chair(command, index=0):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="chair")
    body.add_freejoint(name="chair_free")
    body.explicitinertial = True
    body.mass = command.mass
    body.ipos = [command.parts * .01, .02, .03]
    body.fullinertia = [1., 1.5, 2., .1 * command.parts, .1, .05]
    for part in range(command.parts):
        vertices = np.array([[x + .3 * part, y, z] for x in [-.1, .1]
                             for y in [-.1, .1] for z in [-.1, .1]])
        name = f"part{part}"
        spec.add_mesh(name=name, uservert=vertices.flatten())
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                      mass=0, contype=2, conaffinity=2, priority=1, condim=3)
        body.add_geom(name=name + "_ground", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                      mass=0, contype=0, conaffinity=1, priority=0, condim=3)
    return spec


@pytest.fixture
def variant_commands(monkeypatch):
    monkeypatch.setattr(variants, "make_object_spec", source_chair)
    return [SimpleNamespace(object_enabled=True, parts=1, mass=10.),
            SimpleNamespace(object_enabled=True, parts=3, mass=20.)]


def test_variant_padding_keeps_robot_and_floor_roles_separate(variant_commands):
    from mjlab.scene import Scene, SceneCfg
    cfg = variants.make_chair_variant_cfg(variant_commands)
    scene = Scene(SceneCfg(num_envs=4, entities={"chair": cfg},
                           spec_fn=variants.normalize_variant_collision_slots), device="cpu")
    metadata = scene["chair"].variant_metadata
    assert len(metadata.slots) == 6
    model = scene.compile()
    for slot in metadata.slots:
        geom = model.geom("chair/" + slot.template_geom_name)
        floor = "chair_ground" in slot.key.body_path
        assert int(geom.contype[0]) == (0 if floor else 2)
        assert int(geom.conaffinity[0]) == (1 if floor else 2)
        assert int(geom.priority[0]) == (0 if floor else 1)
        for sources in metadata.variant_slot_specs:
            filled = sources[metadata.slots.index(slot)]
            if filled is not None:
                assert filled.contype == (0 if floor else 2)
                assert filled.conaffinity == (1 if floor else 2)
    for command in variant_commands:
        original = source_chair(command).compile()
        relocated = variants.make_variant_chair_spec(command).compile()
        for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
            np.testing.assert_allclose(getattr(original, field)[1], getattr(relocated, field)[1])
        assert relocated.body_mass[2] == 0


def test_native_mixed_chair_contacts_and_randomized_inertia(variant_commands, cuda_device):
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg
    from supermimic.rl.entity_state import EntityDynamics
    from supermimic.rotations import quat_to_mat
    from supermimic.tasks.entities.events import ObjectPhysicsCfg, randomize_entity_physics
    from supermimic.mdp.events import PhysicalContextCfg, randomize_physics

    def robot_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="torso_link", pos=[10, 0, 1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.1] * 3,
                      contype=0, conaffinity=2)
        return spec

    def configure(spec):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .05], contype=1, conaffinity=0)
        variants.normalize_variant_collision_slots(spec)

    scene = Scene(SceneCfg(num_envs=2, entities={
        "robot": EntityCfg(spec_fn=robot_spec),
        "chair": variants.make_chair_variant_cfg(variant_commands),
    }, spec_fn=configure), device=cuda_device)
    # The production sizing callback compiles the assembled original spec,
    # assigning IDs to the attached entities before the variant model copies it.
    scene.compile()
    sim = Simulation(cfg=SimulationCfg(nconmax=128, njmax=512), num_envs=2,
                     spec=scene.spec, variant_info=scene.collect_variant_info(), device=cuda_device)
    sim.expand_model_fields(("body_mass", "body_inertia", "body_ipos", "geom_friction"))
    scene.initialize(sim.mj_model, sim.model, sim.data)
    assert sim.world_to_variant["chair"].tolist() == [0, 1]
    dynamics = EntityDynamics(1)
    command = SimpleNamespace(
        cfg=SimpleNamespace(object_enabled=True, extra_objects=(), play=True,
                            object_physical_cfg=ObjectPhysicsCfg(), physical_cfg=PhysicalContextCfg()),
        object_entity_names=("chair",), dynamics=dynamics,
        xi=dynamics.empty_context(2, device=cuda_device),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command),
                          num_envs=2, device=cuda_device, sim=sim, scene=scene)
    # Robot COM also has a leading world axis whenever any entity is variant.
    randomize_physics(env)
    randomize_entity_physics(env)
    body = scene["chair"].indexing.root_body_id
    contexts = dynamics.object_context(command.xi)[:, 0]
    for world, cfg in enumerate(variant_commands):
        source = source_chair(cfg).compile()
        np.testing.assert_allclose(contexts[world, 1:4].cpu(), source.body_ipos[1], atol=1e-7)
        rotation = quat_to_mat(torch.as_tensor(source.body_iquat[1], dtype=torch.float32))
        expected = rotation @ torch.diag(torch.as_tensor(source.body_inertia[1], dtype=torch.float32)) @ rotation.T
        expected *= 10. / cfg.mass
        torch.testing.assert_close(contexts[world, 4:10].cpu(),
                                   expected[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])
    assert sim.model.body_mass[:, body].tolist() == [10., 10.]
    preserved = sim.model.body_inertia[1].clone()
    randomize_entity_physics(env, torch.tensor([0], device=cuda_device))
    torch.testing.assert_close(sim.model.body_inertia[1], preserved)
    # Actual contact detection must ignore absent slots and use floor copies.
    chair = scene["chair"]
    pose = torch.tensor([[0., 0., .08, 1., 0., 0., 0.]], device=cuda_device).expand(2, -1)
    chair.write_root_link_pose_to_sim(pose)
    sim.forward()
    count = int(sim.data.nacon[0])
    assert count > 0
    geom = sim.data.contact.geom[:count]
    world = sim.data.contact.worldid[:count].long()
    chair_ids = chair.indexing.geom_ids
    chair_contacts = torch.isin(geom, chair_ids)
    chair_geom = geom[chair_contacts]
    worlds = world[:, None].expand_as(geom)[chair_contacts]
    assert sim.model.geom_dataid[worlds, chair_geom].ge(0).all()
    assert np.all(sim.mj_model.geom_contype[chair_geom.cpu().numpy()] == 0)
    assert np.all(sim.mj_model.geom_conaffinity[chair_geom.cpu().numpy()] == 1)
    assert not sim.data.overflow.any()
