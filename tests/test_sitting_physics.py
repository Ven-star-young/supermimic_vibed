"""Regression contracts for mixed-chair physics, references and safe Replay."""
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.tasks.sitting import variants


def source_chair(command):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="chair")
    body.add_freejoint(name="chair_free")
    body.explicitinertial = True
    body.mass, body.ipos = command.mass, [command.parts * .01, .02, .03]
    body.fullinertia = [1., 1.5, 2., .1 * command.parts, .1, .05]
    for part in range(command.parts):
        vertices = np.array([[x + .3 * part, y, z] for x in [-.1, .1]
                             for y in [-.1, .1] for z in [-.1, .1]])
        name = f"part{part}"
        spec.add_mesh(name=name, uservert=vertices.flatten())
        for floor in (False, True):
            body.add_geom(name=name + ("_ground" if floor else ""),
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, mass=0,
                contype=0 if floor else 2, conaffinity=1 if floor else 2,
                priority=0 if floor else 1, condim=3)
    return spec


@pytest.fixture
def variant_commands(monkeypatch):
    monkeypatch.setattr(variants, "make_chair_spec", source_chair)
    return [SimpleNamespace(parts=1, mass=10.), SimpleNamespace(parts=3, mass=20.)]


def test_padding_preserves_friction_roles_and_inertia(variant_commands):
    from mjlab.scene import Scene, SceneCfg
    cfg = variants.make_chair_variant_cfg(variant_commands)
    scene = Scene(SceneCfg(num_envs=4, entities={"chair": cfg},
        spec_fn=variants.normalize_variant_collision_slots), device="cpu")
    model = scene.compile()
    metadata = scene["chair"].variant_metadata
    assert len(metadata.slots) == 6
    for slot in metadata.slots:
        geom = model.geom("chair/" + slot.template_geom_name)
        floor = "chair_ground" in slot.key.body_path
        assert int(geom.contype[0]) == (0 if floor else 2)
        assert int(geom.conaffinity[0]) == (1 if floor else 2)
        assert int(geom.priority[0]) == (0 if floor else 1)
    for command in variant_commands:
        original = source_chair(command).compile()
        relocated = variants.make_variant_chair_spec(command).compile()
        for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
            np.testing.assert_allclose(getattr(original, field)[1], getattr(relocated, field)[1])
        assert relocated.body_mass[2] == 0


def test_native_variants_randomize_their_own_inertia_and_ignore_padding(variant_commands, cuda_device):
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg
    from supermimic.rotations import quat_to_mat
    from supermimic.tasks.sitting.events import ObjectPhysicsCfg, randomize_object_physics
    from supermimic.mdp.events import PhysicalContextCfg, randomize_physics

    def robot_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="torso_link", pos=[10, 0, 1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.1] * 3, contype=0, conaffinity=2)
        return spec

    def configure(spec):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .05], contype=1, conaffinity=0)
        variants.normalize_variant_collision_slots(spec)

    scene = Scene(SceneCfg(num_envs=2, entities={"robot": EntityCfg(spec_fn=robot_spec),
        "chair": variants.make_chair_variant_cfg(variant_commands)}, spec_fn=configure), device=cuda_device)
    scene.compile()
    sim = Simulation(cfg=SimulationCfg(nconmax=128, njmax=512), num_envs=2,
                     spec=scene.spec, variant_info=scene.collect_variant_info(), device=cuda_device)
    sim.expand_model_fields(("body_mass", "body_inertia", "body_ipos", "geom_friction"))
    scene.initialize(sim.mj_model, sim.model, sim.data)
    command = SimpleNamespace(cfg=SimpleNamespace(play=True, object_physical_cfg=ObjectPhysicsCfg(),
        physical_cfg=PhysicalContextCfg()), xi=torch.zeros(2, 46, device=cuda_device))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command),
                          num_envs=2, device=cuda_device, sim=sim, scene=scene)
    randomize_physics(env)
    randomize_object_physics(env)
    body = scene["chair"].indexing.root_body_id
    for world, cfg in enumerate(variant_commands):
        source = source_chair(cfg).compile()
        np.testing.assert_allclose(command.xi[world, 34:37].cpu(), source.body_ipos[1], atol=1e-7)
        rotation = quat_to_mat(torch.as_tensor(source.body_iquat[1], dtype=torch.float32))
        inertia = rotation @ torch.diag(torch.as_tensor(source.body_inertia[1], dtype=torch.float32)) @ rotation.T
        inertia *= 10. / cfg.mass
        torch.testing.assert_close(command.xi[world, 37:43].cpu(),
                                   inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])
    assert sim.model.body_mass[:, body].tolist() == [10., 10.]
    preserved = sim.model.body_inertia[1].clone()
    randomize_object_physics(env, torch.tensor([0], device=cuda_device))
    torch.testing.assert_close(sim.model.body_inertia[1], preserved)
    scene["chair"].write_root_link_pose_to_sim(
        torch.tensor([[0., 0., .08, 1., 0., 0., 0.]], device=cuda_device).expand(2, -1))
    sim.forward()
    count = int(sim.data.nacon[0])
    assert count > 0
    geoms = sim.data.contact.geom[:count]
    world = sim.data.contact.worldid[:count].long()
    chair_contacts = torch.isin(geoms, scene["chair"].indexing.geom_ids)
    chair_geoms = geoms[chair_contacts]
    worlds = world[:, None].expand_as(geoms)[chair_contacts]
    assert sim.model.geom_dataid[worlds, chair_geoms].ge(0).all()
    assert np.all(sim.mj_model.geom_contype[chair_geoms.cpu().numpy()] == 0)
    assert not sim.data.overflow.any()


def test_sizing_bounds_real_dense_scene_and_keeps_user_callback(cuda_device):
    from mjlab.scene import Scene, SceneCfg
    from mjlab.entity import EntityCfg
    from mjlab.sim import Simulation, SimulationCfg
    from supermimic.tasks.sitting.env_cfgs import _size_contact_buffers

    def dense_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="chair")
        body.add_freejoint()
        for i in range(256):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.005, .005, .05],
                          pos=[3 + .02 * i, 2, 0], contype=1, conaffinity=1, condim=3)
        return spec

    calls = []
    cfg = SimulationCfg(nconmax=512, njmax=2000)
    def configure(spec):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .05])
        _size_contact_buffers(spec, sim_cfg=cfg, previous=lambda _: calls.append(True))
    scene = Scene(SceneCfg(num_envs=2, entities={"chair": EntityCfg(spec_fn=dense_spec)},
                           spec_fn=configure), device=cuda_device)
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert calls == [True] and data.ncon > 512
    assert data.ncon <= cfg.nconmax <= 2048 and data.nefc <= cfg.njmax <= 8192
    sim = Simulation(num_envs=2, cfg=cfg, spec=scene.spec, device=cuda_device)
    assert sim.data.naconmax >= 2 * data.ncon and sim.data.njmax >= data.nefc


def test_balanced_assignment_is_seeded_and_covers_every_chair():
    from supermimic.tasks.sitting.mixed import balanced_assignment
    first = balanced_assignment(512, num_scenes=83, seed=0)
    assert first == balanced_assignment(512, num_scenes=83, seed=0)
    assert first != balanced_assignment(512, num_scenes=83, seed=1)
    counts = np.bincount(first, minlength=83)
    assert counts.min() == 6 and counts.max() == 7
    with pytest.raises(ValueError, match="at least"):
        balanced_assignment(82, num_scenes=83, seed=0)


def test_mixed_config_reconstruction_is_lazy_and_keeps_current_commands(monkeypatch):
    from copy import deepcopy
    from supermimic.tasks.sitting import assets
    from supermimic.tasks.sitting.experiment import mixed_scene_cfg
    from supermimic.tasks.sitting.mixed import _mixed_scene_spec
    from supermimic.tasks.sitting.env_cfgs import _size_contact_buffers
    monkeypatch.setattr(assets, "load_motion", lambda *_: pytest.fail("Configuration read a motion asset"))
    manifest = dict(fingerprint="fixture", test=[], train=[
        dict(shape_key=f"shape{i}", clips=[dict(path=f"motion{i}.npz", geometry_key=f"geometry{i}")])
        for i in range(2)])
    cfg = mixed_scene_cfg(manifest, "/missing/motions", "/missing/cache", num_envs=4)
    replaced = deepcopy(replace(cfg, commands={"motion": replace(cfg.commands["motion"],
        collision_profile="compact", assignment_seed=11)}))
    assert cfg.commands["motion"].collision_profile == "default"
    assert replaced.commands["motion"].collision_profile == "compact"
    assert replaced.scene.spec_fn.func is _mixed_scene_spec
    sizing = replaced.scene.spec_fn.keywords["size_callback"]
    assert sizing.func is _size_contact_buffers and sizing.keywords["sim_cfg"] is replaced.sim
    assert sizing.keywords["previous"] is None
    for spec_fn in replaced.scene.entities["chair"].variants.values():
        assert spec_fn.args[0].collision_profile == "compact"
    assert set(replaced.scene.entities) == {"robot", "chair"}
    assert "geometry_id" in replaced.observations


def test_sitting_config_rejects_unmodeled_entities_before_rebuilding():
    from mjlab.entity import EntityCfg
    from supermimic.tasks.sitting.env_cfgs import make_env_cfg
    cfg = make_env_cfg()
    changed_scene = replace(cfg.scene, entities={**cfg.scene.entities, "obstacle": EntityCfg()})
    with pytest.raises(ValueError, match="unsupported scene entities.*obstacle"):
        replace(cfg, scene=changed_scene)
    assert set(cfg.scene.entities) == {"robot", "chair"}
    assert "obstacle" in changed_scene.entities


def test_mixed_native_resets_keep_reference_geometry_and_replay_aligned(tmp_path, monkeypatch, cuda_device):
    import importlib.metadata
    import sys
    import trimesh
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from motion_data import collision, load_motion, save_motion
    from motion_data.chair_split import create_split
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
    from supermimic.tasks.sitting.experiment import mixed_scene_cfg
    from supermimic.rl.vecenv import ReplayEnvAdapter
    from tests.helpers import reset_frames

    root, cache = tmp_path / "motions", tmp_path / "cache"
    # The fixture already consists of convex boxes; replace offline CoACD only.
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
        lambda name: collision.COACD_VERSION if name == "coacd" else version(name))
    monkeypatch.setitem(sys.modules, "coacd", SimpleNamespace(set_log_level=lambda _: None,
        Mesh=lambda vertices, faces: (vertices, faces), run_coacd=lambda mesh, **_: [mesh]))
    for chair in range(3):
        mesh = trimesh.creation.box(extents=[.4 + .1 * chair, .4, .5])
        for clip in range(2):
            path = root / f"chair{chair}_clip{clip}.npz"
            save_motion(path, dict(fps=25., joint_names=G1_JOINT_NAMES,
                root_pos=np.tile([.1 * clip, 0., 1.5], (20, 1)),
                root_quat=np.tile([1., 0., 0., 0.], (20, 1)),
                joint_pos=np.tile(default_joint_pos_list(), (20, 1)),
                object_pos=np.tile([2., 0., .25], (20, 1)),
                object_quat=np.tile([1., 0., 0., 0.], (20, 1)), object_scale=np.ones(3),
                object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces)))
            collision.prepare_collision(load_motion(path), cache)
    manifest = create_split(root, tmp_path / "split.json")
    cfg = mixed_scene_cfg(manifest, root, cache, num_envs=4)
    cfg.commands["motion"].reset_noise = False
    native = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        command = native.command_manager.get_term("motion")
        replay = ReplayEnvAdapter(RslRlVecEnvWrapper(native))
        assert command.env_scene.bincount().tolist() == [2, 2]
        assert len(command.geometry_bank.keys) == 2
        ids = torch.arange(4, device=cuda_device)
        assignment = command.env_scene.clone()
        expected_geometry = torch.tensor(command.scene_geometry_ids, device=cuda_device)[assignment]
        for _ in range(3):
            observations, _ = native.reset()
            torch.testing.assert_close(command.clip_scene[command.motion.clip_id[command.time_steps]], assignment)
            torch.testing.assert_close(command.xi[:, 45].long(), expected_geometry)
            assert observations["policy_history"].shape == (4, 1020)
            torch.testing.assert_close(observations["geometry_id"], command.xi[:, 45:46])
        frames = command.motion.clip_start[assignment * 2 + ids % 2] + 3
        reset_frames(native, frames)
        for actual, reference in zip(command.state(), command.motion.full_ref_state(frames)):
            torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
        subset = ids[[0, 2]]
        with pytest.raises(ValueError, match="assigned chair"):
            command.set_reset_frames(subset, command.motion.clip_start[(1 - assignment[subset]) * 2])
        before_frame, before_context = command.time_steps.clone(), command.xi.clone()
        transition = replay.step(torch.zeros(4, 29, device=cuda_device))
        torch.testing.assert_close(transition["f"], before_frame)
        torch.testing.assert_close(transition["xi"], before_context)
        torch.testing.assert_close(command.env_scene, assignment)
        assert transition["s"].shape == (4, 84) and transition["xi"].shape == (4, 46)
    finally:
        native.close()


def test_replay_rejects_sticky_physics_overflow_before_recording(eval_env):
    from mujoco_warp._src.types import OverflowType
    # Warp ORs these flags across substeps until reset. A later substep with
    # fewer contacts must not make a previously truncated transition usable.
    flags = eval_env.env.sim.data.overflow
    flags[0] = int(OverflowType.NEFC)
    try:
        with pytest.raises(RuntimeError, match="overflow.*NEFC"):
            eval_env.step(torch.zeros(eval_env.N, 29, device=eval_env.device))
    finally:
        flags.zero_()


def test_sitting_scene_reserves_epa_edges_without_changing_contact_geometry(cuda_device, monkeypatch):
    """Real convex collision with an intentionally undersized EPA edge buffer."""
    import numpy as np
    import mujoco_warp as mjwarp
    import warp as wp
    from scipy.spatial.transform import Rotation
    from mujoco_warp._src import collision_convex
    from mujoco_warp._src.types import OverflowType
    from mjlab.sim import SimulationCfg
    from supermimic.tasks.sitting.env_cfgs import _size_contact_buffers

    # Near-parallel box poses from MuJoCo-Warp's Apache-2.0
    # collision_gjk_test.test_box_box_horizon (needs 13 boundary edges).
    positions = [[.065118454396725, -.125125020742416, .124963559210300],
                 [.065104484558105, -.124979749321938, .174992129206657]]
    rotations = [
        [.996357858181, .085266821086407, -.000942531623878,
         -.085266284644604, .996358215808868, .000591202871874,
         .000989508931525, -.000508683384396, .999999582767487],
        [.996556758880615, -.082913912832737, -.000453041866422,
         .082915119826794, .996536433696747, .006357696373016,
         -.000075668765930, -.006373368669301, .999979794025421],
    ]
    spec = mujoco.MjSpec()
    spec.add_mesh(name="box", uservert=np.array([
        [i, j, k] for i in [-.025, .025] for j in [-.025, .025] for k in [-.025, .025]
    ]).flatten().tolist())
    for index, (pos, rotation) in enumerate(zip(positions, rotations)):
        quat = np.roll(Rotation.from_matrix(np.array(rotation).reshape(3, 3)).as_quat(), 1)
        body = spec.worldbody.add_body(name=f"box{index}", pos=pos, quat=quat)
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="box")
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    monkeypatch.setattr(collision_convex, "MJ_MAX_EPAHORIZON", 5)
    with wp.ScopedDevice(cuda_device):
        warp_model = mjwarp.put_model(model)
        def collide():
            warp_data = mjwarp.put_data(model, data, nworld=1, nconmax=32, njmax=128)
            mjwarp.forward(warp_model, warp_data)
            wp.synchronize_device(cuda_device)
            return warp_data
        bad = collide()
        assert int(bad.overflow.numpy()[0]) & int(OverflowType.EPA_HORIZON)
        # The same lazy callback used when constructing a chair scene grows
        # the backend allocation, before step/forward graphs capture it.
        _size_contact_buffers(spec, sim_cfg=SimulationCfg())
        good = collide()
        assert int(good.overflow.numpy()[0]) == 0
        assert int(good.nacon.numpy()[0]) > 0
