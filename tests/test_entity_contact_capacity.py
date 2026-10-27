"""Many convex parts must fit the real MuJoCo-Warp contact allocation."""
import mujoco
from copy import deepcopy
from dataclasses import replace
import pytest
import torch


def test_dense_object_contact_buffers_fit_compiled_scene(cuda_device, monkeypatch):
    from mjlab.scene import Scene
    from mjlab.sim import Simulation
    from supermimic.tasks.entities import env_cfgs
    from supermimic.tasks.entities.assets import ObjectCfg

    def dense_object(*_):
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="box")
        body.add_freejoint()
        # Disjoint parts of one rigid body, away from G1, penetrating the plane
        # in qpos0 just as a detailed collision decomposition can at startup.
        for i in range(256):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.005, .005, .05],
                          pos=[3 + .02 * i, 2, 0], contype=1, conaffinity=1, condim=3)
        return spec

    monkeypatch.setattr(env_cfgs, "make_object_spec", dense_object)
    cfg = env_cfgs.make_env_cfg(object_enabled=False, extra_objects=(ObjectCfg(),))
    cfg.scene.num_envs = 2
    callbacks = []
    cfg = replace(cfg, scene=replace(cfg.scene, spec_fn=lambda _: callbacks.append(True)))
    # Match CLI reconstruction and the multi-chair runner's per-visit deepcopy.
    original = cfg
    cfg = deepcopy(replace(cfg))
    scene = Scene(cfg.scene, device=cuda_device)
    assert callbacks == [True]
    assert original.sim.nconmax == 512 and original.sim.njmax == 2000
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 512 and data.nefc > 2000
    # Doubling then rounding used to allocate 4096/16384 for this 1065-contact
    # fixture. Across hundreds of worlds that wastes CUDA graph workspace.
    assert cfg.sim.nconmax <= 2048
    assert cfg.sim.njmax <= 8192
    # Exercise the allocator itself, not just an estimate mirroring the fix.
    simulation = Simulation(num_envs=2, cfg=cfg.sim, spec=scene.spec, device=cuda_device)
    assert simulation.data.naconmax >= data.ncon * 2
    assert simulation.data.njmax >= data.nefc


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


def test_entity_scene_reserves_epa_edges_without_changing_contact_geometry(cuda_device, monkeypatch):
    """Real convex collision with an intentionally undersized EPA edge buffer."""
    import numpy as np
    import mujoco_warp as mjwarp
    import warp as wp
    from scipy.spatial.transform import Rotation
    from mujoco_warp._src import collision_convex
    from mujoco_warp._src.types import OverflowType
    from mjlab.sim import SimulationCfg
    from supermimic.tasks.entities.env_cfgs import _size_contact_buffers

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
        # The same lazy callback used when constructing an entity scene grows
        # the backend allocation, before step/forward graphs capture it.
        _size_contact_buffers(spec, sim_cfg=SimulationCfg())
        good = collide()
        assert int(good.overflow.numpy()[0]) == 0
        assert int(good.nacon.numpy()[0]) > 0
