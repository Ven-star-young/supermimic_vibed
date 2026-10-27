"""Many convex parts must fit the real MuJoCo-Warp contact allocation."""
import mujoco
from copy import deepcopy
from dataclasses import replace


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
    # Exercise the allocator itself, not just an estimate mirroring the fix.
    simulation = Simulation(num_envs=2, cfg=cfg.sim, spec=scene.spec, device=cuda_device)
    assert simulation.data.naconmax >= data.ncon * 2
    assert simulation.data.njmax >= data.nefc
