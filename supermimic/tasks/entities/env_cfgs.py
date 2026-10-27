"""Lazy native scenes sharing one variable-entity physics contract."""

from dataclasses import dataclass, replace
from functools import partial
import math
import os

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg

from ...assets import DATA_DIR
from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from .assets import load_object_configs, make_object_spec, object_names
from .commands import EntityCommandCfg
from .events import randomize_entity_physics


def _size_contact_buffers(spec, *, sim_cfg, previous=None, epa_horizon_capacity=64):
    """Size from the assembled scene before mjlab uploads its qpos0 to Warp.

    Detailed convex decompositions can exceed the per-object floor even before
    reset. Keep 25% headroom in small aligned blocks; doubling then rounding to
    powers of two could reserve almost four times the required CUDA workspace.
    Explicit larger capacities remain lower bounds.
    """
    if previous is not None:
        previous(spec)
    # MuJoCo-Warp 3.11 allocates this edge scratch array in Python when capturing
    # convex collision graphs. It is independent of contact count / njmax and
    # has no public SimulationCfg option. Keep the version-specific adjustment
    # here, before graph capture; never shrink a larger process-wide allocation.
    from mujoco_warp._src import collision_convex
    collision_convex.MJ_MAX_EPAHORIZON = max(collision_convex.MJ_MAX_EPAHORIZON, epa_horizon_capacity)
    model = spec.compile()
    sim_cfg.mujoco.apply(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for field, count, block in (("nconmax", data.ncon, 128), ("njmax", data.nefc, 512)):
        required = math.ceil(math.ceil(1.25 * count) / block) * block
        old = getattr(sim_cfg, field) or 0
        if required > old:
            setattr(sim_cfg, field, required)
            print(f"[entity contacts] {field}: {old} -> {required} (startup {count})", flush=True)


@dataclass(kw_only=True)
class EntityEnvCfg(ManagerBasedRlEnvCfg):
    epa_horizon_capacity: int = 64
    """Minimum convex EPA boundary-edge capacity per collision pair (24 upstream)."""

    def __post_init__(self):
        if not 24 <= self.epa_horizon_capacity <= 256:
            raise ValueError("epa_horizon_capacity must be between 24 and 256")
        command = self.commands["motion"]
        if command.extra_objects_file is not None:
            command = replace(command, extra_objects=load_object_configs(command.extra_objects_file))
            self.commands = {**self.commands, "motion": command}
        names = object_names(command)
        # Reconstruct after Tyro/dataclasses.replace so asset closures use the
        # current command. Only this task owns the entity scene declaration.
        entities = {"robot": self.scene.entities["robot"]}
        for index, name in enumerate(names):
            target = command.object_enabled and index == 0
            extra = None if target else command.extra_objects[index - int(command.object_enabled)]
            position, quaternion = ((0., 0., 2.), (1., 0., 0., 0.)) if target else (extra.pos, extra.quat)
            entities[name] = EntityCfg(spec_fn=partial(make_object_spec, command, index),
                init_state=EntityCfg.InitialStateCfg(pos=position, rot=quaternion))
        previous = self.scene.spec_fn
        # Config reconstruction/deepcopy must rebind to the current simulation
        # config and preserve a user callback without wrapping ourselves twice.
        if isinstance(previous, partial) and previous.func is _size_contact_buffers:
            previous = previous.keywords["previous"]
        self.events = dict(self.events)
        # Even empty scenes record robot-ground effective pair properties.
        self.events["entity_physics"] = EventTermCfg(func=randomize_entity_physics, mode="reset")
        if names:
            # Capacity depends on actual entities, never the padded token count.
            self.sim = replace(self.sim, nconmax=max(512 * len(names), self.sim.nconmax or 0),
                               njmax=max(2000 * len(names), self.sim.njmax or 0))
            spec_fn = partial(_size_contact_buffers, sim_cfg=self.sim, previous=previous,
                              epa_horizon_capacity=self.epa_horizon_capacity)
        else:
            spec_fn = previous
        self.scene = replace(self.scene, entities=entities, spec_fn=spec_fn)


def make_env_cfg(play=False, *, object_enabled=True, max_objects=None, extra_objects=()):
    baseline = make_tracking_env_cfg(play)
    baseline.scene.num_envs = 32 if play else 128
    if max_objects is None:
        max_objects = int(os.environ.get("SUPERMIMIC_MAX_OBJECTS", "8"))
    command = EntityCommandCfg(
        **{name: getattr(baseline.commands["motion"], name)
           for name in baseline.commands["motion"].__dataclass_fields__},
        object_enabled=object_enabled, max_objects=max_objects, extra_objects=extra_objects,
        extra_objects_file=(os.environ.get("SUPERMIMIC_EXTRA_OBJECTS_FILE") if not extra_objects else None),
    )
    command.motion_dir = os.environ.get("SUPERMIMIC_MOTION_DIR",
        str(DATA_DIR / "motions" / ("sitting" if object_enabled else "lafan1")))
    baseline.commands["motion"] = command
    baseline.scene.entities["robot"] = make_g1_robot_cfg(interaction_collisions=True)
    return EntityEnvCfg(**{name: getattr(baseline, name) for name in baseline.__dataclass_fields__})
