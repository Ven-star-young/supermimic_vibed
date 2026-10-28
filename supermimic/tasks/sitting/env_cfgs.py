"""A fixed G1 + chair topology with bounded native collision workspaces."""
from dataclasses import dataclass, replace
from functools import partial
import math
import os

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from ...assets import DATA_DIR
from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from .assets import make_chair_spec
from .commands import SittingCommandCfg, geometry_id
from .events import randomize_object_physics


def _size_contact_buffers(spec, *, sim_cfg, previous=None, epa_horizon_capacity=64):
    """Measure startup contacts before Warp allocation; reserve 25% headroom."""
    if previous is not None:
        previous(spec)
    # MuJoCo-Warp 3.11 exposes no SimulationCfg option for this convex scratch
    # allocation. Raise it before graph capture and never shrink an existing value.
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
            print(f"[sitting contacts] {field}: {old} -> {required} (startup {count})", flush=True)


@dataclass(kw_only=True)
class SittingEnvCfg(ManagerBasedRlEnvCfg):
    epa_horizon_capacity: int = 64

    def __post_init__(self):
        if not 24 <= self.epa_horizon_capacity <= 256:
            raise ValueError("epa_horizon_capacity must be between 24 and 256")
        extra = set(self.scene.entities) - {"robot", "chair"}
        if extra:
            raise ValueError(f"Sitting models only robot and chair; unsupported scene entities: {sorted(extra)}")
        command = self.commands["motion"]
        entities = {"robot": self.scene.entities["robot"], "chair": EntityCfg(
            spec_fn=partial(make_chair_spec, command),
            init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 2.)))}
        previous = self.scene.spec_fn
        if isinstance(previous, partial) and previous.func is _size_contact_buffers:
            previous = previous.keywords["previous"]
        self.sim = replace(self.sim, nconmax=max(512, self.sim.nconmax or 0),
                           njmax=max(2000, self.sim.njmax or 0))
        self.scene = replace(self.scene, entities=entities, spec_fn=partial(
            _size_contact_buffers, sim_cfg=self.sim, previous=previous,
            epa_horizon_capacity=self.epa_horizon_capacity))
        self.events = {**self.events, "object_physics": EventTermCfg(func=randomize_object_physics, mode="reset")}
        self.observations = {**self.observations, "geometry_id": ObservationGroupCfg(
            terms={"geometry_id": ObservationTermCfg(func=geometry_id)})}


def make_env_cfg(play=False):
    baseline = make_tracking_env_cfg(play)
    baseline.scene.num_envs = 32 if play else 128
    command = SittingCommandCfg(**vars(baseline.commands["motion"]))
    command.motion_dir = os.environ.get("SUPERMIMIC_MOTION_DIR", str(DATA_DIR / "motions" / "sitting"))
    baseline.commands["motion"] = command
    baseline.scene.entities["robot"] = make_g1_robot_cfg(interaction_collisions=True)
    return SittingEnvCfg(**vars(baseline))
