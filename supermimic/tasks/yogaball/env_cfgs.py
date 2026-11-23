"""One G1 and one free rigid yoga ball, using the migrated PA-HOI geometry."""

from dataclasses import dataclass, replace
from functools import partial
import os

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from ...assets import DATA_DIR
from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from ..sitting.commands import geometry_id
from ..sitting.env_cfgs import _size_contact_buffers
from ..sitting.events import randomize_object_physics
from .assets import make_ball_spec
from .commands import YogaBallCommandCfg


@dataclass(kw_only=True)
class YogaBallEnvCfg(ManagerBasedRlEnvCfg):
    def __post_init__(self):
        command = self.commands["motion"]
        if command.object_entity_name != "ball":
            raise ValueError("YogaBall requires object_entity_name='ball'")
        extra = set(self.scene.entities) - {"robot", "ball"}
        if extra:
            raise ValueError(f"YogaBall models only robot and ball; unsupported entities: {sorted(extra)}")
        entities = {"robot": self.scene.entities["robot"], "ball": EntityCfg(
            spec_fn=partial(make_ball_spec, command),
            init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 2.)))}
        previous = self.scene.spec_fn
        if isinstance(previous, partial) and previous.func is _size_contact_buffers:
            previous = previous.keywords["previous"]
        self.sim = replace(self.sim, nconmax=max(512, self.sim.nconmax or 0),
                           njmax=max(2000, self.sim.njmax or 0))
        self.scene = replace(self.scene, entities=entities, spec_fn=partial(
            _size_contact_buffers, sim_cfg=self.sim, previous=previous, epa_horizon_capacity=64))
        self.events = {**self.events, "object_physics": EventTermCfg(func=randomize_object_physics, mode="reset")}
        self.observations = {**self.observations, "geometry_id": ObservationGroupCfg(
            terms={"geometry_id": ObservationTermCfg(func=geometry_id)})}


def make_env_cfg(play=False):
    baseline = make_tracking_env_cfg(play)
    baseline.scene.num_envs = 32 if play else 128
    command = YogaBallCommandCfg(**vars(baseline.commands["motion"]))
    command.motion_dir = os.environ.get("SUPERMIMIC_MOTION_DIR", str(
        DATA_DIR / "motions/hold/1_273__s0p85.npz"))
    baseline.commands["motion"] = command
    baseline.scene.entities["robot"] = make_g1_robot_cfg(interaction_collisions=True)
    return YogaBallEnvCfg(**vars(baseline))
