"""Native G1 + passive hinge scene, reusing tracking managers and control."""

from dataclasses import dataclass, replace
from functools import partial
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg

from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from .assets import DEFAULT_MOTION_DIR, configure_scene, make_door_spec
from .commands import DoorCommandCfg
from .events import randomize_door_physics


@dataclass(kw_only=True)
class DoorEnvCfg(ManagerBasedRlEnvCfg):
    def __post_init__(self):
        extra = set(self.scene.entities) - {"robot", "door"}
        if extra:
            raise ValueError(f"OpenDoor models only robot and door; unsupported entities: {sorted(extra)}")
        # Fixed-base entities have a mocap parent. The XML already positions its
        # child, so leave this parent at the origin (no second translation).
        door = EntityCfg(spec_fn=partial(make_door_spec, self.commands["motion"]),
                         init_state=EntityCfg.InitialStateCfg(
                             pos=(0., 0., 0.), joint_pos={"door_hinge": 0.},
                             joint_vel={"door_hinge": 0.}))
        self.scene = replace(self.scene,
                             entities={"robot": self.scene.entities["robot"], "door": door},
                             spec_fn=configure_scene)
        self.sim = replace(self.sim, nconmax=max(512, self.sim.nconmax or 0),
                           njmax=max(2000, self.sim.njmax or 0))
        self.events = {**self.events, "door_physics": EventTermCfg(func=randomize_door_physics, mode="reset")}


def make_env_cfg(play=False):
    baseline = make_tracking_env_cfg(play)
    baseline.scene.num_envs = 32 if play else 128
    command = DoorCommandCfg(**vars(baseline.commands["motion"]))
    command.motion_dir = os.environ.get("SUPERMIMIC_MOTION_DIR", str(DEFAULT_MOTION_DIR))
    baseline.commands["motion"] = command
    baseline.scene.entities["robot"] = make_g1_robot_cfg(interaction_collisions=True)
    return DoorEnvCfg(**vars(baseline))
