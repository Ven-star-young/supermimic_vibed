"""Native tracking configuration plus a dynamic object and one extra reward."""

from dataclasses import dataclass, replace
from functools import partial
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg

from ...assets import DATA_DIR
from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from .assets import make_chair_spec
from .commands import SittingCommandCfg
from .events import randomize_object_physics


@dataclass(kw_only=True)
class SittingEnvCfg(ManagerBasedRlEnvCfg):
    def __post_init__(self):
        # Tyro reconstructs command configs. Rebind the lazy asset factory to
        # that new command rather than retaining a closure over CLI defaults.
        command = self.commands["motion"]
        entities = dict(self.scene.entities)
        if command.object_enabled:
            chair = entities.get("chair", EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 2.))))
            entities["chair"] = replace(chair, spec_fn=partial(make_chair_spec, command))
        else:
            entities.pop("chair", None)
        self.scene = replace(self.scene, entities=entities)
        self.events = dict(self.events)
        if command.object_enabled:
            self.events["object_physics"] = EventTermCfg(func=randomize_object_physics, mode="reset")
            self.sim = replace(self.sim, nconmax=max(512, self.sim.nconmax), njmax=max(2000, self.sim.njmax))
        else:
            self.events.pop("object_physics", None)


def make_env_cfg(play=False, *, object_enabled=True):
    baseline = make_tracking_env_cfg(play)
    command = SittingCommandCfg(
        **{name: getattr(baseline.commands["motion"], name)
           for name in baseline.commands["motion"].__dataclass_fields__},
        object_enabled=object_enabled,
    )
    command.motion_dir = os.environ.get(
        "SUPERMIMIC_MOTION_DIR", str(DATA_DIR / "motions" / ("sitting" if object_enabled else "lafan1")))
    baseline.commands["motion"] = command
    baseline.scene.entities["robot"] = make_g1_robot_cfg(interaction_collisions=True)
    # Avoid asdict: it would recursively convert manager config dataclasses to
    # dictionaries and lose the native mjlab interfaces.
    return SittingEnvCfg(**{name: getattr(baseline, name) for name in baseline.__dataclass_fields__})
