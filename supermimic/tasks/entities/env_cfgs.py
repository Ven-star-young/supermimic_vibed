"""Lazy native scenes sharing one variable-entity physics contract."""

from dataclasses import dataclass, replace
from functools import partial
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg

from ...assets import DATA_DIR
from ...assets.g1.g1_constants import make_g1_robot_cfg
from ..g1_tracking.env_cfgs import make_env_cfg as make_tracking_env_cfg
from .assets import load_object_configs, make_object_spec, object_names
from .commands import EntityCommandCfg
from .events import randomize_entity_physics


@dataclass(kw_only=True)
class EntityEnvCfg(ManagerBasedRlEnvCfg):
    def __post_init__(self):
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
        self.scene = replace(self.scene, entities=entities)
        self.events = dict(self.events)
        # Even empty scenes record robot-ground effective pair properties.
        self.events["entity_physics"] = EventTermCfg(func=randomize_entity_physics, mode="reset")
        if names:
            # Capacity depends on actual entities, never the padded token count.
            self.sim = replace(self.sim, nconmax=max(512 * len(names), self.sim.nconmax),
                               njmax=max(2000 * len(names), self.sim.njmax))


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
