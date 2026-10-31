"""Independent side-opening task with shared door state, dynamics and models."""

from copy import deepcopy

from ..open_door.task import OpenDoorTask, build_task as build_door_task
from .assets import select_motion_files
from .contacts import SIDE_CONTACT_SCHEMA
from .geometry import SideDoorGeometry
from .mdp import SideDoorMDP


class OpenDoorSideTask(OpenDoorTask):
    task_name = "g1_open_door_side"
    geometry_cls = SideDoorGeometry
    mdp_cls = SideDoorMDP
    select_motion_files = staticmethod(select_motion_files)

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(),
                "contact_geometry": deepcopy(SIDE_CONTACT_SCHEMA),
                "contact_phase": "reference-proximity-v1"}


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    return build_door_task(motion_dir, device=device, play=play, motion=motion,
                           kin=kin, cfg=cfg, task_cls=OpenDoorSideTask)
