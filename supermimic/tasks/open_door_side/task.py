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

    def validate_checkpoint(self, checkpoint):
        metadata = checkpoint.get("task")
        if (isinstance(metadata, dict) and metadata.get("name") == self.task_name
                and metadata.get("schema_version") == 2
                and isinstance(metadata.get("door_reward"), dict)
                and "illegal_margin" not in metadata["door_reward"]):
            # Older side checkpoints penalized penetration only. Never silently
            # apply the new clearance margin to their evaluation or full resume.
            checkpoint = {**checkpoint, "task": {**metadata, "door_reward": {
                **metadata["door_reward"], "illegal_margin": 0.0}}}
        super().validate_checkpoint(checkpoint)

    def validate_finetune_checkpoint(self, checkpoint):
        """Reuse side Actor/WM with new rewards; all other task contracts stay strict."""
        self.validate_world_model_checkpoint(checkpoint)
        metadata, current = checkpoint["task"], self.checkpoint_metadata()
        if any(not isinstance(metadata.get(key), dict) for key in ("door_reward", "robot_reward")):
            raise ValueError("Invalid side finetuning checkpoint reward metadata")
        self.validate_checkpoint({**checkpoint, "task": {
            **metadata, "door_reward": current["door_reward"], "robot_reward": current["robot_reward"]}})


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    return build_door_task(motion_dir, device=device, play=play, motion=motion,
                           kin=kin, cfg=cfg, task_cls=OpenDoorSideTask)
