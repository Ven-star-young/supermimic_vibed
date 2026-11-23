"""YogaBall tracking with the existing G1/rigid-object learning contract."""

from dataclasses import asdict, replace
from pathlib import Path

from ..sitting.task import SittingTask, build_task as build_sitting_task
from .assets import selected_asset


class YogaBallTask(SittingTask):
    task_name = "g1_yogaball"

    def checkpoint_metadata(self):
        metadata = super().checkpoint_metadata()
        metadata.update(
            dynamics_schema="yogaball-g1-ball-v1",
            state=metadata["state"].replace("chair", "ball"),
            actor_observation=metadata["actor_observation"].replace("chair", "ball"),
            ball_delta=metadata.pop("chair_delta").replace("chair", "ball"),
            object_entity=self.env_cfg.commands["motion"].object_entity_name,
            object_physics=asdict(self.env_cfg.commands["motion"].object_physical_cfg),
        )
        metadata["geometry"] = {**metadata["geometry"], "frame": "ball_body_meters"}
        return metadata


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    if motion_dir is not None:
        command = replace(cfg.commands["motion"], motion_dir=str(motion_dir))
        cfg = replace(cfg, commands={**cfg.commands, "motion": command})
    command = cfg.commands["motion"]
    selected_asset(command)
    return build_sitting_task(
        device=device, play=play, motion=motion, kin=kin, cfg=cfg,
        files=[Path(command.motion_dir)], task_cls=YogaBallTask,
    )
