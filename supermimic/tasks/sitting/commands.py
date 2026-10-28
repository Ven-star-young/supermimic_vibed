"""Synchronized G1/chair references and scene geometry metadata."""
from dataclasses import dataclass, field

import torch

from motion_data.collision import DEFAULT_CACHE_DIR
from ...mdp.commands import MotionCommand, MotionCommandCfg
from ...rotations import mat_to_quat
from ...rl.geometry import GeometryBank
from .assets import selected_asset
from .events import ObjectPhysicsCfg
from .mdp import ObjectRewardCfg


def geometry_id(env, command_name="motion"):
    """Lookup metadata, deliberately separate from numerical actor observations."""
    return env.command_manager.get_term(command_name).xi[:, 45:46]


@dataclass(kw_only=True)
class SittingCommandCfg(MotionCommandCfg):
    motion_files: tuple[str, ...] = ()
    object_motion_file: str | None = None
    collision_cache_dir: str = str(DEFAULT_CACHE_DIR)
    collision_profile: str = "default"
    object_physical_cfg: ObjectPhysicsCfg = field(default_factory=ObjectPhysicsCfg)
    object_reward_cfg: ObjectRewardCfg = field(default_factory=ObjectRewardCfg)

    def build(self, env):
        return SittingCommand(self, env)


class SittingCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry_bank = GeometryBank()
        asset = selected_asset(cfg)
        self.xi[:, 45] = self.geometry_bank.add(asset.key, asset.points, asset.extent)
        self.task.geometry_keys = tuple(self.geometry_bank.keys)

    def build_task(self):
        from .task import build_task
        return build_task(self.cfg.motion_dir, device=self.device, play=self.cfg.play, cfg=self._env.cfg)

    def reset_reference(self, env_ids):
        super().reset_reference(env_ids)
        position, rotation, velocity, angular_velocity = self.motion.object_ref_state(self.time_steps[env_ids])
        chair = self.scene["chair"]
        pose = torch.cat((position + self.scene.env_origins[env_ids], mat_to_quat(rotation)), -1)
        chair.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        chair.write_root_link_velocity_to_sim(torch.cat((velocity, angular_velocity), -1), env_ids=env_ids)
