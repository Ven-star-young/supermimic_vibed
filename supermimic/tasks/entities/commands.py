"""Native command lifecycle for synchronized robot/entity references."""

from dataclasses import dataclass, field

import torch

from motion_data.collision import DEFAULT_CACHE_DIR
from ...mdp.commands import MotionCommand, MotionCommandCfg
from ...rotations import mat_to_quat
from ...rl.entity_world_model import EntityGeometryBank
from .assets import ObjectCfg, entity_asset, object_names
from .events import ObjectPhysicsCfg
from .mdp import ObjectRewardCfg


@dataclass(kw_only=True)
class EntityCommandCfg(MotionCommandCfg):
    max_objects: int = 8
    object_enabled: bool = True
    object_motion_file: str | None = None
    motion_files: tuple[str, ...] = ()
    collision_cache_dir: str = str(DEFAULT_CACHE_DIR)
    extra_objects: tuple[ObjectCfg, ...] = ()
    extra_objects_file: str | None = None
    object_physical_cfg: ObjectPhysicsCfg = field(default_factory=ObjectPhysicsCfg)
    object_reward_cfg: ObjectRewardCfg = field(default_factory=ObjectRewardCfg)

    def __post_init__(self):
        if not 1 <= self.max_objects <= 32:
            raise ValueError("max_objects must be in [1,32]")
        object_names(self)

    def build(self, env):
        return EntityCommand(self, env)


class EntityCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.object_entity_names = object_names(cfg)
        self.geometry_bank = EntityGeometryBank()
        context = self.dynamics.object_context(self.xi)
        context[..., 10] = self.motion.valid_mask
        context[..., 11] = self.motion.dynamic_mask
        self.dynamics.ground_context(self.xi)[..., 2] = 1.
        for index, _ in enumerate(self.object_entity_names):
            asset = entity_asset(cfg, index)
            context[..., index, 12] = self.geometry_bank.add(asset.key, asset.points, asset.extent)

    def build_task(self):
        from .task import build_task
        return build_task(self.cfg.motion_dir, device=self.device, play=self.cfg.play, cfg=self._env.cfg)

    def reset_reference(self, env_ids):
        super().reset_reference(env_ids)
        p, R, v, w = self.motion.object_ref_state(self.time_steps[env_ids])
        for index, name in enumerate(self.object_entity_names):
            # Fixed obstacles retain their configured body transform. Free
            # bodies receive reference pose/velocity from this exact same frame.
            if not bool(self.motion.dynamic_mask[index]):
                continue
            entity = self.scene[name]
            pose = torch.cat((p[:, index] + self.scene.env_origins[env_ids], mat_to_quat(R[:, index])), -1)
            entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            entity.write_root_link_velocity_to_sim(torch.cat((v[:, index], w[:, index]), -1), env_ids=env_ids)
