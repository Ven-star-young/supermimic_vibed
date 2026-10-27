"""Task objectives using shared entity dynamics and object reference goals."""

from dataclasses import asdict, dataclass, replace
import re

import torch

from ...assets.g1.g1_constants import G1_JOINT_NAMES
from ...assets.g1.kinematics import G1Kinematics
from ...rl.entity_state import EntityDynamics
from ..base import TrackingTask
from ..g1_tracking.task import validate_env_cfg
from ..interaction.assets import select_motion_files
from .assets import entity_asset, object_names
from .mdp import EntityMDP
from .motion import EntityMotionLib


@dataclass
class EntityTask(TrackingTask):
    geometry_keys: tuple[str, ...] = ()

    def critic_goal_dim(self, frames):
        return self.goal_dim(frames) + 9 * self.dynamics.max_objects * frames

    def critic_goal(self, S, f, frames):
        return self.motion.object_goal(S, f, frames)

    def checkpoint_metadata(self):
        cmd = self.env_cfg.commands["motion"]
        metadata = dict(name=self.name, schema_version=5,
                    geometry_encoder="shared;updated_by_world_model_only",
                    dynamics_schema=self.dynamics.schema(),
                    dimensions=asdict(self.dimensions), critic_object_goal_per_frame=9,
                    geometry_keys=list(self.geometry_keys), object_names=list(object_names(cmd)),
                    extra_objects=[asdict(obj) for obj in cmd.extra_objects],
                    object_reward=asdict(cmd.object_reward_cfg),
                    target_slots=self.motion.target_mask.cpu().tolist())
        if cmd.collision_profile != "default":
            metadata["collision_profile"] = cmd.collision_profile
        return metadata


def build_task(motion_dir=None, *, device="cuda:0", play=False, motion=None, kin=None, cfg=None, files=None):
    if cfg is None:
        from .env_cfgs import make_env_cfg
        cfg = make_env_cfg(play)
    validate_env_cfg(cfg)
    command = cfg.commands["motion"]
    if motion_dir is not None:
        command = replace(command, motion_dir=motion_dir)
    kin = G1Kinematics(device) if kin is None else kin
    if motion is None:
        files = select_motion_files(command) if files is None else files
        motion = EntityMotionLib(files, kin, device, object_enabled=command.object_enabled,
                                 max_objects=command.max_objects, extra_objects=command.extra_objects)
    if motion.object_enabled != command.object_enabled or motion.max_objects != command.max_objects:
        raise ValueError("Motion references and scene object schema must agree")
    joint_positions = cfg.scene.entities["robot"].init_state.joint_pos
    defaults = []
    for name in G1_JOINT_NAMES:
        value = 0.
        for pattern, position in joint_positions.items():
            if re.fullmatch(pattern, name):
                value = position
        defaults.append(value)
    mdp = EntityMDP(kin, motion, torch.tensor(defaults, device=device), cfg=command.reward_cfg,
                   observation_cfg=command.observation_cfg, termination_cfg=command.termination_cfg,
                   control_dt=cfg.sim.mujoco.timestep * cfg.decimation,
                   max_objects=command.max_objects, object_reward_cfg=command.object_reward_cfg)
    names = object_names(command)
    return EntityTask("g1_entity_sitting" if command.object_enabled else "g1_entity_tracking", motion,
        mdp, EntityDynamics(command.max_objects), cfg, modeled_entities=("robot", *names),
        geometry_keys=tuple(entity_asset(command, i).key for i in range(len(names))))
