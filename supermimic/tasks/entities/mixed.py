"""One persistent native scene with a fixed chair/reference identity per world."""
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np
import torch

from ...mdp.commands import MotionCommand, ReferenceMotionCommand
from ...rl.entity_world_model import EntityGeometryBank
from .assets import entity_asset
from .commands import EntityCommand, EntityCommandCfg
from .env_cfgs import EntityEnvCfg
from .task import build_task
from .variants import make_chair_variant_cfg, normalize_variant_collision_slots


def balanced_assignment(num_envs, *, num_scenes, seed):
    if num_envs < num_scenes:
        raise ValueError(f"Mixed training needs at least {num_scenes} environments per rank to cover every chair")
    rng = np.random.default_rng(seed)
    # Randomize both remainder recipients and world order; counts differ by <=1.
    assignment = rng.permutation(num_scenes)[np.arange(num_envs) % num_scenes]
    return rng.permutation(assignment).tolist()


@dataclass(kw_only=True)
class MixedChairCommandCfg(EntityCommandCfg):
    scene_motion_files: tuple[tuple[str, ...], ...] = ()
    experiment: dict = field(default_factory=dict)
    assignment_seed: int = 0

    def build(self, env):
        return MixedChairCommand(self, env)

    def source_commands(self):
        fields = {name: getattr(self, name) for name in EntityCommandCfg.__dataclass_fields__}
        return tuple(EntityCommandCfg(**(fields | dict(motion_files=files, object_motion_file=files[0])))
                     for files in self.scene_motion_files)


class ChairReferenceSampler(ReferenceMotionCommand):
    """Existing adaptive RSI probabilities, normalized within each world's chair."""

    def __init__(self, cfg, motion, num_envs, device, max_return, env_scene, clip_scene):
        super().__init__(cfg, motion, num_envs, device, max_return)
        self.env_scene, self.clip_scene = env_scene, clip_scene
        self.bin_scene = clip_scene[motion.bin_clip]
        self.scene_bins = []
        for scene in range(len(cfg.scene_motion_files)):
            self.scene_bins.append((self.bin_scene == scene).nonzero().flatten())
        width = max(len(bins) for bins in self.scene_bins)
        self.bin_table = torch.zeros(len(self.scene_bins), width, dtype=torch.long, device=device)
        self.bin_valid = torch.zeros_like(self.bin_table, dtype=torch.bool)
        for index, bins in enumerate(self.scene_bins):
            self.bin_table[index, :len(bins)] = bins
            self.bin_valid[index, :len(bins)] = True
            self.bin_history[bins] = 1. / len(bins)

    def sample_for_envs(self, ids):
        m, cfg = self.motion, self.cfg
        selected_scene = self.env_scene[ids]
        if cfg.sampling_mode == "start" or cfg.play:
            # Clip0 per chair is deterministic; training's adaptive mode covers
            # every allowed clip and never samples another chair's reference.
            first_bins = self.bin_table[selected_scene, 0]
            frames = m.clip_start[m.bin_clip[first_bins]]
            return frames, m.frame_bin[frames]
        scores = -cfg.temperature * self.bin_returns[self.bin_table] / self.max_return
        soft = scores.masked_fill(~self.bin_valid, -torch.inf).softmax(-1)
        active = torch.zeros(len(self.scene_bins), device=self.device, dtype=torch.bool)
        active[selected_scene] = True
        update = self.bin_valid & active[:, None]
        indices = self.bin_table[update]
        self.bin_history[indices] = torch.lerp(self.bin_history[indices], soft[update], cfg.history_update)
        historical = self.bin_history[self.bin_table] * self.bin_valid
        historical = historical / historical.sum(-1, keepdim=True)
        probs = (cfg.uniform_weight * self.bin_valid / self.bin_valid.sum(-1, keepdim=True)
                 + cfg.current_weight * soft + cfg.history_weight * historical)
        offsets = torch.multinomial(probs[selected_scene], 1).squeeze(-1)
        bins = self.bin_table[selected_scene, offsets]
        start = m.bin_start[bins]
        last = m.clip_end[m.bin_clip[bins]]
        end = torch.minimum(start + m.bin_frames, last)
        frames = start + (torch.rand(len(ids), device=self.device) * (end - start)).long()
        return torch.minimum(frames, last - 1), bins

    def sample(self, count):
        raise RuntimeError("Mixed-chair reference sampling requires environment IDs")

    def reset(self, ids, frames=None):
        if frames is None:
            frames, _ = self.sample_for_envs(ids)
        if torch.any(self.clip_scene[self.motion.clip_id[frames]] != self.env_scene[ids]):
            raise ValueError("Reference does not belong to the environment's assigned chair")
        return super().reset(ids, frames)


class MixedChairCommand(EntityCommand):
    def build_task(self):
        from ...rl.chair_experiment import ChairExperimentTask
        if not self.cfg.scene_motion_files or not self.cfg.object_enabled or self.cfg.extra_objects:
            raise ValueError("Mixed-chair scenes require target chairs and no extra objects")
        files = [file for group in self.cfg.scene_motion_files for file in group]
        task = build_task(cfg=self._env.cfg, device=self.device, files=files)
        self.geometry_bank = EntityGeometryBank()
        self.scene_geometry_ids = []
        for command in self.cfg.source_commands():
            asset = entity_asset(command, 0)
            self.scene_geometry_ids.append(self.geometry_bank.add(asset.key, asset.points, asset.extent))
        task = ChairExperimentTask(**vars(task), experiment=self.cfg.experiment)
        task.name = "g1_entity_multi_chair"
        task.geometry_keys = tuple(self.geometry_bank.keys[1:])
        return task

    def __init__(self, cfg, env):
        MotionCommand.__init__(self, cfg, env)
        self.object_entity_names = ("chair",)
        self.env_scene = env.sim.world_to_variant["chair"].clone()
        self.clip_scene = torch.repeat_interleave(
            torch.arange(len(cfg.scene_motion_files), device=self.device),
            torch.tensor([len(files) for files in cfg.scene_motion_files], device=self.device))
        self.curriculum = ChairReferenceSampler(cfg, self.motion, self.num_envs, self.device,
            self.mdp.max_step_reward / (1 - cfg.gamma), self.env_scene, self.clip_scene)
        self.time_steps = self.curriculum.time_steps
        # Initial observations may be queried before the first managed reset.
        self.time_steps[:] = self.curriculum.sample_for_envs(torch.arange(self.num_envs, device=self.device))[0]
        context = self.dynamics.object_context(self.xi)
        context[..., 10], context[..., 11] = self.motion.valid_mask, self.motion.dynamic_mask
        context[:, 0, 12] = torch.tensor(self.scene_geometry_ids, device=self.device)[self.env_scene].float()
        self.dynamics.ground_context(self.xi)[..., 2] = 1.

    def set_reset_frames(self, env_ids, frames):
        frames = torch.as_tensor(frames, device=self.device, dtype=torch.long)
        if frames.shape != env_ids.shape or torch.any(frames < 0) or torch.any(frames >= self.motion.num_frames):
            raise ValueError("Reset frames must supply a valid frame for each environment")
        if torch.any(self.clip_scene[self.motion.clip_id[frames]] != self.env_scene[env_ids]):
            raise ValueError("Reference does not belong to the environment's assigned chair")
        super().set_reset_frames(env_ids, frames)

    def _resample_command(self, env_ids):
        frames, _ = self.curriculum.sample_for_envs(env_ids)
        queued = self._reset_frames[env_ids]
        frames = torch.where(queued >= 0, queued, frames)
        self.curriculum.reset(env_ids, frames)
        self._reset_frames[env_ids] = -1
        self.reset_reference(env_ids)
        self._last_step[env_ids] = self._env.common_step_counter
        self._forward_step = -1


def _mixed_scene_spec(spec, *, size_callback, robot_friction, ground_friction):
    normalize_variant_collision_slots(spec, robot_friction=robot_friction, ground_friction=ground_friction)
    size_callback(spec)


@dataclass(kw_only=True)
class MixedChairEnvCfg(EntityEnvCfg):
    def __post_init__(self):
        previous = self.scene.spec_fn
        if isinstance(previous, partial) and previous.func is _mixed_scene_spec:
            self.scene = replace(self.scene, spec_fn=previous.keywords["size_callback"])
        super().__post_init__()
        command = self.commands["motion"]
        if not isinstance(command, MixedChairCommandCfg) or not command.scene_motion_files:
            raise ValueError("MixedChairEnvCfg needs explicit scene/reference groups")
        if self.scene.num_envs < len(command.scene_motion_files):
            raise ValueError("Mixed training needs at least one environment per chair on each rank")
        physics = command.object_physical_cfg
        entity = make_chair_variant_cfg(command.source_commands(), assignment=partial(
            balanced_assignment, num_scenes=len(command.scene_motion_files), seed=command.assignment_seed))
        self.scene = replace(self.scene, entities={"robot": self.scene.entities["robot"], "chair": entity},
            spec_fn=partial(_mixed_scene_spec, size_callback=self.scene.spec_fn,
                robot_friction=physics.evaluation_robot_friction, ground_friction=physics.ground_friction))
