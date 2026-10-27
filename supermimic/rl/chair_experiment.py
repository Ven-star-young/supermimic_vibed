"""One persistent learner over a bounded, rotating set of physical chair scenes.

Only one native MuJoCo batch is resident. All training references and geometry
IDs have stable global indices; replay and optimizers survive scene changes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import gc
from pathlib import Path

import numpy as np
import torch

from motion_data import load_motion
from motion_data.chair_split import cache_report, scenes, verify_files
from motion_data.collision import load_collision_parts
from ..tasks.entities.assets import entity_asset
from ..tasks.entities.env_cfgs import make_env_cfg
from ..tasks.entities.task import EntityTask, build_task
from ..tasks.interaction.assets import select_motion_files
from .entity_world_model import EntityGeometryBank
from .runner import SuperMimicRunner
from .vecenv import ReplayEnvAdapter


def scene_cfg(root, scene, *, cache_dir, num_envs, max_objects=8, play=False, seed=0):
    cfg = make_env_cfg(play=play, object_enabled=True, max_objects=max_objects)
    # Experiment definitions never inherit unrelated shell-provided extra objects.
    command = replace(cfg.commands["motion"], motion_dir=str(root),
                      motion_files=tuple(str(root / c["path"]) for c in scene["clips"]),
                      object_motion_file=str(root / scene["clips"][0]["path"]),
                      extra_objects=(), extra_objects_file=None, collision_cache_dir=str(cache_dir))
    cfg = replace(cfg, commands={"motion": command})
    cfg.scene.num_envs, cfg.auto_reset, cfg.seed = num_envs, False, seed
    return cfg


def preflight(manifest, root, cache_dir, partition):
    verify_files(manifest, root, partition)
    report = cache_report(manifest, cache_dir)[partition]
    if report["missing"]:
        raise FileNotFoundError(
            f"{partition}: {report['missing_collision_scenes']} collision assets missing; "
            "run tools/prepare_collisions.py --split-manifest <split.json> "
            f"--partition {partition} --cache-dir {cache_dir}. First missing: {report['missing'][0]}"
        )
    for scene in scenes(manifest, partition):
        load_collision_parts(load_motion(root / scene["clips"][0]["path"]), cache_dir)


@dataclass
class ChairExperimentTask(EntityTask):
    experiment: dict | None = None

    def checkpoint_metadata(self):
        metadata = super().checkpoint_metadata()
        metadata["chair_experiment"] = self.experiment
        return metadata


class GlobalFrameReplay(ReplayEnvAdapter):
    def __init__(self, env, gamma, offset):
        super().__init__(env, gamma)
        self.offset = offset

    def step(self, action):
        transition = super().step(action)
        transition["f"] += self.offset
        return transition


class ChairExperimentRunner(SuperMimicRunner):
    def __init__(self, manifest, root, cache_dir, train_cfg, *, num_envs=128, max_objects=8,
                 iterations_per_chair=50, schedule_seed=0, log_dir=None, device="cuda:0"):
        if iterations_per_chair < 1 or num_envs < 1:
            raise ValueError("iterations_per_chair and num_envs must be positive")
        if train_cfg.get("world_model_file") or train_cfg.get("world_model_replays"):
            raise ValueError("Strict held-out experiments disallow unverified pretraining/replays; use same-split resume")
        self.manifest, self.data_root, self.cache_dir = manifest, Path(root), Path(cache_dir)
        self.scenes = scenes(manifest, "train")
        self.iterations_per_chair, self.schedule_seed = iterations_per_chair, schedule_seed
        self.collection_envs, self.capacity = num_envs, max_objects
        self.device = torch.device(device)
        self.active_visit = -1
        self.visited_shapes = set()
        self.visit_log = []
        preflight(manifest, self.data_root, self.cache_dir, "train")
        bank = EntityGeometryBank()
        self.scene_configs = []
        for scene in self.scenes:
            cfg = scene_cfg(self.data_root, scene, cache_dir=cache_dir, num_envs=num_envs,
                            max_objects=max_objects, seed=train_cfg.get("seed", 0))
            asset = entity_asset(cfg.commands["motion"], 0)
            if asset.key != scene["geometry_key"]:
                raise ValueError("Training asset identity differs from split")
            bank.add(asset.key, asset.points, asset.extent)
            self.scene_configs.append(cfg)
        files = [self.data_root / clip["path"] for scene in self.scenes for clip in scene["clips"]]
        task = build_task(cfg=self.scene_configs[0], device=device, files=files)
        protocol = dict(version=1, split_fingerprint=manifest["fingerprint"],
                        train_shapes=[g["shape_key"] for g in manifest["train"]],
                        test_shapes=[g["shape_key"] for g in manifest["test"]],
                        iterations_per_chair=iterations_per_chair, schedule_seed=schedule_seed,
                        num_envs=num_envs)
        global_task = ChairExperimentTask(**vars(task), experiment=protocol)
        global_task.name = "g1_entity_multi_chair"
        global_task.geometry_keys = tuple(bank.keys[1:])
        # Protocol details identify reference content, not machine-specific paths.
        self.offsets, clip_index = [], 0
        for scene in self.scenes:
            self.offsets.append(int(task.motion.clip_start[clip_index]))
            clip_index += len(scene["clips"])
        self.global_bank = bank
        self.active_scene = self.scene_for_visit(0)
        env = self.open_scene(self.active_scene, 0)
        try:
            super().__init__(env, train_cfg, log_dir=log_dir, device=device)
            self.task, self.motion, self.mdp, self.dynamics = (
                global_task, global_task.motion, global_task.mdp, global_task.dynamics)
            self.dims = global_task.dimensions
        except BaseException:
            env.close()
            raise
        self.active_visit = 0

    def scene_for_visit(self, visit):
        epoch, index = divmod(visit, len(self.scenes))
        return int(np.random.default_rng(self.schedule_seed + epoch).permutation(len(self.scenes))[index])

    def open_scene(self, index, visit):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        cfg = deepcopy(self.scene_configs[index])
        cfg.seed += visit
        native = ManagerBasedRlEnv(cfg, device=str(self.device))
        try:
            command = native.command_manager.get_term("motion")
            actual = [Path(p).name for p in select_motion_files(command.cfg)]
            expected = [c["path"] for c in self.scenes[index]["clips"]]
            if actual != expected:
                raise ValueError("Native scene reference order differs from global Replay frame mapping")
            mapping = self.global_bank.merge(command.geometry_bank.state_dict()).to(self.device)
            context = command.dynamics.object_context(command.xi)
            context[..., 12] = mapping[context[..., 12].long()].float()
            command.geometry_bank = self.global_bank
            return RslRlVecEnvWrapper(native)
        except BaseException:
            native.close()
            raise

    def make_replay_env(self):
        return GlobalFrameReplay(self.env, self.cfg.gamma, self.offsets[self.active_scene])

    def prepare_collection(self):
        visit = self.current_learning_iteration // self.iterations_per_chair
        index = self.scene_for_visit(visit)
        if visit != self.active_visit:
            # Stop all old trajectories at this scene boundary. Their final
            # transitions remain valid but windows/history cannot cross it.
            if self.buffer is not None and self.buffer.size:
                self.buffer.data["done"][(self.buffer.ptr - 1) % self.buffer.T] = True
            self.env.close()
            self.replay_env = None
            self.env = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.env = self.open_scene(index, visit)
            self.active_scene, self.active_visit = index, visit
            self.replay_env = self.make_replay_env()
        shape = self.scenes[index]["shape_key"]
        self.visited_shapes.add(shape)
        if not self.visit_log or self.visit_log[-1]["visit"] != visit:
            self.visit_log.append(dict(visit=visit, iteration=self.current_learning_iteration,
                                      shape=shape, geometry=self.scenes[index]["geometry_key"]))
            print(f"[multi-chair] visit={visit} scene={index + 1}/{len(self.scenes)} "
                  f"covered={len(self.visited_shapes)}/{len(self.manifest['train'])}", flush=True)

    def save(self, path, infos=None):
        details = dict(visited_train_shapes=sorted(self.visited_shapes), scene_visits=self.visit_log,
                       split_fingerprint=self.manifest["fingerprint"])
        super().save(path, infos={**(infos or {}), "chair_experiment": details})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        details = (infos or {}).get("chair_experiment", {})
        visited = set(details.get("visited_train_shapes", []))
        if not visited <= {g["shape_key"] for g in self.manifest["train"]}:
            raise ValueError("Checkpoint contains geometry exposure outside this training split")
        self.visited_shapes = visited
        self.visit_log = list(details.get("scene_visits", []))
        return infos

    def close(self):
        if self.env is not None:
            self.env.close()
