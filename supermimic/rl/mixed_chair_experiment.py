"""Balanced simultaneous chair training, retaining the original learning loop."""
from pathlib import Path

from motion_data.chair_split import scenes
from . import distributed
from .chair_experiment import ChairExperimentRunner, preflight, scene_cfg
from .runner import SuperMimicRunner
from .vecenv import ReplayEnvAdapter
from ..tasks.entities.mixed import MixedChairCommandCfg, MixedChairEnvCfg


def mixed_scene_cfg(manifest, root, cache_dir, *, num_envs, max_objects=8, seed=0,
                    assignment_seed=0, collision_profile="default"):
    root = Path(root)
    selected = scenes(manifest, "train")
    cfg = scene_cfg(root, selected[0], cache_dir=cache_dir, num_envs=num_envs,
                    max_objects=max_objects, seed=seed, collision_profile=collision_profile)
    protocol = dict(version=2, split_fingerprint=manifest["fingerprint"],
        train_shapes=[g["shape_key"] for g in manifest["train"]],
        test_shapes=[g["shape_key"] for g in manifest["test"]], num_envs=num_envs,
        world_size=distributed.world_size(), scene_distribution="fixed_balanced_per_environment",
        schedule_seed=seed, collision_profile=collision_profile)
    command = MixedChairCommandCfg(**vars(cfg.commands["motion"]), experiment=protocol,
        assignment_seed=assignment_seed,
        scene_motion_files=tuple(tuple(str(root / clip["path"]) for clip in scene["clips"]) for scene in selected))
    fields = vars(cfg) | dict(commands={"motion": command})
    return MixedChairEnvCfg(**fields)


class MixedChairExperimentRunner(ChairExperimentRunner):
    """Each rank owns all training geometries and one fixed variant per world."""

    def __init__(self, manifest, root, cache_dir, train_cfg, *, num_envs=128, max_objects=8,
                 schedule_seed=0, log_dir=None, device="cuda:0", collision_profile="default"):
        if train_cfg.get("world_model_file") or train_cfg.get("world_model_replays"):
            raise ValueError("Strict held-out experiments disallow unverified pretraining/replays")
        self.device = distributed.initialize(device)
        self.manifest, self.data_root, self.cache_dir = manifest, Path(root), Path(cache_dir)
        self.scenes = scenes(manifest, "train")
        self.visited_shapes, self.visit_log = set(), []
        self.collection_envs, self.capacity = num_envs, max_objects
        self.scene_configs = []
        self.env = None
        with distributed.local_phase(self.device, "mixed-chair scene initialization"):
            preflight(manifest, self.data_root, self.cache_dir, "train", collision_profile)
            cfg = mixed_scene_cfg(manifest, self.data_root, self.cache_dir, num_envs=num_envs,
                max_objects=max_objects, seed=schedule_seed,
                assignment_seed=schedule_seed + distributed.rank(), collision_profile=collision_profile)
            # Rank-local randomization differs; the task protocol seed is shared.
            cfg.seed = train_cfg.get("seed", 0) + distributed.rank()
            from mjlab.envs import ManagerBasedRlEnv
            from mjlab.rl import RslRlVecEnvWrapper
            native = ManagerBasedRlEnv(cfg, device=str(self.device))
            try:
                env = RslRlVecEnvWrapper(native)
                SuperMimicRunner.__init__(self, env, train_cfg, log_dir=log_dir, device=str(self.device))
            except BaseException:
                native.close()
                raise
        command = self.env.unwrapped.command_manager.get_term("motion")
        self.global_bank = command.geometry_bank
        self.env_scene = command.env_scene

    def make_replay_env(self):
        # The mixed command already uses the shared reference library's frames.
        return ReplayEnvAdapter(self.env, self.cfg.gamma)

    def prepare_collection(self):
        if self.visit_log:
            return
        counts = self.env_scene.bincount(minlength=len(self.scenes)).cpu().tolist()
        for index, count in enumerate(counts):
            if count:
                scene = self.scenes[index]
                self.visited_shapes.add(scene["shape_key"])
                self.visit_log.append(dict(visit=0, iteration=self.current_learning_iteration,
                    shape=scene["shape_key"], geometry=scene["geometry_key"], environments=count))
        print(f"[mixed-chair rank={distributed.rank()}] {len(self.scenes)} scenes in "
              f"{len(self.env_scene)} environments; fixed assignment, no periodic scene resets", flush=True)
