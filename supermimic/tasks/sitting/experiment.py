"""One Sitting learner with balanced, fixed chair variants in every GPU batch."""
from dataclasses import replace
from pathlib import Path

from motion_data import load_motion
from motion_data.chair_split import scenes, verify_files
from motion_data.collision import load_collision_parts
from ...rl import distributed
from ...rl.runner import SuperMimicRunner
from .env_cfgs import make_env_cfg
from .mixed import MixedChairCommandCfg, MixedChairEnvCfg


def scene_cfg(root, scene, *, cache_dir, num_envs, play=False, seed=0, collision_profile="default"):
    root = Path(root)
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    cfg = make_env_cfg(play)
    command = replace(cfg.commands["motion"], motion_dir=str(root),
        motion_files=tuple(str(root / clip["path"]) for clip in scene["clips"]),
        object_motion_file=str(root / scene["clips"][0]["path"]),
        collision_cache_dir=str(cache_dir), collision_profile=collision_profile)
    return replace(cfg, commands={"motion": command},
                   scene=replace(cfg.scene, num_envs=num_envs), auto_reset=False, seed=seed)


def preflight(manifest, root, cache_dir, partition="train", collision_profile="default"):
    root = Path(root)
    verify_files(manifest, root, partition)
    for scene in scenes(manifest, partition):
        load_collision_parts(load_motion(Path(root) / scene["clips"][0]["path"]),
                             cache_dir, profile=collision_profile)


def mixed_scene_cfg(manifest, root, cache_dir, *, num_envs, seed=0,
                    assignment_seed=0, collision_profile="default"):
    root = Path(root)
    selected = scenes(manifest, "train")
    if not selected:
        raise ValueError("Training split contains no chair scenes")
    cfg = scene_cfg(root, selected[0], cache_dir=cache_dir, num_envs=num_envs,
                    seed=seed, collision_profile=collision_profile)
    protocol = dict(version=1, split_fingerprint=manifest["fingerprint"],
        train_shapes=[g["shape_key"] for g in manifest["train"]],
        test_shapes=[g["shape_key"] for g in manifest["test"]], num_envs=num_envs,
        world_size=distributed.world_size(), scene_distribution="fixed_balanced_per_environment",
        schedule_seed=seed, collision_profile=collision_profile)
    command = MixedChairCommandCfg(**vars(cfg.commands["motion"]), experiment=protocol,
        assignment_seed=assignment_seed,
        scene_motion_files=tuple(tuple(str(root / clip["path"]) for clip in scene["clips"])
                                 for scene in selected))
    return MixedChairEnvCfg(**(vars(cfg) | dict(commands={"motion": command})))


class SittingExperimentRunner(SuperMimicRunner):
    """Task-specific experiment provenance around the shared optimization loop."""
    def __init__(self, manifest, root, cache_dir, train_cfg, *, num_envs=128,
                 schedule_seed=0, log_dir=None, device="cuda:0", collision_profile="default"):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        self.manifest, self.data_root, self.cache_dir = manifest, Path(root), Path(cache_dir)
        self.scenes = scenes(manifest, "train")
        self.device = distributed.initialize(device)
        self.env = None
        # Only local work belongs in this scope: the parent runner may perform
        # its own broadcast collectives after every rank has created its scene.
        native = None
        try:
            with distributed.local_phase(self.device, "sitting scene initialization"):
                preflight(manifest, self.data_root, cache_dir, "train", collision_profile)
                cfg = mixed_scene_cfg(manifest, root, cache_dir, num_envs=num_envs,
                    seed=schedule_seed, assignment_seed=schedule_seed + distributed.rank(),
                    collision_profile=collision_profile)
                cfg.seed = train_cfg.get("seed", 0) + distributed.rank()
                native = ManagerBasedRlEnv(cfg, device=str(self.device))
            super().__init__(RslRlVecEnvWrapper(native), train_cfg, log_dir=log_dir, device=str(self.device))
        except BaseException:
            if native is not None:
                native.close()
            raise
        command = native.command_manager.get_term("motion")
        self.protocol = dict(command.cfg.experiment)
        self.env_scene = command.env_scene
        self.scene_counts = self.env_scene.bincount(minlength=len(self.scenes)).cpu().tolist()
        print(f"[sitting rank={distributed.rank()}] {len(self.scenes)} chairs, "
              f"{num_envs} environments, {min(self.scene_counts)}–{max(self.scene_counts)} per chair", flush=True)

    def save(self, path, infos=None):
        counts = distributed.gather_objects(self.scene_counts)
        # Initialization is not training exposure. A saved trained checkpoint has
        # collected from every environment in this fixed, balanced assignment.
        visited = sorted({scene["shape_key"] for scene, count in zip(self.scenes, self.scene_counts) if count})
        if not self.sim_steps:
            visited = []
        details = self.protocol | dict(visited_train_shapes=visited, scene_assignments_by_rank=counts)
        super().save(path, infos={**(infos or {}), "sitting_experiment": details})

    def validate_checkpoint_extras(self, checkpoint, *, actor_only=False):
        if actor_only:
            return
        protocol = (checkpoint.get("infos") or {}).get("sitting_experiment", {})
        for key in ("split_fingerprint", "collision_profile", "scene_distribution"):
            if protocol.get(key) != self.protocol[key]:
                raise ValueError(f"Checkpoint sitting experiment mismatch: {key}")
        if not set(protocol.get("visited_train_shapes", [])) <= set(self.protocol["train_shapes"]):
            raise ValueError("Checkpoint includes a chair outside the training split")

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
