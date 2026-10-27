"""Chair-disjoint multi-scene training and frozen-policy zero-shot evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from motion_data.chair_split import cache_report, file_digest, load_split, scenes
from motion_data import load_motion
from motion_data.collision import DEFAULT_CACHE_DIR, COLLISION_PROFILES
from supermimic.rl.chair_experiment import ChairExperimentRunner, preflight, profile_cache_report, scene_cfg
from supermimic.rl.mixed_chair_experiment import MixedChairExperimentRunner
from supermimic.tasks.interaction.assets import geometry_key
from supermimic.rl.runner import SuperMimicRunner
from supermimic.tasks.entities.rl_cfg import make_runner_cfg
from supermimic.rl import distributed


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_heldout_actor(runner, checkpoint, manifest, root=None):
    """Explicit transfer of frozen policy semantics, not full-scene resume.

    New geometry is encoded by the trained encoder using the evaluation bank's
    own IDs. Training asset IDs/points are never installed into the test scene.
    """
    metadata = checkpoint.get("task", {})
    protocol = metadata.get("chair_experiment", {})
    if (metadata.get("schema_version") != 5 or metadata.get("name") != "g1_entity_multi_chair"
            or protocol.get("split_fingerprint") != manifest["fingerprint"]):
        raise ValueError("Zero-shot evaluation requires a matching multi-chair experiment checkpoint")
    current = runner.task.checkpoint_metadata()
    for key in ("dynamics_schema", "dimensions", "geometry_encoder", "object_reward", "target_slots"):
        if metadata.get(key) != current.get(key):
            raise ValueError(f"Incompatible held-out policy contract: {key}")
    train_keys = {s["geometry_key"] for s in scenes(manifest, "train")}
    profile = protocol.get("collision_profile", "default")
    if profile != "default":
        if root is None:
            raise ValueError("Compact collision provenance requires the training motion root")
        train_keys = {geometry_key(load_motion(root / s["clips"][0]["path"]), profile)
                      for s in scenes(manifest, "train")}
    if set(checkpoint["entity_geometry"]["keys"][1:]) != train_keys:
        raise ValueError("Checkpoint geometry bank is not confined to the declared training partition")
    for key, value in checkpoint["pi"].items():
        if key.startswith("geometry_encoder.") and any(
            key not in checkpoint[name] or not torch.equal(value, checkpoint[name][key])
            for name in ("wm", "critic")
        ):
            raise ValueError("Checkpoint has inconsistent shared geometry encoder weights")
    for key in ("history_length", "goal_frames"):
        if checkpoint["policy_config"][key] != runner.policy_config[key]:
            raise ValueError(f"Incompatible evaluation {key}")
    runner.pi.load_state_dict(checkpoint["pi"], strict=True)
    runner.pi.eval().requires_grad_(False)


def evaluate_split(manifest, root, checkpoint_file, *, cache_dir, device="cuda:0",
                   allow_incomplete_training=False, output=None, seed=0):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .evaluate import evaluate

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("task", {}).get("chair_experiment", {})
    if protocol.get("split_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Checkpoint and evaluation split fingerprints differ")
    exposure = set((checkpoint.get("infos") or {}).get("chair_experiment", {}).get("visited_train_shapes", []))
    training_shapes = {g["shape_key"] for g in manifest["train"]}
    if not exposure <= training_shapes:
        raise ValueError("Checkpoint exposure includes a non-training chair")
    if exposure != training_shapes and not allow_incomplete_training:
        raise ValueError(f"Checkpoint visited only {len(exposure)}/{len(training_shapes)} training chairs; "
                         "use --allow-incomplete-training only for an explicitly labeled diagnostic")
    collision_profile = protocol.get("collision_profile", "default")
    preflight(manifest, root, cache_dir, "test", collision_profile)
    capacity = checkpoint["task"]["dynamics_schema"]["max_objects"]
    records = []
    # Evaluation parameters are fixed, not sampled, and no training runner or
    # model optimizer is initialized. Each chair starts at every clip's first frame.
    for scene in scenes(manifest, "test"):
        cfg = scene_cfg(root, scene, cache_dir=cache_dir, num_envs=len(scene["clips"]),
                        max_objects=capacity, play=True, seed=seed, collision_profile=collision_profile)
        command_cfg = cfg.commands["motion"]
        command_cfg.history_length = checkpoint["policy_config"]["history_length"]
        command_cfg.goal_frames = checkpoint["policy_config"]["goal_frames"]
        reward = checkpoint["task"]["object_reward"]
        from supermimic.tasks.interaction.mdp import ObjectRewardCfg
        command_cfg.object_reward_cfg = ObjectRewardCfg(**reward)
        native = ManagerBasedRlEnv(cfg, device=device)
        try:
            env = RslRlVecEnvWrapper(native)
            train_cfg = checkpoint["train_cfg"] | {"amp": "none", "world_model_file": None,
                                                    "world_model_replays": (), "no_compile": True}
            runner = SuperMimicRunner(env, train_cfg, device=device)
            load_heldout_actor(runner, checkpoint, manifest, root)
            before = policy_digest(runner.pi)
            result = evaluate(env, runner)
            if before != policy_digest(runner.pi):
                raise RuntimeError("Policy parameters or normalization changed during zero-shot evaluation")
            if runner.wm is not None or runner.critic is not None or runner.buffer is not None:
                raise RuntimeError("Evaluation unexpectedly initialized training state")
            records.append(dict(shape_key=scene["shape_key"], geometry_key=scene["geometry_key"],
                                clips=len(scene["clips"]), **result))
            print(f"[zero-shot] {len(records)}/{len(scenes(manifest, 'test'))} "
                  f"return={result['norm_return']:.4f} failed={result['n_failed']}/{len(scene['clips'])}", flush=True)
            del runner, env
        finally:
            native.close()
        del native
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    per_chair = []
    for group in manifest["test"]:
        variants = [r for r in records if r["shape_key"] == group["shape_key"]]
        clips = sum(r["clips"] for r in variants)
        failed = sum(r["n_failed"] for r in variants)
        per_chair.append(dict(shape_key=group["shape_key"], clips=clips,
                              norm_return=sum(r["norm_return"] * r["clips"] for r in variants) / clips,
                              clip_completion_rate=1 - failed / clips))
    report = dict(protocol="held_out_geometry_frozen_policy", split_fingerprint=manifest["fingerprint"],
                  checkpoint=str(Path(checkpoint_file).resolve()), checkpoint_sha256=file_digest(checkpoint_file),
                  evaluation_seed=seed, evaluated_chairs=len(per_chair), collision_profile=collision_profile,
                  visited_training_chairs=len(exposure), total_training_chairs=len(training_shapes),
                  incomplete_training=exposure != training_shapes,
                  unseen_geometry=True, reference_conditioned=True,
                  macro_chair_return=sum(r["norm_return"] for r in per_chair) / len(per_chair),
                  macro_chair_completion=sum(r["clip_completion_rate"] for r in per_chair) / len(per_chair),
                  clip_weighted_return=sum(r["norm_return"] * r["clips"] for r in per_chair) / sum(r["clips"] for r in per_chair),
                  metric_note="Completion means reaching reference end without existing robot termination; not a dedicated sitting-success detector.",
                  per_chair=per_chair, per_scene=records)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "evaluate"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/sitting_80_20_seed0.json"))
    parser.add_argument("--data-root", type=Path, help="Override relocatable motion root")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--scene-mode", choices=("rotate", "mixed"), default="rotate",
                        help="mixed fixes one chair per world, balanced across all training chairs")
    parser.add_argument("--collision-profile", choices=COLLISION_PROFILES, default="default",
                        help="Training/check collision approximation; evaluation uses checkpoint profile")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--iterations-per-chair", type=int, default=50)
    parser.add_argument("--iterations", type=int, help="Additional learning iterations; default agent.max_iterations")
    parser.add_argument("--agent-json", type=Path, help="JSON overrides of SuperMimicRunnerCfg fields")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Full checkpoint from this exact experiment")
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/zero_shot_chairs.json"))
    parser.add_argument("--allow-incomplete-training", action="store_true")
    args = parser.parse_args()
    if distributed.world_size() > 1 and args.mode != "train":
        parser.error("Use a single process for check/evaluate; torchrun is for distributed training")
    if args.mode == "train":
        args.device = str(distributed.initialize(args.device))
    manifest, root = load_split(args.split, args.data_root)
    if args.mode == "check":
        print(json.dumps(profile_cache_report(manifest, root, args.cache_dir, args.collision_profile), indent=2))
        return
    if args.mode == "evaluate":
        if args.checkpoint_file is None:
            parser.error("evaluate requires --checkpoint-file")
        report = evaluate_split(manifest, root, args.checkpoint_file, cache_dir=args.cache_dir,
                                device=args.device, output=args.output, seed=args.seed,
                                allow_incomplete_training=args.allow_incomplete_training)
        print(json.dumps({k: v for k, v in report.items() if k not in ("per_scene", "per_chair")}, indent=2))
        return
    agent = asdict(make_runner_cfg())
    if args.agent_json:
        overrides = json.loads(args.agent_json.read_text())
        if not isinstance(overrides, dict) or overrides.keys() - agent.keys():
            parser.error("agent-json contains unknown training fields")
        agent.update(overrides)
    agent["seed"] = args.seed
    iterations = args.iterations if args.iterations is not None else agent["max_iterations"]
    if iterations < 1:
        parser.error("iterations must be positive")
    run_dir = args.run_dir or Path("logs/multi_chair") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if distributed.active():
        if args.run_dir is not None:
            distributed.assert_same(str(args.run_dir), "experiment output directory")
        run_dir = Path(distributed.gather_objects(str(run_dir))[0])
    occupied = run_dir.exists() and any(run_dir.iterdir()) and not args.resume
    if distributed.active():
        occupied = any(distributed.gather_objects(occupied))
    if occupied:
        parser.error("run-dir is not empty; use another directory or --resume")
    runner_cls = MixedChairExperimentRunner if args.scene_mode == "mixed" else ChairExperimentRunner
    options = {} if args.scene_mode == "mixed" else {"iterations_per_chair": args.iterations_per_chair}
    runner = runner_cls(manifest, root, args.cache_dir, agent, num_envs=args.num_envs,
        max_objects=args.max_objects, collision_profile=args.collision_profile,
        schedule_seed=args.seed, log_dir=run_dir, device=args.device, **options)
    try:
        if args.resume:
            # Validate mode/profile/split before touching an existing run's metadata.
            runner.load(args.resume)
        if distributed.primary():
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {**manifest, "data_root": os.path.relpath(root, run_dir.resolve())}
            (run_dir / "split.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            (run_dir / "experiment.json").write_text(json.dumps(dict(
                split_file=str(args.split.resolve()), data_root=str(root), cache_dir=str(args.cache_dir.resolve()),
                seed=args.seed, iterations_per_chair=args.iterations_per_chair, num_envs=args.num_envs,
                scene_mode=args.scene_mode, collision_profile=args.collision_profile,
                max_objects=args.max_objects, world_size=distributed.world_size(), agent=agent), indent=2) + "\n")
        if distributed.primary():
            print(f"[multi-chair] {len(manifest['train'])} training chairs; "
                  f"{args.num_envs} envs/rank, {distributed.world_size()} ranks; logs={run_dir}")
        runner.learn(iterations)
    finally:
        runner.close()
        distributed.shutdown()


if __name__ == "__main__":
    main()
