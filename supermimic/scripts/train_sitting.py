"""Train one sitting policy across the training chairs, one chair per environment.

Single GPU:
  python -m supermimic.scripts.train_sitting --split data/splits/sitting_back_references_seed0.json
Two GPUs:
  torchrun --standalone --nproc_per_node=2 -m supermimic.scripts.train_sitting --num-envs 128
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from motion_data.chair_split import load_split
from motion_data.collision import COLLISION_PROFILES, DEFAULT_CACHE_DIR
from supermimic.rl import distributed
from supermimic.tasks.sitting.rl_cfg import make_runner_cfg


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=Path("data/splits/sitting_back_references_seed0.json"))
    parser.add_argument("--data-root", type=Path, help="Relocate the motion NPZ files")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--collision-profile", choices=COLLISION_PROFILES, default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel environments per GPU")
    parser.add_argument("--iterations", type=int, help="Additional learner iterations, including when resuming")
    parser.add_argument("--agent-json", type=Path, help="JSON overrides of SuperMimicRunnerCfg")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a full checkpoint from this experiment")
    parser.add_argument("--check", action="store_true", help="Validate training references and collision caches without simulation")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    if args.check and distributed.world_size() > 1:
        parser.error("--check uses one process; torchrun is for training")
    from supermimic.tasks.sitting.experiment import SittingExperimentRunner, preflight

    manifest, root = load_split(args.split, args.data_root)
    if args.check:
        preflight(manifest, root, args.cache_dir, "train", args.collision_profile)
        print(f"Validated {len(manifest['train'])} training chairs; split={manifest['fingerprint']}")
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
    args.device = str(distributed.initialize(args.device))
    runner = None
    try:
        run_dir = args.run_dir or (args.resume.parent if args.resume else
            Path("logs/sitting") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        if distributed.active():
            if args.run_dir or args.resume:
                distributed.assert_same(str(run_dir), "experiment output directory")
            run_dir = Path(distributed.gather_objects(str(run_dir))[0])
        occupied = run_dir.exists() and any(run_dir.iterdir()) and not args.resume
        if distributed.active():
            occupied = any(distributed.gather_objects(occupied))
        if occupied:
            parser.error("run-dir is not empty; use another directory or --resume")
        runner = SittingExperimentRunner(manifest, root, args.cache_dir, agent,
            num_envs=args.num_envs, schedule_seed=args.seed, log_dir=run_dir,
            device=args.device, collision_profile=args.collision_profile)
        if args.resume:
            runner.load(args.resume)
        if distributed.primary():
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {**manifest, "data_root": os.path.relpath(root, run_dir.resolve())}
            (run_dir / "split.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            (run_dir / "experiment.json").write_text(json.dumps(dict(
                split_file=str(args.split.resolve()), split_fingerprint=manifest["fingerprint"],
                data_root=str(root), cache_dir=str(args.cache_dir.resolve()),
                seed=args.seed, num_envs=args.num_envs, world_size=distributed.world_size(),
                collision_profile=args.collision_profile, iterations=iterations, agent=agent,
                resume=str(args.resume.resolve()) if args.resume else None), indent=2) + "\n")
            print(f"[sitting] {len(manifest['train'])} training chairs | {args.num_envs} envs/GPU "
                  f"x {distributed.world_size()} GPUs | logs={run_dir}", flush=True)
        runner.learn(iterations)
    finally:
        if runner is not None:
            runner.close()
        distributed.shutdown()


if __name__ == "__main__":
    main()
