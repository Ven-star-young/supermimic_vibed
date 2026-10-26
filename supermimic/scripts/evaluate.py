"""Quantitative whole-library evaluation using the registered mjlab environment.

Interactive playback belongs to mjlab's ``play``. This script scores each motion
from its first frame, writes per-clip results, and optionally records trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict

from supermimic.tasks import TASK_ID  # Registers the external task.
from supermimic.rotations import mat_to_quat


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--checkpoint-file", "--ckpt", dest="checkpoint_file")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--motion-dir", type=Path, help="Override the selected task's default motion directory")
    parser.add_argument("--object-motion-file", type=str, help="Select the sitting scene's object geometry")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--every", type=int, default=1, help="evaluate every Nth saved checkpoint")
    parser.add_argument("--record", type=Path, help="save G1 trajectory NPZ (requires --checkpoint-file)")
    return parser


def _policy_config(checkpoint):
    if "policy_config" in checkpoint:
        return checkpoint["policy_config"]
    legacy = checkpoint.get("args", {})
    return {"history_length": legacy.get("hist", 10), "goal_frames": legacy.get("goal_frames", 10),
            "amp": legacy.get("amp", "none")}


@torch.no_grad()
def evaluate(env, runner, record=None):
    """One world per clip; score only until its first termination or clip end."""
    native = env.unwrapped
    command = native.command_manager.get_term("motion")
    task, motion = command.task, command.motion
    policy = runner.get_inference_policy(device=native.device)
    ids = torch.arange(native.num_envs, device=native.device)
    if native.num_envs != len(motion.names):
        raise ValueError("Whole-library evaluation requires one environment per motion")
    command.set_reset_frames(ids, motion.clip_start)
    obs, _ = native.reset()
    returns = torch.zeros(native.num_envs, device=native.device)
    finished = torch.zeros(native.num_envs, dtype=torch.bool, device=native.device)
    failed = torch.zeros_like(finished)
    trajectory = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    object_enabled = getattr(command.cfg, "object_enabled", False)
    if object_enabled:
        trajectory.update(object_p=[], object_quat=[])
    while not bool(finished.all()):
        alive = ~finished
        if record is not None:
            state = command.state()
            trajectory["root_p"].append(state[0].cpu())
            trajectory["root_quat"].append(mat_to_quat(state[1]).cpu())
            trajectory["q"].append(state[4].cpu())
            trajectory["f"].append(command.time_steps.cpu().clone())
            trajectory["alive"].append(alive.cpu())
            if object_enabled:
                trajectory["object_p"].append(state[6].cpu())
                trajectory["object_quat"].append(mat_to_quat(state[7]).cpu())
        obs, reward, terminated, truncated, _ = native.step(policy(obs))
        returns += torch.where(alive, reward, 0)
        failed |= alive & terminated
        finished |= terminated | truncated
    scores = returns / ((motion.clip_len - 1).float() * task.max_step_reward)
    result = {
        "norm_return": scores.mean().item(),
        "norm_return_framew": (returns.sum() / ((motion.clip_len - 1).sum() * task.max_step_reward)).item(),
        "n_failed": int(failed.sum()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": {name: float(score) for name, score in zip(motion.names, scores.tolist())},
    }
    if record is not None:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        object_reference = (dict(ref_object_p=motion.object_p.cpu().numpy(),
                                 ref_object_quat=mat_to_quat(motion.object_R).cpu().numpy())
                            if object_enabled else {})
        np.savez_compressed(record, names=np.array(motion.names), **object_reference,
                            **{k: torch.stack(v).numpy() for k, v in trajectory.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(),
                            ref_q=motion.q.cpu().numpy(), clip_start=motion.clip_start.cpu().numpy(),
                            clip_len=motion.clip_len.cpu().numpy())
    return result


def score_checkpoint(path: Path, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy_cfg = _policy_config(checkpoint)
    cfg = load_env_cfg(args.task, play=True)
    if args.motion_dir is not None:
        cfg.commands["motion"].motion_dir = str(args.motion_dir)
    if args.object_motion_file is not None:
        if not hasattr(cfg.commands["motion"], "object_motion_file"):
            raise ValueError("--object-motion-file requires an interaction task")
        cfg.commands["motion"].object_motion_file = args.object_motion_file
    cfg.commands["motion"].history_length = policy_cfg["history_length"]
    cfg.commands["motion"].goal_frames = policy_cfg["goal_frames"]
    if hasattr(cfg.commands["motion"], "object_enabled"):
        from supermimic.tasks.interaction.assets import select_motion_files
        cfg.scene.num_envs = len(select_motion_files(cfg.commands["motion"]))
    else:
        cfg.scene.num_envs = len(list(Path(cfg.commands["motion"].motion_dir).glob("*.npz")))
    if not cfg.scene.num_envs:
        raise ValueError(f"No standard motion NPZ files found in {cfg.commands['motion'].motion_dir}")
    native = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env = RslRlVecEnvWrapper(native)
        runner = load_runner_cls(args.task)(env, asdict(load_rl_cfg(args.task)), device=args.device)
        runner.load(str(path), load_cfg={"actor": True}, map_location=args.device)
        result = evaluate(env, runner, args.record)
        result.update(checkpoint=str(path), iteration=checkpoint.get("iteration"),
                      train_wall_h=checkpoint.get("wall_time", 0) / 3600,
                      saved_at=checkpoint.get("saved_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))),
                      eval_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        path.with_name(path.stem + "_eval.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}, indent=2), flush=True)
        return result
    finally:
        native.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.every < 1:
        raise ValueError("--every must be positive")
    if args.checkpoint_file:
        return score_checkpoint(Path(args.checkpoint_file), args)
    if args.run_dir is None:
        raise ValueError("Provide --checkpoint-file or --run-dir")
    if args.record is not None:
        raise ValueError("--record requires --checkpoint-file")
    output = args.run_dir / "eval.csv"
    done = set()
    if output.exists():
        with output.open() as stream:
            done = {row["checkpoint"] for row in csv.DictReader(stream)}
    else:
        with output.open("w") as stream:
            csv.writer(stream).writerow(("checkpoint", "iteration", "train_wall_h", "saved_at", "eval_at",
                                        "norm_return", "norm_return_framew", "n_failed", "failed"))
    pattern = re.compile(r"(?:model|ckpt)_(\d+)\.pt$")
    while True:
        paths = sorted((p for p in args.run_dir.glob("*.pt") if pattern.fullmatch(p.name)),
                       key=lambda p: int(pattern.fullmatch(p.name).group(1)))
        for path in paths[::args.every]:
            if path.name in done:
                continue
            result = score_checkpoint(path, args)
            with output.open("a") as stream:
                csv.writer(stream).writerow((path.name, result["iteration"], result["train_wall_h"],
                                            result["saved_at"], result["eval_at"], result["norm_return"],
                                            result["norm_return_framew"], result["n_failed"], "|".join(result["failed"])))
            done.add(path.name)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
