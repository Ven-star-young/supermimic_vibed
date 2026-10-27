"""Measure a mixed-chair native scene without allocating learning networks.

Run with python -m tools.measure_chair_scene --split ... --num-envs 128.
Whole-device memory includes Warp and CUDA graphs, unlike PyTorch peak counters.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from motion_data.chair_split import load_split
from motion_data.collision import COLLISION_PROFILES, DEFAULT_CACHE_DIR
from supermimic.rl.chair_experiment import preflight
from supermimic.rl.mixed_chair_experiment import mixed_scene_cfg
from supermimic.rl.vecenv import ReplayEnvAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--collision-profile", choices=COLLISION_PROFILES, default="default")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, root = load_split(args.split, args.data_root)
    preflight(manifest, root, args.cache_dir, "train", args.collision_profile)
    cfg = mixed_scene_cfg(manifest, root, args.cache_dir, num_envs=args.num_envs,
                          collision_profile=args.collision_profile)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    start = time.monotonic()
    native = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env = RslRlVecEnvWrapper(native)
        replay = ReplayEnvAdapter(env)
        command = replay.command
        torch.cuda.synchronize(args.device)
        samples = []
        for step in range(args.steps):
            transition = replay.step(torch.zeros(args.num_envs, 29, device=args.device))
            if not torch.equal(command.clip_scene[command.motion.clip_id[transition["f"]]], command.env_scene):
                raise AssertionError("Mixed reference assignment mismatch")
            torch.cuda.synchronize(args.device)
            free, total = torch.cuda.mem_get_info(args.device)
            samples.append(total - free)
        result = dict(collision_profile=args.collision_profile, num_envs=args.num_envs, steps=args.steps,
                      seconds=time.monotonic() - start,
                      assigned_counts=command.env_scene.bincount().cpu().tolist(),
                      ngeom=native.sim.mj_model.ngeom, nmesh=native.sim.mj_model.nmesh,
                      nconmax=cfg.sim.nconmax, njmax=cfg.sim.njmax,
                      max_sampled_device_used_bytes=max(samples),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      sampling_note="Device memory sampled after each control step; not an instantaneous peak")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        native.close()


if __name__ == "__main__":
    main()
