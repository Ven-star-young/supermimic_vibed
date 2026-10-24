"""Spawn G1s in the stock mjlab environment and report control-step throughput."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES
from tools._smoke import native_env, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with native_env(args.num_envs, device=args.device, play=False) as env:
        cmd = env.command_manager.get_term("motion")
        print("SIMULATOR JOINTS", cmd.robot.joint_names)
        print("POLICY JOINTS", G1_JOINT_NAMES)
        print("BODIES", cmd.robot.body_names)
        action = torch.zeros(args.num_envs, 29, device=args.device)
        for _ in range(5):
            env.step(action)
        synchronize(args.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(action)
        synchronize(args.device)
        elapsed = time.perf_counter() - start
        assert all(torch.isfinite(value).all() for value in cmd.state())
        print(f"{args.steps * args.num_envs / elapsed:.0f} environment transitions/s at 50 Hz control")


if __name__ == "__main__":
    main()
