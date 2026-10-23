"""Spawn G1s directly in mjlab and report simulator control-step throughput."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list
from tools._smoke import load_motion, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    kin = G1Kinematics(args.device)
    motion = load_motion(kin, args.device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
    with TrackingSim(args.num_envs, motion, mdp, device=args.device) as env:
        print("SIMULATOR JOINTS", env.robot.joint_names)
        print("POLICY JOINTS", G1_JOINT_NAMES)
        print("BODIES", env.robot.body_names)
        action = torch.zeros(args.num_envs, 29, device=args.device)
        for _ in range(5):
            env.step(action)
        synchronize(args.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            result = env.step(action)
        synchronize(args.device)
        elapsed = time.perf_counter() - start
        assert torch.isfinite(result["s1"]).all()
        print(f"{args.steps * args.num_envs / elapsed:.0f} environment transitions/s at 50 Hz control")


if __name__ == "__main__":
    main()
