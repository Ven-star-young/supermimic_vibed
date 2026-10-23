"""Check immediate and partial mjlab resets, including root-link velocity."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.robot import default_joint_pos_list
from tools._smoke import load_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion_dir", help="Optional LAFAN1 CSV directory; default: synthetic motion")
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("--num_envs must be at least 2 for a partial reset check")
    kin = G1Kinematics(args.device)
    motion = load_motion(kin, args.device, args.motion_dir)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
    with TrackingSim(args.num_envs, motion, mdp, device=args.device, eval_mode=True) as env:
        for _ in range(3):
            env.step(torch.zeros(args.num_envs, 29, device=args.device))
        before = env.state()
        ids = torch.arange(0, args.num_envs, 2, device=args.device)
        untouched = torch.arange(1, args.num_envs, 2, device=args.device)
        frames = motion.clip_start[ids % len(motion.names)]
        env.reset(ids, frames)
        for name, actual, expected, previous in zip(("p", "R", "v", "w", "q", "qd"), env.state(), motion.ref_state(frames), before):
            torch.testing.assert_close(actual[ids], expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(actual[untouched], previous[untouched], atol=2e-5, rtol=2e-5)
            print(f"reset {name}: max error {(actual[ids] - expected).abs().max().item():.2e}")
    print("mjlab reset checks passed")


if __name__ == "__main__":
    main()
