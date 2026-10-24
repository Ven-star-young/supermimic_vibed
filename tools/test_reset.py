"""Check immediate and partial native mjlab resets and root-link velocities."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools._smoke import native_env, reset_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion_dir", help="Optional LAFAN1 CSV directory; default: synthetic motion")
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("--num_envs must be at least 2 for a partial reset check")
    with native_env(args.num_envs, device=args.device, motion_dir=args.motion_dir) as env:
        cmd = env.command_manager.get_term("motion")
        for _ in range(3):
            env.step(torch.zeros(args.num_envs, 29, device=args.device))
        before = tuple(value.clone() for value in cmd.state())
        ids = torch.arange(0, args.num_envs, 2, device=args.device)
        untouched = torch.arange(1, args.num_envs, 2, device=args.device)
        frames = cmd.motion.clip_start[ids % len(cmd.motion.names)]
        reset_frames(env, frames, ids)
        for name, actual, expected, previous in zip(("p", "R", "v", "w", "q", "qd"), cmd.state(), cmd.motion.ref_state(frames), before):
            torch.testing.assert_close(actual[ids], expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(actual[untouched], previous[untouched], atol=2e-5, rtol=2e-5)
            print(f"reset {name}: max error {(actual[ids] - expected).abs().max().item():.2e}")
    print("mjlab reset checks passed")


if __name__ == "__main__":
    main()
