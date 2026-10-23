"""Check mjlab state round-trip, FK, and reference playback (no Isaac launcher)."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat
from tools._smoke import load_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion_dir", help="Optional LAFAN1 CSV directory; default: synthetic motion")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    kin = G1Kinematics(args.device)
    motion = load_motion(kin, args.device, args.motion_dir)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
    with TrackingSim(args.num_envs, motion, mdp, device=args.device, eval_mode=True) as env:
        ids = torch.arange(args.num_envs, device=args.device)
        frames = motion.clip_start[ids % len(motion.names)]
        env.reset(ids, frames)
        state = env.state()
        for name, actual, expected in zip(("p", "R", "v", "w", "q", "qd"), state, motion.ref_state(frames)):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            print(f"roundtrip {name}: max error {(actual - expected).abs().max().item():.2e}")
        body_ids = [env.robot.body_names.index(name) for name in TRACKED_BODIES]
        data = env.robot.data
        expected_fk = (
            data.body_link_pos_w[:, body_ids] - env.scene.env_origins[:, None],
            quat_to_mat(data.body_link_quat_w[:, body_ids]),
            data.body_link_lin_vel_w[:, body_ids],
            data.body_link_ang_vel_w[:, body_ids],
        )
        for name, actual, expected in zip(("position", "rotation", "linear velocity", "angular velocity"), kin.forward(*state), expected_fk):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            print(f"FK {name}: max error {(actual - expected).abs().max().item():.2e}")
        for step in range(args.steps):
            f1 = torch.minimum(env.f + 1, motion.clip_end[motion.clip_id[env.f]])
            action = (motion.q[f1] - env.default_q) / env.alpha
            result = env.step(action)
            assert torch.isfinite(result["s1"]).all()
            assert torch.isfinite(result["r"]).all()
            if step % 10 == 0 or step == args.steps - 1:
                print(f"step {step}: normalized reward {result['r'].mean().item() / 0.1:.4f}, failed fraction {env.failed_eval.float().mean().item():.3f}")
            if env.done_eval.all():
                break
    print("mjlab environment checks passed")


if __name__ == "__main__":
    main()
