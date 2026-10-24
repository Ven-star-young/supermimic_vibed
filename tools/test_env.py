"""Check native mjlab state round-trip, FK and reference playback."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.assets.g1.g1_constants import TRACKED_BODIES, action_scale, G1_JOINT_NAMES
from supermimic.rotations import quat_to_mat
from tools._smoke import native_env, reset_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion_dir", help="Optional LAFAN1 CSV directory; default: synthetic motion")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    with native_env(args.num_envs, device=args.device, motion_dir=args.motion_dir) as env:
        cmd = env.command_manager.get_term("motion")
        ids = torch.arange(args.num_envs, device=args.device)
        frames = cmd.motion.clip_start[ids % len(cmd.motion.names)]
        reset_frames(env, frames)
        state = cmd.state()
        for name, actual, expected in zip(("p", "R", "v", "w", "q", "qd"), state, cmd.motion.ref_state(frames)):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            print(f"roundtrip {name}: max error {(actual - expected).abs().max().item():.2e}")
        body_ids = [cmd.robot.body_names.index(name) for name in TRACKED_BODIES]
        data = cmd.robot.data
        expected_fk = (
            data.body_link_pos_w[:, body_ids] - env.scene.env_origins[:, None],
            quat_to_mat(data.body_link_quat_w[:, body_ids]),
            data.body_link_lin_vel_w[:, body_ids],
            data.body_link_ang_vel_w[:, body_ids],
        )
        for name, actual, expected in zip(("position", "rotation", "linear velocity", "angular velocity"), cmd.mdp.kin.forward(*state), expected_fk):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            print(f"FK {name}: max error {(actual - expected).abs().max().item():.2e}")
        scale = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=args.device)
        for step in range(args.steps):
            f1 = torch.minimum(cmd.time_steps + 1, cmd.motion.clip_end[cmd.motion.clip_id[cmd.time_steps]])
            action = (cmd.motion.q[f1] - cmd.mdp.default_q) / scale
            _, reward, terminated, _, _ = env.step(action)
            assert all(torch.isfinite(value).all() for value in cmd.state())
            assert torch.isfinite(reward).all()
            if step % 10 == 0 or step == args.steps - 1:
                print(f"step {step}: normalized reward {reward.mean().item() / cmd.task.max_step_reward:.4f}, terminated fraction {terminated.float().mean().item():.3f}")
    print("mjlab environment checks passed")


if __name__ == "__main__":
    main()
