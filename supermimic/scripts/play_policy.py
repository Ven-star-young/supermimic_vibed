"""Run a checkpoint from random reference states in a live viewer or an MP4.

python -m supermimic.scripts.play_policy --checkpoint-file model_2000.pt
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess


class LivePolicyEnv:
    """Adapt manual episode resets to mjlab's interactive viewer loop."""

    def __init__(self, env, reset_reference):
        self.unwrapped, self.cfg = env, env.cfg
        self.num_envs, self.device = env.num_envs, env.device
        self.reset_reference = reset_reference
        self.reset()

    def get_observations(self):
        return self.unwrapped.get_observations()

    def reset(self):
        obs, row, _ = self.reset_reference()
        print("RESET " + json.dumps(row), flush=True)
        return obs, {}

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.unwrapped.step(actions)
        if bool((terminated | truncated).any()):
            manager = self.unwrapped.termination_manager
            reasons = [name for name in manager.active_terms if bool(manager.get_term(name)[0])]
            print("END " + ", ".join(reasons), flush=True)
            obs, _ = self.reset()
        return obs, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--task", default="Mjlab-SuperMimic-Sitting-Unitree-G1")
    parser.add_argument("--motion-dir", type=Path)
    parser.add_argument("--object-motion-file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8, help="Number of episodes when recording video")
    parser.add_argument("--start", choices=("random", "first"), default="random")
    parser.add_argument("--training-conditions", action="store_true",
                        help="Enable training domain randomization, reset and observation noise")
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument("--video", type=Path, help="Record to MP4 instead of opening the live viewer")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.video:
        if not shutil.which("ffmpeg"):
            parser.error("ffmpeg is required to record video")
        os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    import supermimic.tasks  # noqa: F401 -- register tasks

    checkpoint = torch.load(args.checkpoint_file, map_location="cpu", weights_only=True)
    policy_cfg = checkpoint["policy_config"]
    report = dict(checkpoint=str(args.checkpoint_file.resolve()), iteration=checkpoint.get("iteration"),
                  task=args.task, seed=args.seed, start=args.start,
                  training_conditions=args.training_conditions, episodes=[])
    del checkpoint
    cfg = load_env_cfg(args.task, play=not args.training_conditions)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, args.seed, False
    command_cfg = cfg.commands["motion"]
    command_cfg.history_length = policy_cfg["history_length"]
    command_cfg.goal_frames = policy_cfg["goal_frames"]
    if args.motion_dir:
        command_cfg.motion_dir = str(args.motion_dir)
    if args.object_motion_file:
        if not hasattr(command_cfg, "object_motion_file"):
            parser.error("--object-motion-file requires an object task")
        command_cfg.object_motion_file = args.object_motion_file
    command_cfg.debug_vis = False
    cfg.viewer.width, cfg.viewer.height = 960, 720
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 3.3, -18, 135
    cfg.viewer.max_extra_envs = 0
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array" if args.video else None)
    encoder = None
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner = load_runner_cls(args.task)(wrapped, asdict(load_rl_cfg(args.task)), device=args.device)
        runner.load(str(args.checkpoint_file), load_cfg={"actor": True})
        policy = runner.get_inference_policy(device=args.device)
        command = env.command_manager.get_term("motion")
        motion = command.motion
        ids = torch.arange(1, device=args.device)
        rng = np.random.default_rng(args.seed)

        def reset_reference():
            clip = int(rng.integers(len(motion.names)))
            begin, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
            start = int(rng.integers(begin, end)) if args.start == "random" else begin
            command.set_reset_frames(ids, torch.tensor([start], device=args.device))
            obs, _ = env.reset()
            row = dict(clip=motion.names[clip], start_frame=start - begin,
                       clip_frames=end - begin + 1, steps=0, reward_sum=0.0)
            if getattr(command.cfg, "object_enabled", False):
                row.update(chair_mass=float(command.xi[0, 33]),
                           robot_chair_friction=float(command.xi[0, 43]))
            return obs, row, clip

        if not args.video:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

            viewer_cls = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
            print(f"Live policy iteration {report['iteration']} | {args.start} reference resets", flush=True)
            with torch.no_grad():
                viewer_cls(LivePolicyEnv(env, reset_reference), policy).run()
            return

        from PIL import Image, ImageDraw, ImageFont

        args.video.parent.mkdir(parents=True, exist_ok=True)
        fps = round(1 / env.step_dt)
        report.update(control_dt=env.step_dt, clips=list(motion.names), policy_config=policy_cfg,
                      video=str(args.video.resolve()), terminal_hold_seconds=0.5)
        encoder = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "960x720", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.video),
        ], stdin=subprocess.PIPE)
        font = ImageFont.load_default(size=19)

        def render(lines, copies=1):
            image = Image.fromarray(env.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 960, 28 * len(lines) + 12), fill=(20, 24, 30))
            draw.multiline_text((12, 6), "\n".join(lines), font=font, fill="white", spacing=8)
            pixels = image.tobytes()
            for _ in range(copies):
                encoder.stdin.write(pixels)

        with torch.inference_mode():
            for episode in range(args.episodes):
                obs, row, clip = reset_reference()
                row["episode"] = episode + 1
                label = [f"Policy iteration {report['iteration']} | episode {episode + 1}/{args.episodes}",
                         f"Clip {clip} | {args.start} start {row['start_frame']}/{row['clip_frames'] - 1}"]
                render(label + ["RESET to reference state; subsequent motion uses policy + physics"])
                while True:
                    action = policy(obs)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("Policy produced non-finite actions")
                    obs, reward, terminated, truncated, _ = env.step(action)
                    row["steps"] += 1
                    row["reward_sum"] += float(reward[0])
                    done = bool(terminated[0] | truncated[0])
                    if done:
                        row["terminated"] = bool(terminated[0])
                        row["truncated"] = bool(truncated[0])
                        row["end_reasons"] = [name for name in env.termination_manager.active_terms
                                              if bool(env.termination_manager.get_term(name)[0])]
                    status = " + ".join(row["end_reasons"]) + " (terminal frame held)" if done else "RUNNING"
                    render(label + [f"t={row['steps'] * env.step_dt:.2f}s | {status}"],
                           copies=max(1, fps // 2) if done else 1)
                    if done:
                        break
                row["duration_s"] = row["steps"] * env.step_dt
                row["mean_reward"] = row["reward_sum"] / row["steps"]
                report["episodes"].append(row)
                print(json.dumps(row), flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        encoder = None
        args.video.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Video: {args.video.resolve()}", flush=True)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
        env.close()


if __name__ == "__main__":
    main()
