"""Run a frozen chair policy in local physics, from a chosen reference.

Examples (run from repository root):
  uv run python tools/deploy_policy.py --list
  uv run python tools/deploy_policy.py --index 0 --init clean
  uv run python tools/deploy_policy.py --index 0 --init perturbed --episodes 5
  uv run python tools/deploy_policy.py --reference /path/to/clip.npz
  uv run python tools/deploy_policy.py --viewer none --episodes 3 --output result.json

Clean starts at the reference pose AND velocity. Perturbed starts add bounded
robot state noise, while keeping reference, chair physics and sensors fixed.
This is simulated policy deployment; it sends no commands to robot hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import gc
import json
import math
import os
from pathlib import Path

import torch

from motion_data import load_motion
from motion_data.chair_split import file_digest, load_split, shape_key
from motion_data.collision import DEFAULT_CACHE_DIR, load_collision_parts
from motion_data.reference_training import load_reference_source
from supermimic.rl.chair_experiment import scene_cfg
from supermimic.rl.runner import SuperMimicRunner
from supermimic.rotations import mat_to_quat, so3_exp
from supermimic.scripts.chair_experiment import load_heldout_actor, policy_digest
from supermimic.tasks.entities.commands import EntityCommand, EntityCommandCfg
from supermimic.tasks.interaction.mdp import ObjectRewardCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "logs/cuhk_reference_deployment/model_2000.pt"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", type=Path, help="Training manifest; defaults to checkpoint's sibling split.json")
    parser.add_argument("--data-root", type=Path, help="Relocate the NPZ data without changing the split")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reference", type=Path, help="One NPZ path, or a filename relative to data-root")
    source.add_argument("--references", type=Path, help="Your reference choices JSON or exported manifest.json")
    parser.add_argument("--selection", type=Path, help="Initial keep/reject JSON for relocated reference choices")
    parser.add_argument("--partition", choices=("train", "test"), default="train")
    parser.add_argument("--index", type=int, default=0, help="Index in the list printed by --list")
    parser.add_argument("--list", action="store_true", help="List references and exit without creating physics")
    parser.add_argument("--all", action="store_true", help="Run all listed references sequentially (headless only)")
    parser.add_argument("--init", dest="init_mode", choices=("clean", "perturbed"), default="clean")
    parser.add_argument("--noise-scale", type=float, default=1., help="Multiply bounded robot initialization noise")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--start-frame", type=int, default=0, help="Frame on the resampled 50 Hz reference clock")
    phase.add_argument("--random-start", action="store_true", help="Randomize reference phase separately from state noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per reference; 0 repeats until viewer closes")
    parser.add_argument("--max-steps", type=int, help="Optional per-episode cutoff, reported separately from completion")
    parser.add_argument("--viewer", choices=("native", "viser", "none"), default="native")
    parser.add_argument("--port", type=int, default=8087, help="Viser port on localhost")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, help="Write initialization, episode scores and frozen-policy verification")
    return parser


def resolve_references(args, manifest, root):
    root = Path(root)
    if args.reference:
        path = args.reference.expanduser()
        paths = [path.resolve() if path.is_file() else (root / path).resolve()]
    elif args.references:
        source, groups, _ = load_reference_source(args.references, args.data_root, selection_path=args.selection)
        paths = [source / clip["path"] for group in groups for clip in group["clips"]]
    else:
        paths = [root / clip["path"] for group in manifest[args.partition] for clip in group["clips"]]
    known = {c["path"]: c["sha256"] for part in ("train", "test") for g in manifest[part] for c in g["clips"]}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Reference does not exist: {path}")
        if path.name in known and file_digest(path) != known[path.name]:
            raise ValueError(f"Reference hash changed from the training manifest: {path}")
    return sorted(set(paths), key=lambda p: str(p))


@dataclass(kw_only=True)
class DeploymentCommandCfg(EntityCommandCfg):
    init_mode: str = "clean"
    noise_scale: float = 1.
    start_frame: int = 0
    random_start: bool = False

    def build(self, env):
        return DeploymentCommand(self, env)


class DeploymentCommand(EntityCommand):
    def _resample_command(self, env_ids):
        # Exactly one clip is loaded into this physical scene. Keep phase
        # selection independent of the choice to perturb the robot's state.
        if self.cfg.random_start:
            frames = torch.randint(0, int(self.motion.clip_len[0]) - 1, (len(env_ids),), device=self.device)
        else:
            if not 0 <= self.cfg.start_frame < int(self.motion.clip_len[0]) - 1:
                raise ValueError("start-frame must leave at least one reference transition")
            frames = torch.full_like(env_ids, self.cfg.start_frame)
        self.set_reset_frames(env_ids, frames)
        super()._resample_command(env_ids)

    def reset_reference(self, env_ids):
        super().reset_reference(env_ids)
        if self.cfg.init_mode == "clean" or self.cfg.noise_scale == 0:
            return
        # Apply before the observation manager fills history. play=True remains
        # in force: no mass/friction randomization or observation/action noise.
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(self.time_steps[env_ids]))
        noise = self.cfg.reset_cfg

        def perturb(value, bounds):
            return value + (2 * torch.rand_like(value) - 1) * value.new_tensor(bounds) * self.cfg.noise_scale

        p = perturb(p, noise.position)
        R = so3_exp(perturb(torch.zeros_like(p), noise.rotation)) @ R
        v, w = perturb(v, noise.linear_velocity), perturb(w, noise.angular_velocity)
        q = perturb(q, noise.joint_position).clamp(self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self.robot.write_root_link_pose_to_sim(torch.cat((p + self.scene.env_origins[env_ids], mat_to_quat(R)), -1), env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.joint_ids, env_ids=env_ids)


def make_deployment_cfg(checkpoint, reference, args):
    reference = Path(reference).resolve()
    task = checkpoint["task"]
    if task.get("extra_objects"):
        raise ValueError("This deployment entry supports one reference chair, without additional scene objects")
    scene = {"clips": [{"path": reference.name}]}
    cfg = scene_cfg(reference.parent, scene, cache_dir=args.cache_dir, num_envs=1,
                    max_objects=task["dynamics_schema"]["max_objects"], play=True, seed=args.seed,
                    collision_profile=task.get("chair_experiment", {}).get("collision_profile", "default"))
    original = cfg.commands["motion"]
    command = DeploymentCommandCfg(**{f.name: getattr(original, f.name) for f in fields(original)},
        init_mode=args.init_mode, noise_scale=args.noise_scale,
        start_frame=args.start_frame, random_start=args.random_start)
    command.history_length = checkpoint["policy_config"]["history_length"]
    command.goal_frames = checkpoint["policy_config"]["goal_frames"]
    command.object_reward_cfg = ObjectRewardCfg(**task["object_reward"])
    return replace(cfg, commands={"motion": command})


class DeploymentEnv(RslRlVecEnvWrapper):
    """One episode at a time; retain terminal state until metrics are captured."""
    def __init__(self, native, episodes=1, max_steps=None):
        self.records, self.finished = [], False
        self.episodes, self.max_steps = episodes, max_steps
        self.steps, self.total_reward, self.initial = 0, 0., None
        self._last_result = None
        super().__init__(native)
        self._remember_start()

    def _remember_start(self):
        command = self.unwrapped.command_manager.get_term("motion")
        self.steps, self.total_reward = 0, 0.
        self.frame0 = int(command.time_steps[0])
        self.initial = dict(frame=self.frame0, state=command.dynamics.pack(command.state())[0].cpu().tolist())

    def reset(self):
        result = super().reset()
        self.finished = False
        self._remember_start()
        return result

    def step(self, actions):
        if self.finished:
            return self._last_result
        result = super().step(actions)
        self._last_result = result
        native = self.unwrapped
        if bool(native.sim.data.overflow.any()):
            raise RuntimeError("Physics buffer overflow; this rollout is invalid")
        command = native.command_manager.get_term("motion")
        if not bool(torch.isfinite(command.dynamics.pack(command.state())).all()):
            raise FloatingPointError("Nonfinite physical state")
        self.steps += 1
        self.total_reward += float(result[1][0])
        done = bool(result[2][0])
        if done or (self.max_steps is not None and self.steps >= self.max_steps):
            failed = bool(native.reset_terminated[0])
            completed = not failed and int(command.time_steps[0]) >= int(command.motion.clip_end[0])
            remaining = int(command.motion.clip_end[0]) - self.frame0
            record = dict(episode=len(self.records) + 1, initial=self.initial, steps=self.steps,
                seconds=self.steps * native.step_dt,
                outcome="tracking_failure" if failed else "reference_complete" if completed else "step_limit",
                normalized_return=self.total_reward / (remaining * command.task.max_step_reward),
                mean_step_reward=self.total_reward / (self.steps * command.task.max_step_reward))
            self.records.append(record)
            print("[deploy] " + json.dumps({k: v for k, v in record.items() if k != "initial"}), flush=True)
            self.finished = self.episodes > 0 and len(self.records) >= self.episodes
            if not self.finished:
                observation, _ = self.reset()
                result = (observation, *result[1:])
        return result


def run_reference(checkpoint, manifest, reference, args):
    clip = load_motion(reference)
    load_collision_parts(clip, args.cache_dir)  # Explicit missing-cache error before allocating CUDA.
    key = shape_key(clip)
    partitions = [p for p in ("train", "test") if key in {g["shape_key"] for g in manifest[p]}]
    digest = file_digest(reference)
    reference_partitions = [p for p in ("train", "test") if digest in {
        c["sha256"] for g in manifest[p] for c in g["clips"]}]
    exposure = (checkpoint.get("infos") or {}).get("chair_experiment", {}).get("visited_train_shapes", [])
    identity = dict(reference=str(reference), sha256=digest, shape_key=key,
                    geometry_partition=partitions[0] if partitions else "external",
                    reference_partition=reference_partitions[0] if reference_partitions else "external",
                    visited_in_checkpoint=key in exposure)
    print("[deploy] " + json.dumps(identity), flush=True)
    native = ManagerBasedRlEnv(make_deployment_cfg(checkpoint, reference, args), device=args.device)
    try:
        env = DeploymentEnv(native, args.episodes, args.max_steps)
        train_cfg = checkpoint["train_cfg"] | dict(amp="none", world_model_file=None,
                                                   world_model_replays=(), no_compile=True)
        runner = SuperMimicRunner(env, train_cfg, device=args.device)
        # Explicit frozen-policy transfer, including a new chair's local geometry
        # bank. Full training resume would incorrectly require identical scene IDs.
        _, training_root = load_split(args.split or args.checkpoint.parent / "split.json", args.data_root)
        load_heldout_actor(runner, checkpoint, manifest, training_root)
        before = policy_digest(runner.pi)
        policy = runner.get_inference_policy()
        if args.viewer == "none":
            with torch.inference_mode():
                while not env.finished:
                    env.step(policy(env.get_observations()))
        else:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
            viewer_type = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer

            class EpisodeViewer(viewer_type):
                def is_running(self):
                    return not env.finished and super().is_running()

            server = None
            try:
                if args.viewer == "viser":
                    import viser
                    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="G1 policy deployment")
                viewer = EpisodeViewer(env, policy, **({"viser_server": server} if server else {}))
                viewer.run()
                if viewer._last_error:
                    raise RuntimeError(viewer._last_error)
            finally:
                if server is not None:
                    server.stop()
        if policy_digest(runner.pi) != before:
            raise RuntimeError("Policy weights or normalization changed during deployment")
        assert runner.wm is runner.critic is runner.buffer is None
        return dict(**identity, frozen_policy=True, episodes=env.records)
    finally:
        native.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.index < 0 or args.episodes < 0 or args.start_frame < 0
            or not math.isfinite(args.noise_scale) or args.noise_scale < 0
            or (args.max_steps is not None and args.max_steps < 1)):
        raise ValueError("Invalid index, episode count, frame, noise scale or step limit")
    if args.viewer == "none" and args.episodes == 0:
        raise ValueError("Headless runs require a finite --episodes count")
    if args.all and args.viewer != "none":
        raise ValueError("--all requires --viewer none; use --index to view one reference")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Deployment uses one process; do not launch with torchrun")
    manifest, root = load_split(args.split or args.checkpoint.parent / "split.json", args.data_root)
    paths = resolve_references(args, manifest, root)
    if args.list:
        for i, path in enumerate(paths):
            print(f"{i:3d}  {path.name}")
        return
    if not paths or args.index >= len(paths):
        raise ValueError(f"Reference index {args.index} out of range (found {len(paths)})")
    # Full checkpoint stays on CPU; only the Actor is allocated on the GPU.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("task", {}).get("chair_experiment", {}).get("split_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Checkpoint and training split fingerprints differ")
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_digest(args.checkpoint),
                  iteration=checkpoint["iteration"], init_mode=args.init_mode, noise_scale=args.noise_scale,
                  seed=args.seed, random_start=args.random_start, start_frame=args.start_frame, references=[])
    print(f"[deploy] checkpoint iteration={checkpoint['iteration']}, init={args.init_mode}, seed={args.seed}", flush=True)
    for reference in paths if args.all else [paths[args.index]]:
        report["references"].append(run_reference(checkpoint, manifest, reference, args))
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
