"""Run a frozen chair policy in local physics, from a chosen reference.

Examples (run from repository root):
  uv run python -m supermimic.scripts.play_policy --checkpoint logs/sitting/model_2000.pt --list
  uv run python -m supermimic.scripts.play_policy --checkpoint logs/sitting/model_2000.pt --index 0 --init clean
  uv run python -m supermimic.scripts.play_policy --checkpoint logs/sitting/model_2000.pt --index 0 --init perturbed --episodes 5
  uv run python -m supermimic.scripts.play_policy --checkpoint logs/sitting/model_2000.pt --reference /path/to/clip.npz
  uv run python -m supermimic.scripts.play_policy --checkpoint logs/sitting/model_2000.pt --viewer none --episodes 3 --output result.json

Clean starts at the reference pose AND velocity. Perturbed starts add bounded
robot state noise, while keeping reference, chair physics and sensors fixed.
This is simulated policy deployment; it sends no commands to robot hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from motion_data import load_motion
from motion_data.chair_split import file_digest, load_split, shape_key
from motion_data.collision import DEFAULT_CACHE_DIR, load_collision_parts
from motion_data.reference_training import load_reference_source
from supermimic.tasks.sitting.experiment import scene_cfg
from supermimic.rl.runner import SuperMimicRunner
from supermimic.rotations import mat_to_quat, so3_exp
from supermimic.tasks.sitting.commands import SittingCommand, SittingCommandCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict
from supermimic.rl.evaluation_report import planned_seeds, build_report_summary, atomic_write_report
from supermimic.tasks.sitting.evaluation import SittingEvaluationCfg, SittingEpisodeMetrics

def experiment_metadata(checkpoint):
    return (checkpoint.get("infos") or {}).get("sitting_experiment", {})


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_actor(runner, checkpoint, manifest):
    """Load frozen policy semantics while retaining the new scene's geometry IDs."""
    runner.task.validate_checkpoint(checkpoint)
    if experiment_metadata(checkpoint).get("split_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Checkpoint and training split fingerprints differ")
    exposure = set(experiment_metadata(checkpoint).get("visited_train_shapes", []))
    if not exposure <= {group["shape_key"] for group in manifest["train"]}:
        raise ValueError("Checkpoint exposure includes a non-training chair")
    for key in ("history_length", "goal_frames"):
        if checkpoint["policy_config"][key] != runner.policy_config[key]:
            raise ValueError(f"Incompatible policy {key}")
    encoder = {key: value for key, value in checkpoint["pi"].items()
               if key.startswith("geometry_encoder.")}
    if not encoder or any(key not in checkpoint["wm"] or not torch.equal(value, checkpoint["wm"][key])
                          for key, value in encoder.items()):
        raise ValueError("Checkpoint has inconsistent shared geometry encoder weights")
    # Geometry-bank indices are scene metadata, never trained embeddings. The
    # held-out cloud is evaluated through the trained shared PointNet weights.
    runner.pi.load_state_dict(checkpoint["pi"], strict=True)
    runner.pi.eval().requires_grad_(False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "--checkpoint-file", type=Path, required=True)
    parser.add_argument("--split", type=Path, help="Training manifest; defaults to checkpoint's sibling split.json")
    parser.add_argument("--data-root", type=Path, help="Relocate the NPZ data without changing the split")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reference", type=Path, help="One NPZ path, or a filename relative to data-root")
    source.add_argument("--references", type=Path, help="Your reference choices JSON or exported manifest.json")
    parser.add_argument("--selection", type=Path, help="Initial keep/reject JSON for relocated reference choices")
    parser.add_argument("--partition", choices=("train", "test", "all"), default="train")
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
    parser.add_argument("--output", type=Path, help="Atomic JSON report, updated after every episode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all selected references from frame 0, headless, with both conditions")
    parser.add_argument("--conditions", nargs="+", choices=("clean", "perturbed"), default=("clean", "perturbed"))
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit episode seeds, reused for every reference and condition")
    parser.add_argument("--metrics-json", type=Path, help="Stable-contact defaults and optional per-reference overrides")
    parser.add_argument("--browse", action="store_true", help="Open local reference picker and live policy viewer")
    parser.add_argument("--report", type=Path, help="Existing evaluation JSON to display in the browser")
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
        partitions = ("train", "test") if args.partition == "all" else (args.partition,)
        paths = [root / clip["path"] for part in partitions for group in manifest[part] for clip in group["clips"]]
    known = {c["path"]: c["sha256"] for part in ("train", "test") for g in manifest[part] for c in g["clips"]}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Reference does not exist: {path}")
        if path.name in known and file_digest(path) != known[path.name]:
            raise ValueError(f"Reference hash changed from the training manifest: {path}")
    return sorted(set(paths), key=lambda p: str(p))


@dataclass(kw_only=True)
class DeploymentCommandCfg(SittingCommandCfg):
    init_mode: str = "clean"
    noise_scale: float = 1.
    start_frame: int = 0
    random_start: bool = False

    def build(self, env):
        return DeploymentCommand(self, env)


class DeploymentCommand(SittingCommand):
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
    scene = {"clips": [{"path": reference.name}]}
    cfg = scene_cfg(reference.parent, scene, cache_dir=args.cache_dir, num_envs=1,
                    play=True, seed=args.seed, collision_profile=experiment_metadata(checkpoint).get("collision_profile", "default"))
    original = cfg.commands["motion"]
    command = DeploymentCommandCfg(**{f.name: getattr(original, f.name) for f in fields(original)},
        init_mode=args.init_mode, noise_scale=args.noise_scale,
        start_frame=args.start_frame, random_start=args.random_start)
    command.history_length = checkpoint["policy_config"]["history_length"]
    command.goal_frames = checkpoint["policy_config"]["goal_frames"]
    for field, key in (("object_reward_cfg", "object_reward"), ("reward_cfg", "robot_reward"),
                       ("observation_cfg", "observation"), ("termination_cfg", "termination")):
        if key in task:
            setattr(command, field, replace(getattr(command, field), **task[key]))
    if "control_dt" in task and not math.isclose(task["control_dt"], cfg.decimation * cfg.sim.mujoco.timestep):
        raise ValueError("Checkpoint control timestep differs from this sitting environment")
    return replace(cfg, commands={"motion": command})


def classify_outcome(*, failed, reference_end, timed_out=False, step_limit=False):
    """Failure takes precedence even on the reference's last transition."""
    if failed:
        return "tracking_failure"
    if timed_out:
        return "timeout"
    if reference_end:
        return "reference_complete"
    return "step_limit" if step_limit else None


class DeploymentEnv(RslRlVecEnvWrapper):
    """Seeded trials with terminal-state scoring before any reset."""
    def __init__(self, native, episodes=1, max_steps=None, *, seeds=None,
                 metrics_cfg=None, on_record=None):
        self.records, self.finished = [], False
        self.episodes, self.max_steps = episodes, max_steps
        self.seeds = list(seeds) if seeds is not None else None
        if self.seeds is not None:
            self.episodes = len(self.seeds)
        self.base_seed = int(native.cfg.seed or 0)
        self.on_record = on_record
        self.steps, self.total_reward, self.initial = 0, 0., None
        self._last_result = None
        self._trial_open = False
        self.metrics = None
        super().__init__(native)
        self.metrics = SittingEpisodeMetrics(native, metrics_cfg or SittingEvaluationCfg())
        self.reset()

    def reset(self):
        if self._trial_open and self.steps:
            self._finish("manual_reset")
        # A viewer may request a reset immediately after opening. This is still
        # the same unexecuted trial, with exactly the same seed.
        index = len(self.records)
        if self.seeds is not None and index >= len(self.seeds):
            self.finished = True
            return self.get_observations(), {}
        self.episode_seed = self.seeds[index] if self.seeds is not None else (self.base_seed + index) % 2**32
        observation, extras = self.env.reset(seed=self.episode_seed)
        command = self.unwrapped.command_manager.get_term("motion")
        self.finished = False
        self.steps, self.total_reward = 0, 0.
        self.frame0 = int(command.time_steps[0])
        self.initial = dict(frame=self.frame0, state=command.dynamics.pack(command.state())[0].cpu().tolist())
        self.metrics.reset(self.frame0)
        self._trial_open = True
        return TensorDict(observation, batch_size=[self.num_envs]), extras

    def _finish(self, outcome, *, valid=True, error=None):
        if not self._trial_open:
            return
        native = self.unwrapped
        command = native.command_manager.get_term("motion")
        valid = valid and outcome not in {"interrupted", "viewer_closed", "manual_reset"}
        completed = outcome == "reference_complete"
        full_eligible = self.frame0 == int(command.motion.clip_start[0])
        full_success = valid and full_eligible and completed
        try:
            result = self.metrics.finalize(full_success)
        except Exception as metric_error:
            # An invalid physics buffer or interrupted first step must still
            # leave a seed/denominator record; never mask the original failure.
            valid, full_success = False, False
            error = f"{error or outcome}; metrics: {metric_error}"
            result = dict(stable_contact_success=False, stable_contact_eligible=False,
                          failure_reasons=["metrics_unavailable"], metrics={})
        remaining = int(command.motion.clip_end[0]) - self.frame0
        reasons = list(result.pop("failure_reasons", []))
        if not completed:
            reasons.insert(0, outcome)
        if not full_eligible:
            reasons.append("partial_reference_start")
        if not valid:
            result["stable_contact_success"] = False
        record = dict(episode=len(self.records) + 1, seed=self.episode_seed,
            condition=command.cfg.init_mode, initial=self.initial, steps=self.steps,
            seconds=self.steps * native.step_dt, outcome=outcome, valid=valid,
            reference_completed=completed, full_reference_eligible=full_eligible,
            full_reference_success=full_success,
            normalized_return=self.total_reward / (max(remaining, 1) * command.task.max_step_reward),
            mean_step_reward=self.total_reward / (max(self.steps, 1) * command.task.max_step_reward),
            **result, failure_reasons=list(dict.fromkeys(reasons)))
        if error is not None:
            record["error"] = str(error)
        self.records.append(record)
        self._trial_open = False
        self.finished = self.episodes > 0 and len(self.records) >= self.episodes
        print("[deploy] " + json.dumps({k: v for k, v in record.items() if k not in ("initial", "metrics")}), flush=True)
        if self.on_record is not None:
            self.on_record(record)

    def interrupt(self, outcome="interrupted", *, valid=True, error=None):
        self._finish(outcome, valid=valid, error=error)
        self.finished = True

    def step(self, actions):
        if self.finished:
            return self._last_result
        native = self.unwrapped
        try:
            result = super().step(actions)
            self._last_result = result
            if bool(native.sim.data.overflow.any()):
                raise RuntimeError("Physics buffer overflow; this rollout is invalid")
            command = native.command_manager.get_term("motion")
            if not bool(torch.isfinite(command.dynamics.pack(command.state())).all()):
                raise FloatingPointError("Nonfinite physical state")
            self.steps += 1
            self.total_reward += float(result[1][0])
            self.metrics.update()
            failed = bool(native.reset_terminated[0])
            timed_out = any(bool(native.termination_manager.get_term(name)[0])
                for name in native.termination_manager.active_terms
                if name != "motion_complete" and native.termination_manager.get_term_cfg(name).time_out)
            reference_end = int(command.time_steps[0]) >= int(command.motion.clip_end[0])
            limited = self.max_steps is not None and self.steps >= self.max_steps
            outcome = classify_outcome(failed=failed, reference_end=reference_end,
                                       timed_out=timed_out, step_limit=limited)
            if outcome:
                self._finish(outcome)
                if not self.finished:
                    observation, _ = self.reset()
                    result = (observation, *result[1:])
            return result
        except Exception as exc:
            self.interrupt("simulation_error", valid=False, error=exc)
            raise

    def close_metrics(self):
        if self.metrics is not None:
            self.metrics.close()

    def close(self):
        self.interrupt("viewer_closed")
        self.close_metrics()
        super().close()



def reference_identity(checkpoint, manifest, reference, args):
    clip = load_motion(reference)
    collision_profile = experiment_metadata(checkpoint).get("collision_profile", "default")
    # Validate the same approximation that will be used by physics and the
    # geometry encoder, before allocating CUDA. Compact bundles need no default cache.
    load_collision_parts(clip, args.cache_dir, profile=collision_profile)
    key = shape_key(clip)
    partitions = [p for p in ("train", "test") if key in {g["shape_key"] for g in manifest[p]}]
    digest = file_digest(reference)
    reference_partitions = [p for p in ("train", "test") if digest in {
        c["sha256"] for g in manifest[p] for c in g["clips"]}]
    exposure = experiment_metadata(checkpoint).get("visited_train_shapes", [])
    identity = dict(reference=str(reference), sha256=digest, shape_key=key,
                    collision_profile=collision_profile,
                    geometry_partition=partitions[0] if partitions else "external",
                    reference_partition=reference_partitions[0] if reference_partitions else "external",
                    visited_in_checkpoint=key in exposure)
    return identity


def run_reference(checkpoint, manifest, reference, args, *, on_record=None, metrics_cfg=None):
    identity = reference_identity(checkpoint, manifest, reference, args)
    print("[deploy] " + json.dumps(identity), flush=True)
    native = ManagerBasedRlEnv(make_deployment_cfg(checkpoint, reference, args), device=args.device)
    env = None
    try:
        env = DeploymentEnv(native, args.episodes, args.max_steps,
            seeds=getattr(args, "trial_seeds", None), metrics_cfg=metrics_cfg, on_record=on_record)
        train_cfg = checkpoint["train_cfg"] | dict(amp="none", no_compile=True)
        runner = SuperMimicRunner(env, train_cfg, device=args.device)
        # Explicit frozen-policy transfer, including a new chair's local geometry
        # bank. Full training resume would incorrectly require identical scene IDs.
        load_actor(runner, checkpoint, manifest)
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

                def _handle_gui_reset(self, all_envs=False):
                    # Viser normally bypasses the wrapper for command resets.
                    # Route this through seeded trial bookkeeping instead.
                    self.reset_environment()

            server = None
            try:
                if args.viewer == "viser":
                    import viser
                    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="G1 policy deployment")
                viewer = EpisodeViewer(env, policy, **({"viser_server": server} if server else {}))
                if args.viewer == "viser":
                    # mjviser measures elevation above the horizon; the native
                    # MuJoCo default is negative and would start below the floor.
                    viewer.cfg = replace(viewer.cfg, elevation=20., azimuth=120., distance=2.8)
                viewer.run()
                if viewer._last_error:
                    raise RuntimeError(viewer._last_error)
            finally:
                if server is not None:
                    server.stop()
        if not env.finished:
            env.interrupt("viewer_closed")
        if policy_digest(runner.pi) != before:
            raise RuntimeError("Policy weights or normalization changed during deployment")
        assert runner.wm is runner.critic is runner.buffer is None
        return dict(**identity, frozen_policy=True, episodes=env.records)
    except BaseException as exc:
        if env is not None:
            env.interrupt("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "evaluation_error",
                          valid=isinstance(exc, (KeyboardInterrupt, SystemExit)), error=exc)
        raise
    finally:
        if env is not None:
            env.close_metrics()
        native.close()


def metric_configuration(path):
    if path is None:
        return {"defaults": {}, "references": {}}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) - {"defaults", "references"}:
        raise ValueError("metrics-json must contain defaults and/or references objects")
    defaults, references = payload.get("defaults", {}), payload.get("references", {})
    if not isinstance(defaults, dict) or not isinstance(references, dict):
        raise ValueError("metrics-json defaults/references must be objects")
    SittingEvaluationCfg(**defaults)
    for override in references.values():
        if not isinstance(override, dict):
            raise ValueError("Each metrics-json reference override must be an object")
        SittingEvaluationCfg(**(defaults | override))
    return {"defaults": defaults, "references": references}


def reference_metrics(configuration, identity):
    override = configuration["references"].get(identity["sha256"],
        configuration["references"].get(Path(identity["reference"]).name, {}))
    return SittingEvaluationCfg(**(configuration["defaults"] | override))


def validate_arguments(args):
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid viewer port")
    if (args.index < 0 or args.episodes < 0 or args.start_frame < 0
            or not math.isfinite(args.noise_scale) or args.noise_scale < 0
            or (args.max_steps is not None and args.max_steps < 1)):
        raise ValueError("Invalid index, episode count, frame, noise scale or step limit")
    if args.browse and args.evaluate:
        raise ValueError("Choose --browse or --evaluate")
    if args.report and not args.browse:
        raise ValueError("--report is for the browser; use --output to save an evaluation")
    if args.evaluate:
        if args.start_frame != 0 or args.random_start:
            raise ValueError("Full-reference evaluation always starts at frame 0")
        if args.episodes == 0:
            raise ValueError("Evaluation requires a finite episode count")
        if args.index:
            raise ValueError("Evaluation covers all selected references; select a file with --reference")
        args.viewer = "none"
        args.all = True
    if args.seeds is not None:
        if args.episodes == 0:
            raise ValueError("Explicit seeds require a finite trial schedule")
        args.episodes = len(args.seeds)
    if args.viewer == "none" and args.episodes == 0 and not args.browse:
        raise ValueError("Headless runs require a finite --episodes count")
    if args.all and args.viewer != "none" and not args.browse:
        raise ValueError("--all requires --viewer none; use --index to view one reference")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Deployment uses one process; do not launch with torchrun")
    # Validate even infinite playback's base seed before creating any resources.
    planned_seeds(args.seed, max(args.episodes, 1), args.seeds)


def runtime_provenance():
    versions = {}
    for package in ("torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "supermimic"], cwd=root)
    except (OSError, subprocess.CalledProcessError):
        revision, diff = None, b""
    sources = hashlib.sha256()
    for path in sorted((root / "supermimic").rglob("*.py")):
        sources.update(str(path.relative_to(root)).encode())
        sources.update(path.read_bytes())
    return dict(packages=versions, git_revision=revision,
                tracked_diff_sha256=hashlib.sha256(diff).hexdigest(), source_sha256=sources.hexdigest())


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    configuration = metric_configuration(args.metrics_json)
    manifest, root = load_split(args.split or args.checkpoint.parent / "split.json", args.data_root)
    paths = resolve_references(args, manifest, root)
    if args.list:
        for i, path in enumerate(paths):
            print(f"{i:3d}  {path.name}")
        return
    if not paths or args.index >= len(paths):
        raise ValueError(f"Reference index {args.index} out of range (found {len(paths)})")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if experiment_metadata(checkpoint).get("split_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Checkpoint and training split fingerprints differ")
    if args.browse:
        from .policy_browser import run_browser
        return run_browser(args, manifest, paths, checkpoint["iteration"])
    if args.evaluate and args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = Path("results/evaluation") / f"{args.checkpoint.stem}_{stamp}.json"
    if args.output and args.output.exists():
        raise FileExistsError(f"Report already exists; choose a new --output: {args.output}")
    conditions = list(dict.fromkeys(args.conditions)) if args.evaluate else [args.init_mode]
    seeds = planned_seeds(args.seed, args.episodes, args.seeds) if args.episodes else []
    from supermimic.mdp.events import ReferenceResetCfg
    report = dict(schema_version=2, task="g1_sitting", status="running",
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_digest(args.checkpoint),
        iteration=checkpoint["iteration"], split_fingerprint=manifest["fingerprint"],
        runtime=runtime_provenance(),
        protocol=dict(evaluation=bool(args.evaluate), conditions=conditions,
            seeds={condition: seeds.copy() for condition in conditions},
            seed_pairing="Same explicit seed schedule for every reference and condition; reset reseeds each episode",
            start_frame=args.start_frame, random_start=args.random_start,
            perturbation=dict(type="initial_robot_state_only", noise_scale=args.noise_scale,
                              uniform_half_widths=asdict(ReferenceResetCfg())),
            metrics=configuration, control_hz=50, physics_hz=200,
            evaluation_physics=dict(chair_mass_kg=10., robot_chair_friction=1.),
            max_steps=args.max_steps, infinite_playback=args.episodes == 0,
            success_definition="Full reference from frame 0, no failure/timeout/interruption before reference completion",
            clean_replicates="Nominal initial state is unchanged; repeated clean seeds are not independent physical variation",
            termination=checkpoint.get("task", {}).get("termination"),
            aggregation="Rates use planned trials; chair macro averages first combine references sharing a shape"),
        references=[])
    selected = paths if args.all else [paths[args.index]]
    # Persist the entire planned roster before running: an interrupted run must
    # not silently drop unexecuted references or seeds from the denominators.
    for reference in selected:
        identity = reference_identity(checkpoint, manifest, reference, args)
        report["references"].append(dict(**identity,
            metric_config=asdict(reference_metrics(configuration, identity)),
            conditions={condition: dict(planned_seeds=seeds.copy(), episodes=[], status="pending")
                        for condition in conditions}))

    def save_report():
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        report["summary"] = build_report_summary(report)
        if args.output:
            atomic_write_report(args.output, report)

    def terminate(signum, frame):
        raise KeyboardInterrupt("Deployment stopped")

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    save_report()
    try:
        for entry in report["references"]:
            for condition in conditions:
                current = entry["conditions"][condition]
                current["status"] = "running"
                save_report()
                trial_args = argparse.Namespace(**vars(args))
                trial_args.init_mode, trial_args.trial_seeds = condition, seeds or None

                def record_episode(record):
                    if args.episodes == 0:
                        current["planned_seeds"].append(record["seed"])
                    current["episodes"].append(record)
                    save_report()

                result = run_reference(checkpoint, manifest, Path(entry["reference"]), trial_args,
                    on_record=record_episode, metrics_cfg=SittingEvaluationCfg(**entry["metric_config"]))
                current["frozen_policy"] = result["frozen_policy"]
                interrupted = any(r["outcome"] in {"interrupted", "viewer_closed", "manual_reset"}
                                  for r in current["episodes"])
                missing = args.episodes > 0 and len(current["episodes"]) < len(current["planned_seeds"])
                current["status"] = ("interrupted" if interrupted or missing else
                    "complete" if all(r["valid"] for r in current["episodes"]) else "invalid")
                save_report()
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        statuses = [value["status"] for entry in report["references"] for value in entry["conditions"].values()]
        report["status"] = "complete" if all(status == "complete" for status in statuses) else "partial"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if "current" in locals():
            current["status"] = report["status"]
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        save_report()
    if args.output:
        print(f"[evaluation] report={args.output.resolve()}", flush=True)
    for partition, values in report["summary"]["task"]["geometry_partitions"].items():
        for condition, stats in values.items():
            if stats["planned_trials"]:
                compact = {key: stats[key] for key in (
                    "planned_trials", "executed_trials", "invalid_trials",
                    "full_reference_success_rate", "stable_contact_success_rate",
                    "chair_macro_full_reference_success_rate", "chair_macro_stable_contact_success_rate")}
                print(f"[evaluation] {partition}/{condition} " + json.dumps(compact), flush=True)
    return report


if __name__ == "__main__":
    main()
