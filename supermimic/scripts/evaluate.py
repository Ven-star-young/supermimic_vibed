"""Quantitative whole-library evaluation using the registered mjlab environment.

Interactive playback belongs to mjlab's ``play``. This script scores each motion
from its first frame, writes per-clip results, and optionally records trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict, replace

from supermimic.tasks import TASK_ID  # Registers the external task.
from supermimic.rotations import mat_to_quat


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--checkpoint-file", "--ckpt", dest="checkpoint_file")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--motion-dir", type=Path, help="Override the task's motion directory (or a single interaction NPZ)")
    parser.add_argument("--object-motion-file", type=str, help="Select the sitting scene's object geometry")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--every", type=int, default=1, help="evaluate every Nth saved checkpoint")
    parser.add_argument("--record", type=Path, help="save robot and task-object trajectory NPZ (requires --checkpoint-file)")
    parser.add_argument("--door-sweep", action="store_true", help="Evaluate one door reference over matched physics samples")
    parser.add_argument("--physics-config", type=Path, help="JSON task.door_randomization_cfg ranges (default: checkpoint)")
    parser.add_argument("--random-samples", type=int, default=128, help="Uniform joint physics samples, in addition to nominal and boundaries")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="Required for --door-sweep; stores protocol, per-trial results and trajectories")
    return parser


def _policy_config(checkpoint):
    if "policy_config" in checkpoint:
        return checkpoint["policy_config"]
    legacy = checkpoint.get("args", {})
    return {"history_length": legacy.get("hist", 10), "goal_frames": legacy.get("goal_frames", 10),
            "amp": legacy.get("amp", "none")}


def _record_state(trajectory, command, alive, state=None):
    """Record the initial and post-step states, including the terminal frame."""
    state = command.state() if state is None else state
    values = dict(root_p=state[0], root_quat=mat_to_quat(state[1]), q=state[4],
                  f=command.time_steps, alive=alive)
    if hasattr(command.motion, "object_ref_state"):
        values.update(object_p=state[6], object_quat=mat_to_quat(state[7]))
    if "door" in command.scene.entities:
        values.update(theta=state[6].squeeze(-1), omega=state[7].squeeze(-1))
    for key, value in values.items():
        trajectory.setdefault(key, []).append(value.cpu().clone())


@torch.no_grad()
def evaluate(env, runner, record=None, metrics=None):
    """One world per clip; score only until its first termination or clip end."""
    native = env.unwrapped
    command = native.command_manager.get_term("motion")
    task, motion = command.task, command.motion
    if native.cfg.auto_reset:
        raise ValueError("Evaluation requires auto_reset=False to preserve terminal states")
    policy = runner.get_inference_policy(device=native.device)
    ids = torch.arange(native.num_envs, device=native.device)
    if native.num_envs != len(motion.names):
        raise ValueError("Whole-library evaluation requires one environment per motion")
    command.set_reset_frames(ids, motion.clip_start)
    obs, _ = native.reset()
    returns = torch.zeros(native.num_envs, device=native.device)
    finished = torch.zeros(native.num_envs, dtype=torch.bool, device=native.device)
    failed = torch.zeros_like(finished)
    completed = torch.zeros_like(finished)
    trajectory = {}
    object_enabled = hasattr(motion, "object_ref_state")
    door_enabled = "door" in command.scene.entities
    if door_enabled:
        door_error = torch.zeros(native.num_envs, 2, device=native.device)
        door_steps = torch.zeros(native.num_envs, device=native.device)
        final_door = torch.zeros_like(door_error)
    if record is not None:
        _record_state(trajectory, command, ~finished)
    while not bool(finished.all()):
        alive = ~finished
        if metrics is not None:
            metrics.begin_step(alive)
        obs, reward, terminated, truncated, _ = native.step(policy(obs))
        returns += torch.where(alive, reward, 0)
        failed |= alive & terminated
        reached_end = command.time_steps >= motion.clip_end
        completed |= alive & reached_end & ~terminated
        finished |= terminated | truncated | reached_end
        state = command.state() if door_enabled or record is not None else None
        if door_enabled:
            actual = torch.cat(state[6:8], -1)
            reference = torch.stack(motion.door_ref_state(command.time_steps), -1)
            door_error += torch.where(alive[:, None], (actual - reference).square(), 0)
            door_steps += alive
            final_door = torch.where(alive[:, None], actual, final_door)
        if record is not None:
            # ``alive`` describes membership in the recorded episode, including
            # its last state. Later states of finished worlds are masked out.
            _record_state(trajectory, command, alive, state)
    scores = returns / ((motion.clip_len - 1).float() * task.max_step_reward)
    result = {
        "norm_return": scores.mean().item(),
        "norm_return_framew": (returns.sum() / ((motion.clip_len - 1).sum() * task.max_step_reward)).item(),
        "n_failed": int(failed.sum()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": {name: float(score) for name, score in zip(motion.names, scores.tolist())},
    }
    if door_enabled:
        rmse = (door_error / door_steps.clamp_min(1)[:, None]).sqrt()
        aggregate = (door_error.sum(0) / door_steps.sum().clamp_min(1)).sqrt()
        result["door_tracking"] = {
            "scope": "executed_transitions_including_terminal_state; fixed_reference_phase",
            "angle_rmse_rad": float(aggregate[0]),
            "angular_velocity_rmse_radps": float(aggregate[1]),
            "reference_completion_rate": float(completed.float().mean()),
            "n_reference_completed": int(completed.sum()),
            "per_clip": {
                name: dict(angle_rmse_rad=float(rmse[i, 0]),
                           angular_velocity_rmse_radps=float(rmse[i, 1]),
                           reference_completed=bool(completed[i]),
                           observed_transitions=int(door_steps[i]),
                           final_theta_rad=float(final_door[i, 0]),
                           final_omega_radps=float(final_door[i, 1]),
                           ref_final_theta_rad=float(motion.door_angle[motion.clip_end[i]]),
                           ref_final_omega_radps=float(motion.door_omega[motion.clip_end[i]]))
                for i, name in enumerate(motion.names)
            },
        }
    if metrics is not None:
        contact_report = metrics.report()
        contact_report["per_clip"] = dict(zip(motion.names, contact_report.pop("per_env")))
        result["door_contact"] = contact_report
    if record is not None:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        object_reference = (dict(ref_object_p=motion.object_p.cpu().numpy(),
                                 ref_object_quat=mat_to_quat(motion.object_R).cpu().numpy())
                            if object_enabled else {})
        if door_enabled:
            object_reference.update(ref_theta=motion.door_angle.cpu().numpy(),
                                    ref_omega=motion.door_omega.cpu().numpy())
        np.savez_compressed(record, names=np.array(motion.names), **object_reference,
                            **{k: torch.stack(v).numpy() for k, v in trajectory.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(),
                            ref_q=motion.q.cpu().numpy(), clip_start=motion.clip_start.cpu().numpy(),
                            clip_len=motion.clip_len.cpu().numpy())
    return result


def _restore_open_door_config(cfg, checkpoint):
    """Restore saved task parameters before the asset factory builds the scene.

    Mutate the existing command: the scene's lazy spec factory holds that same
    object. Motion/XML selection remains caller-owned, and runner.load still
    checks the complete task schema against the resulting environment.
    """
    from supermimic.tasks.open_door.commands import DoorCommandCfg
    from supermimic.tasks.open_door_side.commands import SideDoorCommandCfg

    command, metadata = cfg.commands["motion"], checkpoint.get("task") or {}
    if not isinstance(command, DoorCommandCfg):
        return
    expected = "g1_open_door_side" if isinstance(command, SideDoorCommandCfg) else "g1_open_door"
    if metadata.get("name") != expected:
        raise ValueError(f"OpenDoor checkpoint task mismatch: expected {expected}, got {metadata.get('name')}")
    if metadata.get("schema_version") != 2:
        raise ValueError("OpenDoor checkpoint schema mismatch: fixed-physics checkpoints cannot "
                         "implicitly adopt the new door randomization distribution")
    physical_fields = {"door_mass", "door_damping", "door_friction", "robot_door_friction",
                       "door_stiffness", "door_spring_reference"}
    physics = metadata.get("door_physics", {})
    if set(physics) != physical_fields:
        raise ValueError("OpenDoor checkpoint door_physics fields do not match the task schema")
    updates = dict(physics)
    updates["actor_observation"] = metadata.get("actor_observation_mode", "state")
    tactile = metadata.get("tactile")
    updates["use_tactile"] = tactile is not None
    if tactile is not None:
        for key, field in (("layout", "tactile_layout"), ("contact_distance_m", "tactile_contact_distance_m"),
                           ("full_scale_depth_m", "tactile_counts_full_scale_depth_m"),
                           ("exponent", "tactile_counts_exponent")):
            updates[field] = tactile[key]
    for saved, field in (("door_randomization", "door_randomization_cfg"),
                         ("door_reward", "door_reward_cfg"), ("robot_reward", "reward_cfg"),
                         ("observation", "observation_cfg"), ("termination", "termination_cfg")):
        current, values = getattr(command, field), metadata.get(saved, {})
        if isinstance(command, SideDoorCommandCfg) and saved == "door_reward":
            # Schema-2 side checkpoints predating clearance avoidance used zero.
            values = {"illegal_margin": 0.0, **values}
        if set(values) != set(asdict(current)):
            raise ValueError(f"OpenDoor checkpoint {saved} fields do not match the task schema")
        updates[field] = replace(current, **values)
    restored = replace(command, **updates)  # Validate physical configuration before mutating it.
    for field in updates:
        setattr(command, field, getattr(restored, field))
    if tactile is not None:
        for sensor in cfg.scene.sensors:
            if sensor.name == "tactile":
                sensor.layout = command.tactile_layout
                sensor.contact_distance_m = command.tactile_contact_distance_m
                sensor.counts_full_scale_depth_m = command.tactile_counts_full_scale_depth_m
                sensor.counts_exponent = command.tactile_counts_exponent


def _restore_ledge_config(cfg, checkpoint):
    """Restore the saved ledge reward/physics before building the static scene."""
    from supermimic.tasks.ledge.commands import LedgeCommandCfg

    command, metadata = cfg.commands["motion"], checkpoint.get("task") or {}
    if not isinstance(command, LedgeCommandCfg):
        return
    if metadata.get("name") != "g1_ledge" or metadata.get("schema_version") != 1:
        raise ValueError("Ledge evaluation requires a g1_ledge schema-1 checkpoint")
    updates = {}
    for saved, field in (("ledge_reward", "ledge_reward_cfg"), ("robot_reward", "reward_cfg"),
                         ("observation", "observation_cfg"), ("robot_physics", "physical_cfg")):
        current, values = getattr(command, field), metadata.get(saved, {})
        if not isinstance(values, dict) or set(values) != set(asdict(current)):
            raise ValueError(f"Ledge checkpoint {saved} fields do not match the task schema")
        updates[field] = replace(current, **values)
    # The scene's lazy spec factory intentionally shares this command object.
    for field, value in updates.items():
        setattr(command, field, value)


def _evaluation_config(checkpoint, args):
    policy_cfg = _policy_config(checkpoint)
    cfg = load_env_cfg(args.task, play=True)
    cfg.auto_reset = False
    _restore_open_door_config(cfg, checkpoint)
    _restore_ledge_config(cfg, checkpoint)
    if args.motion_dir is not None:
        cfg.commands["motion"].motion_dir = str(args.motion_dir)
    if args.object_motion_file is not None:
        if not hasattr(cfg.commands["motion"], "object_motion_file"):
            raise ValueError("--object-motion-file requires an interaction task")
        cfg.commands["motion"].object_motion_file = args.object_motion_file
    cfg.commands["motion"].history_length = policy_cfg["history_length"]
    cfg.commands["motion"].goal_frames = policy_cfg["goal_frames"]
    if hasattr(cfg.commands["motion"], "door_xml"):
        from supermimic.tasks.open_door.assets import select_motion_files
        cfg.scene.num_envs = len(select_motion_files(cfg.commands["motion"]))
    elif hasattr(cfg.commands["motion"], "object_motion_file"):
        from supermimic.tasks.sitting.assets import select_motion_files
        cfg.scene.num_envs = len(select_motion_files(cfg.commands["motion"]))
    else:
        motion_path = Path(cfg.commands["motion"].motion_dir)
        cfg.scene.num_envs = 1 if motion_path.is_file() else len(list(motion_path.glob("*.npz")))
    if not cfg.scene.num_envs:
        raise ValueError(f"No standard motion NPZ files found in {cfg.commands['motion'].motion_dir}")
    return cfg


def score_checkpoint(path: Path, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _evaluation_config(checkpoint, args)
    native = ManagerBasedRlEnv(cfg, device=args.device)
    metrics = None
    try:
        if "door" in native.scene.entities:
            from supermimic.tasks.open_door.evaluation import DoorContactMetrics
            from supermimic.tasks.open_door_side.commands import SideDoorCommandCfg
            from supermimic.tasks.open_door_side.evaluation import SideDoorContactMetrics
            metrics_cls = SideDoorContactMetrics if isinstance(cfg.commands["motion"], SideDoorCommandCfg) else DoorContactMetrics
            metrics = metrics_cls(native)
        env = RslRlVecEnvWrapper(native)
        runner = load_runner_cls(args.task)(env, asdict(load_rl_cfg(args.task)), device=args.device)
        runner.load(str(path), load_cfg={"actor": True}, map_location=args.device)
        result = evaluate(env, runner, args.record, metrics)
        result.update(checkpoint=str(path), iteration=checkpoint.get("iteration"),
                      train_wall_h=checkpoint.get("wall_time", 0) / 3600,
                      saved_at=checkpoint.get("saved_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))),
                      eval_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        path.with_name(path.stem + "_eval.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}, indent=2), flush=True)
        return result
    finally:
        if metrics is not None:
            metrics.close()
        native.close()


def _door_scenarios(command, args):
    from supermimic.tasks.open_door.events import DoorRandomizationCfg

    fields = dict(mass="door_mass", damping="door_damping", hinge_friction="door_friction",
                  stiffness="door_stiffness", robot_friction="robot_door_friction")
    ranges = command.door_randomization_cfg
    if args.physics_config is not None:
        ranges = DoorRandomizationCfg(**json.loads(args.physics_config.read_text())["task"]["door_randomization_cfg"])
    nominal = {key: getattr(command, field) for key, field in fields.items()}
    scenarios = [dict(name="nominal", group="nominal", physics=nominal)]
    for key in fields:
        for label, value in zip(("low", "high"), getattr(ranges, key)):
            scenarios.append(dict(name=f"{key}_{label}", group="one_factor", physics={**nominal, key: value}))
    for i, values in enumerate(itertools.product(*(getattr(ranges, key) for key in fields))):
        scenarios.append(dict(name=f"corner_{i:02d}", group="corners", physics=dict(zip(fields, values))))
    rng = np.random.default_rng(args.seed)
    bounds = np.array([getattr(ranges, key) for key in fields])
    for i, values in enumerate(rng.uniform(bounds[:, 0], bounds[:, 1], size=(args.random_samples, len(fields)))):
        scenarios.append(dict(name=f"random_{i:04d}", group="random", physics=dict(zip(fields, values.tolist()))))
    return scenarios


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_door_sweep(paths, args):
    """Reuse one scene across checkpoints; preserve an auditable paired protocol."""
    from motion_data import load_motion
    from supermimic.rl.evaluation_report import atomic_write_report
    from supermimic.tasks.open_door.assets import door_xml_path, select_motion_files
    from supermimic.tasks.open_door.commands import DoorCommandCfg
    from supermimic.tasks.open_door.evaluation import evaluate_door_sweep

    if args.output_dir is None or args.motion_dir is None:
        raise ValueError("--door-sweep requires explicit --output-dir and --motion-dir")
    if args.watch or args.record is not None:
        raise ValueError("Door sweeps save trajectories automatically; omit --watch and --record")
    if args.random_samples < 1 or not 0 <= args.seed < 2**32:
        raise ValueError("Require --random-samples >= 1 and seed in [0, 2**32)")
    if not paths:
        raise ValueError("No checkpoints found")
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False, mmap=True)
    cfg = _evaluation_config(checkpoint, args)
    command = cfg.commands["motion"]
    if not isinstance(command, DoorCommandCfg) or len(select_motion_files(command)) != 1:
        raise ValueError("Door sweeps require one OpenDoor/OpenDoorSide reference NPZ")
    scenarios = _door_scenarios(command, args)
    cfg.scene.num_envs, cfg.seed = len(scenarios), args.seed
    cfg.events["door_physics"].params["physics"] = {
        key: [sample["physics"][key] for sample in scenarios] for key in scenarios[0]["physics"]}
    motion_path, xml_path = select_motion_files(command)[0], door_xml_path(command)
    clip = load_motion(motion_path)
    cfg.episode_length_s = max(cfg.episode_length_s, (len(clip["root_pos"]) - 1) / float(clip["fps"]) + 1)
    import inspect
    import mujoco
    from supermimic.tasks.open_door import events, evaluation
    protocol = dict(schema_version=1, task=args.task, seed=args.seed, scenarios=scenarios,
                    reference=str(motion_path.resolve()), reference_sha256=_sha256(motion_path),
                    door_xml=str(xml_path.resolve()), door_xml_sha256=_sha256(xml_path),
                    checkpoint_task=checkpoint["task"], policy_config=_policy_config(checkpoint),
                    initial_state="exact reference first frame; no reset/observation/action noise",
                    robot_physics=asdict(command.physical_cfg), physics_source=(str(args.physics_config.resolve())
                    if args.physics_config is not None else "checkpoint door_randomization"),
                    device=args.device, torch_version=torch.__version__, mujoco_version=mujoco.__version__,
                    code_sha256={Path(file).name: _sha256(file) for file in
                                 (__file__, inspect.getfile(events), inspect.getfile(evaluation))})
    # Normalize tuples to lists before comparing a previously saved protocol.
    protocol = json.loads(json.dumps(protocol))
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f"Existing protocol differs: use a new --output-dir ({protocol_path})")
    atomic_write_report(protocol_path, protocol)
    protocol_hash = _sha256(protocol_path)
    del checkpoint, clip
    native = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env = RslRlVecEnvWrapper(native)
        runner = load_runner_cls(args.task)(env, asdict(load_rl_cfg(args.task)), device=args.device)
        results = []
        for path in paths:
            result_path = destination / f"{path.stem}_eval.json"
            checkpoint_hash = _sha256(path)
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["protocol_sha256"] != protocol_hash or result["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError(f"Saved result identity differs: {result_path}")
                print(f"Already evaluated: {path.name}", flush=True)
            else:
                started = time.monotonic()
                runner.load(str(path), load_cfg={"actor": True}, map_location=args.device)
                native.seed(args.seed)
                result = evaluate_door_sweep(native, runner, scenarios,
                                             record=destination / f"{path.stem}_trajectories.npz")
                result.update(checkpoint=str(path.resolve()), checkpoint_sha256=checkpoint_hash,
                              iteration=int(re.search(r"_(\d+)\.pt$", path.name).group(1)),
                              protocol_sha256=protocol_hash, elapsed_seconds=time.monotonic() - started,
                              eval_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                atomic_write_report(result_path, result)
                print(f"Evaluated {path.name} in {result['elapsed_seconds']:.1f}s: "
                      f"{json.dumps(result['summaries'])}", flush=True)
            results.append(result)
            rows, summary_rows = [], []
            # A single-checkpoint rerun must retain the other saved results.
            saved = [json.loads(file.read_text()) for file in destination.glob("*_eval.json")]
            for item in sorted(saved, key=lambda value: value["iteration"]):
                if item["protocol_sha256"] != protocol_hash:
                    raise ValueError("Output directory contains results from a different protocol")
                for trial in item["per_trial"]:
                    rows.append(dict(checkpoint=Path(item["checkpoint"]).name, iteration=item["iteration"],
                                     **{key: value for key, value in trial.items() if key != "physics"},
                                     **trial["physics"]))
                for group, summary in item["summaries"].items():
                    summary_rows.append(dict(iteration=item["iteration"], group=group,
                        **{key: value for key, value in summary.items() if not isinstance(value, dict)},
                        **{f"{scope}_{key}": value for scope in ("executed_prefix", "completed_only")
                           for key, value in summary[scope].items()}))
            for name, table in (("trials.csv", rows), ("summary.csv", summary_rows)):
                temporary = destination / f".{name}.tmp"
                with temporary.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                    writer.writeheader()
                    writer.writerows(table)
                temporary.replace(destination / name)
        return results
    finally:
        native.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.every < 1:
        raise ValueError("--every must be positive")
    if args.door_sweep:
        if args.checkpoint_file:
            paths = [Path(args.checkpoint_file)]
        elif args.run_dir is not None:
            pattern = re.compile(r"(?:model|ckpt)_(\d+)\.pt$")
            paths = sorted((p for p in args.run_dir.glob("*.pt") if pattern.fullmatch(p.name)),
                           key=lambda p: int(pattern.fullmatch(p.name).group(1)))[::args.every]
        else:
            raise ValueError("Provide --checkpoint-file or --run-dir")
        return score_door_sweep(paths, args)
    if args.checkpoint_file:
        return score_checkpoint(Path(args.checkpoint_file), args)
    if args.run_dir is None:
        raise ValueError("Provide --checkpoint-file or --run-dir")
    if args.record is not None:
        raise ValueError("--record requires --checkpoint-file")
    output = args.run_dir / "eval.csv"
    done = set()
    if output.exists():
        with output.open() as stream:
            done = {row["checkpoint"] for row in csv.DictReader(stream)}
    else:
        with output.open("w") as stream:
            csv.writer(stream).writerow(("checkpoint", "iteration", "train_wall_h", "saved_at", "eval_at",
                                        "norm_return", "norm_return_framew", "n_failed", "failed"))
    pattern = re.compile(r"(?:model|ckpt)_(\d+)\.pt$")
    while True:
        paths = sorted((p for p in args.run_dir.glob("*.pt") if pattern.fullmatch(p.name)),
                       key=lambda p: int(pattern.fullmatch(p.name).group(1)))
        for path in paths[::args.every]:
            if path.name in done:
                continue
            result = score_checkpoint(path, args)
            with output.open("a") as stream:
                csv.writer(stream).writerow((path.name, result["iteration"], result["train_wall_h"],
                                            result["saved_at"], result["eval_at"], result["norm_return"],
                                            result["norm_return_framew"], result["n_failed"], "|".join(result["failed"])))
            done.add(path.name)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
