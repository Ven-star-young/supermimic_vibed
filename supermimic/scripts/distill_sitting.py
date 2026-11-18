"""Distill the sitting expert into a single-frame depth and proprioception student."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from motion_data.chair_split import load_split, scenes
from supermimic.camera import HDMI_CAMERA_METADATA, configure_depth_camera, depth_observation
from supermimic.distillation.sitting import (
    DaggerCoordinator, FrozenSittingExpert, SittingDaggerConfig, SittingStudentPolicy,
)
from supermimic.scripts.play_policy import policy_digest
from supermimic.tasks.sitting.models import SittingPolicy


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/sitting_dagger_depth.json"


def validate_inputs(config):
    """Check teacher/split provenance and training assets without starting a GPU scene."""
    from supermimic.rl.geometry import GeometryBank
    from supermimic.tasks.sitting.experiment import preflight

    checkpoint = torch.load(config.expert_checkpoint, map_location="cpu", weights_only=False)
    manifest, root = load_split(config.split, config.data_root)
    task, policy = checkpoint.get("task", {}), checkpoint.get("policy_config", {})
    dimensions = task.get("dimensions", {})
    if (task.get("name"), task.get("schema_version"), task.get("control_dt")) != ("g1_sitting", 4, .02):
        raise ValueError("Distillation requires a schema-4 sitting expert with 50 Hz control")
    if (dimensions.get("observation"), dimensions.get("goal_per_frame"), dimensions.get("action")) != (102, 73, 29):
        raise ValueError("Teacher must use observation102/goal73-per-frame/action29")
    if (policy.get("history_length"), policy.get("goal_frames")) != (10, 10):
        raise ValueError("Teacher must use ten history frames and ten robot reference frames")
    protocol = (checkpoint.get("infos") or {}).get("sitting_experiment", {})
    expected = dict(split_fingerprint=manifest["fingerprint"], collision_profile=config.collision_profile,
                    scene_distribution="fixed_balanced_per_environment",
                    train_shapes=[group["shape_key"] for group in manifest["train"]],
                    test_shapes=[group["shape_key"] for group in manifest["test"]])
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Teacher sitting experiment mismatch: {key}")
    if not set(protocol.get("visited_train_shapes", [])) <= set(expected["train_shapes"]):
        raise ValueError("Teacher training includes a chair outside the training split")
    if config.num_envs < len(scenes(manifest, "train")):
        raise ValueError("num_envs must cover every training chair/scale variant")
    bank = GeometryBank.from_dict(checkpoint["geometry_bank"])
    teacher = FrozenSittingExpert(SittingPolicy(bank))
    teacher.load_checkpoint(checkpoint)
    preflight(manifest, root, config.collision_cache_dir, "train", config.collision_profile)
    return checkpoint, manifest, root


def make_env_cfg(config, manifest, root):
    from supermimic.tasks.sitting.experiment import mixed_scene_cfg

    cfg = mixed_scene_cfg(manifest, root, config.collision_cache_dir, num_envs=config.num_envs,
                          seed=config.seed, assignment_seed=config.seed,
                          collision_profile=config.collision_profile)
    command = cfg.commands["motion"]
    command.history_length, command.goal_frames, command.sampling_mode = 10, 10, "adaptive"
    cfg.auto_reset = False
    return configure_depth_camera(cfg)


@torch.no_grad()
def collect(env, coordinator, steps, beta, *, episode_returns=None):
    """Snapshot pre-action inputs; carry RSI episode returns across collection rounds."""
    teacher_count = episodes = 0
    command = env.command_manager.get_term("motion")
    gamma = command.cfg.gamma
    if episode_returns is None:
        episode_returns = {key: torch.zeros(env.num_envs, device=env.device)
                           for key in ("discounted", "total", "steps")}
    first_step = env.common_step_counter
    for _ in range(steps):
        observations = env.observation_manager.compute()
        step = coordinator.label_and_select(
            observations["policy_history"], observations["goal"], beta,
            depth=depth_observation(env), geometry_ids=command.task.geometry_ids(observations["context"]))
        if not torch.isfinite(step.executed_action).all() or not torch.isfinite(step.expert_action).all():
            raise FloatingPointError("Nonfinite expert/student action during distillation")
        _, reward, terminated, truncated, _ = env.step(step.executed_action)
        if bool(env.sim.data.overflow.any()):
            raise RuntimeError("Physics buffer overflow during distillation")
        if not torch.isfinite(command.dynamics.pack(command.state())).all() or not torch.isfinite(reward).all():
            raise FloatingPointError("Nonfinite physics during distillation")
        done = terminated | truncated | (command.time_steps >= command.motion.clip_end[command.motion.clip_id[command.time_steps]])
        coordinator.record(step, done)
        episode_returns["discounted"] += gamma ** episode_returns["steps"] * reward
        episode_returns["total"] += reward
        episode_returns["steps"] += 1
        teacher_count += int(step.expert_mask.sum())
        ids = done.nonzero().flatten()
        episodes += len(ids)
        if len(ids):
            length = episode_returns["steps"][ids]
            bootstrap = torch.where(terminated[ids], 0.0,
                                    gamma ** length * (episode_returns["total"][ids] / length) / (1 - gamma))
            command.curriculum.update_returns(ids, episode_returns["discounted"][ids] + bootstrap)
            for value in episode_returns.values():
                value[ids] = 0
            # Native reset refreshes geometry and sensor output before the next label.
            env.reset(env_ids=ids)
    return dict(teacher_fraction=teacher_count / (steps * env.num_envs), episodes=episodes,
                first_control_step=first_step, last_control_step=env.common_step_counter - 1,
                simulation_seconds=steps * env.step_dt)


def save_student(path, coordinator, checkpoint, protocol, metrics, expert_digest):
    config = coordinator.config
    torch.save({
        "kind": "sitting_depth_student", "schema_version": 1,
        "student": coordinator.student.state_dict(),
        "policy_config": {"history_length": 10, "goal_frames": 10, "proprio_dim": 93,
                          "goal_dim": 730, "depth_shape": [1, 120, 160], "depth_latent_dim": 64,
                          "action_dim": 29, "history_order": "oldest_to_newest", "depth_units": "metres"},
        "config": config.to_dict(), "task": checkpoint["task"],
        "expert_checkpoint": str(Path(config.expert_checkpoint).resolve()),
        "expert_iteration": checkpoint.get("iteration"), "expert_policy_digest": expert_digest,
        "expert_policy_config": checkpoint["policy_config"],
        "camera_metadata": HDMI_CAMERA_METADATA,
        "goal_encoder": {
            "initialized_from_teacher": bool(coordinator.student.goal_initialized_from_teacher),
            "encoder_parameters_trainable_after_initialization": True,
            "normalizer_statistics": "teacher_then_bc_initialization",
        },
        "infos": {"sitting_experiment": protocol},
        "iteration": int(metrics["round"]) + 1, "metrics": metrics,
    }, path)


def train(config, checkpoint, manifest, root):
    from mjlab.envs import ManagerBasedRlEnv

    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    torch.manual_seed(config.seed)
    env = ManagerBasedRlEnv(make_env_cfg(config, manifest, root), device=config.device)
    try:
        command = env.command_manager.get_term("motion")
        command.task.validate_checkpoint(checkpoint)
        expert = FrozenSittingExpert(SittingPolicy(command.geometry_bank).to(config.device))
        expert.load_checkpoint(checkpoint)
        student = SittingStudentPolicy().to(config.device)
        if config.initialize_goal_from_teacher:
            student.initialize_goal_from_expert(expert.expert)
        coordinator = DaggerCoordinator(expert, student, config=config, device=config.device)
        env.reset()
        episode_returns = {key: torch.zeros(env.num_envs, device=env.device)
                           for key in ("discounted", "total", "steps")}
        before = policy_digest(expert.expert)
        protocol = dict(command.cfg.experiment,
                        visited_train_shapes=[group["shape_key"] for group in manifest["train"]],
                        scene_assignments_by_rank=[command.env_scene.bincount(
                            minlength=len(command.cfg.scene_motion_files)).cpu().tolist()])
        output.mkdir(parents=True, exist_ok=True)
        config.save_json(output / "config.json")
        (output / "split.json").write_text(json.dumps({**manifest, "data_root": str(root)}, indent=2) + "\n")
        print(f"[sitting depth] {env.num_envs} environments | 200 Hz physics | 50 Hz depth/action "
              f"(simulation time) | CPU dataset {config.dataset_bytes / 1024**3:.2f} GiB", flush=True)
        fields = ("phase", "round", "beta", "dataset_size", "action_mse", "teacher_fraction", "episodes",
                  "first_control_step", "last_control_step", "simulation_seconds")
        with (output / "train_log.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()

            def record(metrics, path):
                if policy_digest(expert.expert) != before:
                    raise RuntimeError("Expert parameters or normalizers changed during distillation")
                writer.writerow(metrics)
                stream.flush()
                print(json.dumps(metrics), flush=True)
                if path is not None:
                    save_student(path, coordinator, checkpoint, protocol, metrics, before)

            collected = collect(env, coordinator, config.bc_rollout_steps, 1.0, episode_returns=episode_returns)
            metrics = coordinator.pretrain_bc()
            record({**metrics, "round": -1, "beta": 1.0, **collected}, output / "model_bc.pt")
            for round_index in range(config.rounds):
                collected = collect(env, coordinator, config.rollout_steps, coordinator.beta(round_index),
                                    episode_returns=episode_returns)
                metrics = coordinator.train_round(round_index, batches=config.updates_per_round)
                save = (round_index + 1) % config.save_every == 0 or round_index + 1 == config.rounds
                record({"phase": "dagger", **metrics, **collected},
                       output / f"model_{round_index + 1}.pt" if save else None)
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Validate teacher, training split and caches without simulation")
    args = parser.parse_args(argv)
    config = SittingDaggerConfig.from_json(args.config)
    checkpoint, manifest, root = validate_inputs(config)
    if args.check:
        print(json.dumps(dict(config=config.to_dict(), split_fingerprint=manifest["fingerprint"],
                              training_scenes=len(scenes(manifest, "train")),
                              teacher_iteration=checkpoint.get("iteration"),
                              dataset_gib=config.dataset_bytes / 1024**3), indent=2))
        return
    train(config, checkpoint, manifest, root)


if __name__ == "__main__":
    main()
