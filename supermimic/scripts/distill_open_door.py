"""Distill an OpenDoor expert into a 10-frame proprioception/tactile student."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from supermimic.distillation.open_door import (
    DaggerCoordinator, FrozenOpenDoorExpert, OpenDoorDaggerConfig, OpenDoorStudentPolicy,
)
from supermimic.distillation.open_door.models import TactileHistory
from supermimic.scripts.evaluate import _restore_open_door_config
from supermimic.scripts.play_policy import policy_digest
from supermimic.tasks.open_door.models import Policy


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/open_door_dagger_observable.json"
TASKS = {
    "g1_open_door": "Mjlab-SuperMimic-OpenDoor-Unitree-G1",
    "g1_open_door_side": "Mjlab-SuperMimic-OpenDoorSide-Unitree-G1",
}


def make_env_cfg(config, checkpoint):
    task_name = (checkpoint.get("task") or {}).get("name")
    if task_name not in TASKS:
        raise ValueError("Distillation requires an OpenDoor or OpenDoorSide expert checkpoint")
    policy_config = checkpoint.get("policy_config", {})
    if (policy_config.get("history_length"), policy_config.get("goal_frames")) != (10, 10):
        raise ValueError("The expert must use 10 history frames and 10 reference frames")
    expected_observation = checkpoint["task"]["dimensions"]["observation"]
    if config.expert_observation_dim != expected_observation:
        raise ValueError(f"Set expert_observation_dim={expected_observation} for this teacher checkpoint")
    cfg = load_env_cfg(TASKS[task_name], play=False)
    _restore_open_door_config(cfg, checkpoint)
    cfg.scene.num_envs, cfg.auto_reset, cfg.seed = config.num_envs, False, config.seed
    command = cfg.commands["motion"]
    command.motion_dir = config.motion_dir
    command.motion_files = ()
    path = Path(config.motion_dir)
    command.door_xml = str(path.with_suffix(".xml")) if path.is_file() else None
    command.history_length, command.goal_frames = 10, 10
    for sensor in cfg.scene.sensors:
        if sensor.name == "tactile":
            sensor.layout = config.tactile_layout
            sensor.contact_distance_m = config.tactile_contact_distance_m
            sensor.counts_full_scale_depth_m = config.tactile_counts_full_scale_depth_m
            sensor.counts_exponent = config.tactile_counts_exponent
    return cfg


@torch.no_grad()
def collect(env, coordinator, tactile_history, steps, beta):
    """Sample touch and teacher labels at the same pre-action control state."""
    teacher_count = contact_count = valid_count = episodes = 0
    sensor = env.scene["tactile"]
    command = env.command_manager.get_term("motion")
    for _ in range(steps):
        observations = env.observation_manager.compute()
        # State/command resets may refresh FK after the normal sensor update.
        sensor.update(0.)
        tactile = sensor.data
        counts, valid = tactile_history.update(tactile.channel_counts, tactile.channel_valid,
                                               env.common_step_counter)
        step = coordinator.label_and_select(observations["policy_history"], observations["goal"], beta,
                                            tactile_history=counts, tactile_valid=valid)
        if not torch.isfinite(step.executed_action).all() or not torch.isfinite(step.expert_action).all():
            raise FloatingPointError("Nonfinite expert/student action during distillation")
        _, reward, terminated, truncated, _ = env.step(step.executed_action)
        if bool(env.sim.data.overflow.any()):
            raise RuntimeError("Physics buffer overflow during distillation")
        if not torch.isfinite(command.dynamics.pack(command.state())).all() or not torch.isfinite(reward).all():
            raise FloatingPointError("Nonfinite physics during distillation")
        done = terminated | truncated | (command.time_steps >= command.motion.clip_end[command.motion.clip_id[command.time_steps]])
        coordinator.record(step, done)
        teacher_count += int(step.expert_mask.sum())
        contact_count += int(((step.tactile_history[:, -1] > 0) & step.tactile_valid[:, -1]).sum())
        valid_count += int(step.tactile_valid[:, -1].sum())
        ids = done.nonzero().flatten()
        episodes += len(ids)
        if len(ids):
            env.reset(env_ids=ids)
            tactile_history.reset(ids)
    return {"teacher_fraction": teacher_count / (steps * env.num_envs),
            "tactile_nonzero_fraction": contact_count / max(valid_count, 1), "episodes": episodes}


def save_student(path, coordinator, sensor, checkpoint, metrics):
    """Save the deployable student, input schema and tactile provenance."""
    config = coordinator.config
    torch.save({
        "kind": "open_door_tactile_student" if config.use_tactile else "open_door_proprio_student",
        "schema_version": 1,
        "student": coordinator.student.state_dict(),
        "policy_config": {"history_length": 10, "goal_frames": 10, "proprio_dim": 93,
                          "tactile_channels": 112 if config.use_tactile else 0,
                          "tactile_latent_dim": 64 if config.use_tactile else 0, "action_dim": 29,
                          "history_order": "oldest_to_newest",
                          "tactile_units": "uint8_counts" if config.use_tactile else None},
        "config": config.to_dict(), "expert_checkpoint": str(Path(config.expert_checkpoint).resolve()),
        "expert_task": checkpoint["task"], "tactile_metadata": sensor.metadata,
        "goal_encoder": {
            "initialized_from_teacher": bool(coordinator.student.goal_initialized_from_teacher),
            "encoder_parameters_trainable_after_initialization": True,
            "normalizer_statistics": "teacher_then_bc_initialization",
        },
        "metrics": metrics,
    }, path)


def train(config):
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    torch.manual_seed(config.seed)
    checkpoint = torch.load(config.expert_checkpoint, map_location="cpu", weights_only=False)
    env = ManagerBasedRlEnv(make_env_cfg(config, checkpoint), device=config.device)
    try:
        env.command_manager.get_term("motion").task.validate_checkpoint(checkpoint)
        expert = FrozenOpenDoorExpert(Policy(use_tactile="tactile" in checkpoint["task"]).to(config.device))
        expert.load_checkpoint(checkpoint)
        student = OpenDoorStudentPolicy(use_tactile=config.use_tactile).to(config.device)
        if config.initialize_goal_from_teacher:
            student.initialize_goal_from_expert(expert.expert)
        coordinator = DaggerCoordinator(expert, student, config=config, device=config.device)
        history = TactileHistory(config.num_envs, device=config.device)
        env.reset()
        before = policy_digest(expert.expert)
        output.mkdir(parents=True, exist_ok=True)
        config.save_json(output / "config.json")
        fields = ("phase", "round", "beta", "dataset_size", "action_mse",
                  "teacher_fraction", "tactile_nonzero_fraction", "episodes")
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
                    save_student(path, coordinator, env.scene["tactile"], checkpoint, metrics)

            collected = collect(env, coordinator, history, config.bc_rollout_steps, 1.0)
            metrics = coordinator.pretrain_bc()
            record({**metrics, "round": -1, "beta": 1.0, **collected}, output / "model_bc.pt")
            for round_index in range(config.rounds):
                collected = collect(env, coordinator, history, config.rollout_steps, coordinator.beta(round_index))
                metrics = coordinator.train_round(round_index, batches=config.updates_per_round)
                save = (round_index + 1) % config.save_every == 0 or round_index + 1 == config.rounds
                record({"phase": "dagger", **metrics, **collected},
                       output / f"model_{round_index + 1}.pt" if save else None)
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Validate and print configuration without simulation")
    args = parser.parse_args(argv)
    config = OpenDoorDaggerConfig.from_json(args.config)
    if args.check:
        print(json.dumps(config.to_dict(), indent=2))
        return
    train(config)


if __name__ == "__main__":
    main()
