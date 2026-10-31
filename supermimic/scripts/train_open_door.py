"""Load an editable JSON configuration, then use mjlab's native trainer."""

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import tyro
from mjlab import TYRO_FLAGS
from mjlab.scripts.train import TrainConfig, launch_training

import supermimic.tasks  # noqa: F401 -- registers the task, without creating a simulator


TASK_ID = "Mjlab-SuperMimic-OpenDoor-Unitree-G1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/open_door.json"


def _literal(value, default):
    # mjlab's tyro parser expects Python collection literals in a single token.
    # JSON arrays become tuples for tuple fields, including nested COM ranges.
    if isinstance(default, (tuple, list)) and isinstance(value, list):
        items = [_literal(item, default[min(i, len(default) - 1)] if default else None)
                 for i, item in enumerate(value)]
        return tuple(items) if isinstance(default, tuple) else items
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Configuration numbers must be finite")
    return value


def _arguments(values, defaults, prefix):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix} must be a JSON object")
    known = vars(defaults) if is_dataclass(defaults) else defaults
    args = []
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown configuration field: {prefix}.{key}")
        current, flag = known[key], f"{prefix}.{key.replace('_', '-')}"
        if is_dataclass(current) or isinstance(current, dict):
            args.extend(_arguments(value, current, flag))
        else:
            value = _literal(value, current)
            args.extend((f"--{flag}", value if isinstance(value, str) else repr(value)))
    return args


def build_config(path, overrides=()):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict) or set(values) - {"task", "training"}:
        raise ValueError("Configuration must contain only task and training objects")
    if not all(isinstance(values.get(key, {}), dict) for key in ("task", "training")):
        raise ValueError("task and training must be JSON objects")
    defaults = TrainConfig.from_task(TASK_ID)
    task, training = dict(values.get("task", {})), dict(values.get("training", {}))
    args = []
    if "episode_length_s" in task:
        args += _arguments({"episode_length_s": task.pop("episode_length_s")}, defaults.env, "env")
    args += _arguments(task, defaults.env.commands["motion"], "env.commands.motion")
    for key in ("gpu_ids", "log_root"):
        if key in training:
            # Top-level native flags do not have an env/agent prefix.
            value = _literal(training.pop(key), getattr(defaults, key))
            args.extend((f"--{key.replace('_', '-')}", value if isinstance(value, str) else repr(value)))
    if "num_envs" in training:
        args += _arguments({"num_envs": training.pop("num_envs")}, defaults.env.scene, "env.scene")
    if "iterations" in training:
        if "max_iterations" in training:
            raise ValueError("Use training.iterations, not both iterations and max_iterations")
        training["max_iterations"] = training.pop("iterations")
    args += _arguments(training, defaults.agent, "agent")
    cfg = tyro.cli(TrainConfig, default=defaults, args=[*args, *overrides], config=TYRO_FLAGS)
    if cfg.env.scene.num_envs < 1 or cfg.agent.max_iterations < 1:
        raise ValueError("num_envs and iterations must be positive")
    # The native collector already reads the runner discount for returns; keep
    # the command's initial curriculum scale consistent with it too.
    cfg.env.commands["motion"].gamma = cfg.agent.gamma
    return cfg


def config_summary(cfg):
    training = asdict(cfg.agent)
    # Inherited PPO rollout length is unused and suppressed by the native CLI.
    training.pop("num_steps_per_env", None)
    training["iterations"] = training.pop("max_iterations")
    training.update(num_envs=cfg.env.scene.num_envs, gpu_ids=cfg.gpu_ids, log_root=cfg.log_root)
    return {"task": {"episode_length_s": cfg.env.episode_length_s,
                     **asdict(cfg.env.commands["motion"])}, "training": training}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Additional --env.*, --agent.*, --gpu-ids and --log-root arguments override the JSON."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Print the effective configuration without simulation")
    args, overrides = parser.parse_known_args(argv)
    try:
        cfg = build_config(args.config, overrides)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.check:
        print(json.dumps(config_summary(cfg), indent=2, ensure_ascii=False))
        return
    print(f"[OpenDoor] config={args.config.resolve()} | {cfg.env.scene.num_envs} envs/GPU | "
          f"WM batch={cfg.agent.wm_batch}, Actor batch={cfg.agent.pi_batch} per GPU", flush=True)
    launch_training(TASK_ID, cfg)


if __name__ == "__main__":
    main()
