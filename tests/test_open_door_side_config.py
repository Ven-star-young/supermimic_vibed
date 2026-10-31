"""Side-door launches keep separate task semantics through the shared JSON parser."""

import json

import pytest


def test_side_config_resolves_to_its_own_task_and_roundtrips(tmp_path):
    from supermimic.scripts.train_open_door import build_config, config_summary
    from supermimic.scripts.train_open_door_side import DEFAULT_CONFIG, TASK_ID

    cfg = build_config(DEFAULT_CONFIG, task_id=TASK_ID)
    command = cfg.env.commands["motion"]
    assert cfg.agent.experiment_name == "supermimic_open_door_side"
    assert cfg.agent.run_name == "right_side_tracking_dr_rsi"
    assert cfg.env.episode_length_s == 15.0
    assert command.motion_dir.endswith("video6_g1_door_video45_right_side_asset.npz")
    assert "back_weight" not in vars(command.door_reward_cfg)
    assert "contact_phase_angle" not in vars(command.door_reward_cfg)
    assert command.door_reward_cfg.contact_weight == .25
    assert command.door_reward_cfg.reference_contact_std == .05
    assert command.door_reward_cfg.contact_penetration_tolerance == .003
    assert cfg.env.scene.entities["door"].spec_fn.args[0] is command
    summary = config_summary(cfg)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(summary))
    restored = build_config(path, task_id=TASK_ID)
    assert config_summary(restored) == summary


def test_side_config_rejects_back_reward_options(tmp_path):
    from supermimic.scripts.train_open_door import build_config
    from supermimic.scripts.train_open_door_side import TASK_ID

    path = tmp_path / "back_reward.json"
    path.write_text(json.dumps({"task": {"door_reward_cfg": {"back_weight": .25}}}))
    with pytest.raises(ValueError, match="back_weight"):
        build_config(path, task_id=TASK_ID)


def test_side_entrypoint_checks_its_default_config_without_launch(monkeypatch, capsys):
    from supermimic.scripts import train_open_door, train_open_door_side

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("--check started training")

    monkeypatch.setattr(train_open_door, "launch_training", forbidden_launch)
    train_open_door_side.main(["--check", "--env.scene.num-envs", "1536"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["training"]["experiment_name"] == "supermimic_open_door_side"
    assert summary["training"]["num_envs"] == 1536
    assert "back_weight" not in summary["task"]["door_reward_cfg"]


def test_side_entrypoint_launches_registered_side_task(monkeypatch):
    from supermimic.scripts import train_open_door, train_open_door_side

    launches = []
    monkeypatch.setattr(train_open_door, "launch_training", lambda *args: launches.append(args))
    train_open_door_side.main(["--env.scene.num-envs", "32", "--agent.gamma", ".97"])
    [(task_id, cfg)] = launches
    assert task_id == train_open_door_side.TASK_ID
    assert cfg.env.scene.num_envs == 32
    assert cfg.agent.gamma == cfg.env.commands["motion"].gamma == .97
    assert cfg.agent.experiment_name == "supermimic_open_door_side"
