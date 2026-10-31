"""User-edited JSON must reach the native launcher without stale scene settings."""

import json

import pytest


def _config_file(tmp_path, payload):
    path = tmp_path / "training.json"
    path.write_text(json.dumps(payload))
    return path


def test_json_collections_and_optional_values_reach_native_config(tmp_path):
    from supermimic.scripts.train_open_door import build_config

    path = _config_file(tmp_path, {
        "task": {
            "motion_dir": "data/a directory with spaces",
            "motion_files": [],
            "reset_noise": False,
            "episode_length_s": 12.0,
            "door_randomization_cfg": {"enabled": False, "mass": [6.0, 12.0]},
            "door_reward_cfg": {"penetration_limit": None},
            "physical_cfg": {"com_ranges": [[-.01, .01], [-.02, .02], [-.03, .03]]},
            "reset_cfg": {"position": [.01, .02, .03]},
        },
        "training": {"num_envs": 2048, "iterations": 3000, "gpu_ids": [1, 2],
                     "wm_batch": 256, "log_root": "logs/a run"},
    })
    cfg = build_config(path)
    command = cfg.env.commands["motion"]
    assert command.motion_dir == "data/a directory with spaces"
    assert command.motion_files == ()
    assert command.reset_noise is False
    assert cfg.env.episode_length_s == 12.0
    assert command.door_randomization_cfg.enabled is False
    assert command.door_randomization_cfg.mass == (6.0, 12.0)
    assert command.door_reward_cfg.penetration_limit is None
    assert command.physical_cfg.com_ranges == ((-.01, .01), (-.02, .02), (-.03, .03))
    assert command.reset_cfg.position == (.01, .02, .03)
    assert cfg.env.scene.num_envs == 2048
    assert cfg.agent.max_iterations == 3000
    assert cfg.agent.wm_batch == 256
    assert cfg.gpu_ids == [1, 2]
    assert cfg.log_root == "logs/a run"


def test_cli_overrides_rebind_door_factory_to_final_command(tmp_path):
    from supermimic.scripts.train_open_door import build_config

    path = _config_file(tmp_path, {
        "task": {"door_mass": 8.0, "motion_files": ["one.npz", "two file.npz"]},
        "training": {"num_envs": 1024, "wm_batch": 512},
    })
    cfg = build_config(path, ["--env.scene.num-envs", "3072", "--agent.wm-batch", "768",
                              "--env.commands.motion.door-mass", "11.0"])
    command = cfg.env.commands["motion"]
    assert cfg.env.scene.num_envs == 3072
    assert cfg.agent.wm_batch == 768
    assert command.motion_files == ("one.npz", "two file.npz")
    assert command.door_mass == 11.0
    assert cfg.env.scene.entities["door"].spec_fn.args[0] is command


def test_shipped_config_and_effective_summary_can_be_reloaded(tmp_path):
    from supermimic.scripts.train_open_door import DEFAULT_CONFIG, build_config, config_summary

    cfg = build_config(DEFAULT_CONFIG)
    summary = config_summary(cfg)
    restored = build_config(_config_file(tmp_path, summary))
    assert config_summary(restored) == summary
    assert restored.env.scene.entities["door"].spec_fn.args[0] is restored.env.commands["motion"]


def test_training_discount_controls_reference_curriculum_scale(tmp_path):
    from supermimic.scripts.train_open_door import build_config

    path = _config_file(tmp_path, {"task": {"gamma": .95}, "training": {"gamma": .98}})
    cfg = build_config(path)
    assert cfg.agent.gamma == cfg.env.commands["motion"].gamma == .98
    cfg = build_config(path, ["--agent.gamma", ".97"])
    assert cfg.agent.gamma == cfg.env.commands["motion"].gamma == .97


@pytest.mark.parametrize("payload", [
    {"task": {"door_mass": -1.0}},
    {"task": {"door_randomization_cfg": {"mass": [15.0, 5.0]}}},
    {"task": {"door_randomization_cfg": {"mass": [8.0]}}},
    {"task": {"door_mas": 8.0}},
    {"training": {"wm_bach": 128}},
    {"unknown_section": {}},
    {"task": []},
])
def test_bad_config_is_rejected_before_launch(tmp_path, payload):
    from supermimic.scripts.train_open_door import build_config

    with pytest.raises((ValueError, SystemExit)):
        build_config(_config_file(tmp_path, payload))


def test_check_prints_effective_settings_without_launch(tmp_path, monkeypatch, capsys):
    from supermimic.scripts import train_open_door

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("--check started training")

    monkeypatch.setattr(train_open_door, "launch_training", forbidden_launch)
    path = _config_file(tmp_path, {
        "task": {"door_mass": 9.0},
        "training": {"num_envs": 1024, "iterations": 1000},
    })
    train_open_door.main(["--config", str(path), "--check", "--env.scene.num-envs", "4096"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["task"]["door_mass"] == 9.0
    assert summary["training"]["num_envs"] == 4096
    assert summary["training"]["iterations"] == 1000
