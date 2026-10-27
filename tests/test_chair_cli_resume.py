"""Rejected experiment resumes must preserve the existing run's provenance."""

import json
import sys
from pathlib import Path

import pytest

from supermimic.scripts import chair_experiment


def test_rejected_resume_preserves_existing_experiment_metadata(tmp_path, monkeypatch):
    run_dir = tmp_path / "existing_rotate_run"
    run_dir.mkdir()
    original = {
        "split.json": b'{"fingerprint": "original-split", "data_root": "../motions"}\n',
        "experiment.json": b'{"scene_mode": "rotate", "collision_profile": "default", "num_envs": 128}\n',
    }
    for name, contents in original.items():
        (run_dir / name).write_bytes(contents)
    checkpoint = run_dir / "model_2000.pt"
    checkpoint.write_bytes(b"checkpoint loading is stubbed")
    data_root = tmp_path / "motions"
    manifest = {"fingerprint": "different-split", "train": [], "test": []}
    split = tmp_path / "new_split.json"
    split.write_text(json.dumps(manifest))
    monkeypatch.setattr(chair_experiment, "load_split", lambda *_: (manifest, data_root))
    monkeypatch.setattr(sys, "argv", [
        "chair_experiment", "train", "--split", str(split),
        "--scene-mode", "mixed", "--collision-profile", "compact",
        "--run-dir", str(run_dir), "--resume", str(checkpoint), "--iterations", "1",
    ])
    events = []

    class RejectingRunner:
        def __init__(self, *args, **kwargs):
            assert kwargs["log_dir"] == run_dir
            assert kwargs["collision_profile"] == "compact"
            events.append("initialized")

        def load(self, path):
            assert Path(path) == checkpoint
            events.append("load")
            raise ValueError("Checkpoint task/schema mismatch")

        def learn(self, _):
            pytest.fail("An incompatible checkpoint must not start training")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(chair_experiment, "MixedChairExperimentRunner", RejectingRunner)
    distributed = chair_experiment.distributed
    monkeypatch.setattr(distributed, "world_size", lambda: 1)
    monkeypatch.setattr(distributed, "initialize", lambda _: "cpu")
    monkeypatch.setattr(distributed, "active", lambda: False)
    monkeypatch.setattr(distributed, "primary", lambda: True)
    monkeypatch.setattr(distributed, "shutdown", lambda: events.append("shutdown"))

    with pytest.raises(ValueError, match="task/schema mismatch"):
        chair_experiment.main()

    assert events == ["initialized", "load", "closed", "shutdown"]
    for name, contents in original.items():
        assert (run_dir / name).read_bytes() == contents
    assert checkpoint.read_bytes() == b"checkpoint loading is stubbed"
