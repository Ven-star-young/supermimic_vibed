"""Archive/replay retries must preserve durable results without using real SSH or GPUs."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import monitor_open_door as monitor


@pytest.fixture
def archive(tmp_path, monkeypatch):
    mount = tmp_path / "Lexar"
    mount.mkdir()
    cfg = {
        "host": "test-host", "mount": str(mount),
        "output_root": str(mount / "archive"), "remote_repo": "/remote/repo",
        "remote_run": "/remote/logs/run", "remote_motion_dir": "/remote/motion",
        "device": "cpu",
    }
    monkeypatch.setattr(Path, "is_mount", lambda self: self == mount)
    monkeypatch.setattr(monitor, "inspect_training", lambda cfg: {
        "state": "running", "latest": {"iteration": "200"},
        "models": [{"name": "model_200.pt", "iteration": 200, "size": 5}],
    })
    downloads = []

    def download(cfg, source, destination, **kwargs):
        downloads.append(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"model" if source.endswith(".pt") else b"metadata")
        return "digest"

    def reference(cfg, directory):
        folder = directory / "reference"
        folder.mkdir(exist_ok=True)
        (folder / "clip.npz").touch()
        (folder / "clip.xml").touch()

    monkeypatch.setattr(monitor, "download", download)
    monkeypatch.setattr(monitor, "snapshot_reference", reference)
    return SimpleNamespace(cfg=cfg, directory=mount / "archive/run", downloads=downloads,
                           download=download)


def test_missing_mount_fails_before_creating_output_or_contacting_server(archive, monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda self: False)
    monkeypatch.setattr(monitor, "inspect_training", lambda cfg: pytest.fail("contacted server"))
    with pytest.raises(RuntimeError, match="not mounted"):
        monitor.check(archive.cfg)
    assert not Path(archive.cfg["output_root"]).exists()
    assert not archive.downloads


def test_output_symlink_outside_mount_is_rejected(archive, tmp_path):
    elsewhere = tmp_path / "system-disk"
    elsewhere.mkdir()
    Path(archive.cfg["output_root"]).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="Lexar mount"):
        monitor.check(archive.cfg)
    assert not list(elsewhere.iterdir())


@pytest.mark.parametrize("failure", ["download", "render"])
def test_failed_model_is_retried_and_successful_model_is_idempotent(archive, monkeypatch, failure):
    attempts = {"download": 0, "render": 0, "evaluate": 0}

    def download(cfg, source, destination, **kwargs):
        if source.endswith(".pt"):
            attempts["download"] += 1
            if failure == "download" and attempts["download"] == 1:
                raise subprocess.CalledProcessError(1, ["scp"])
        return archive.download(cfg, source, destination, **kwargs)

    def execute(command, **kwargs):
        if "supermimic.scripts.evaluate" in command:
            attempts["evaluate"] += 1
            checkpoint = Path(command[command.index("--checkpoint-file") + 1])
            Path(command[command.index("--record") + 1]).write_bytes(b"trajectory")
            checkpoint.with_name(checkpoint.stem + "_eval.json").write_text(json.dumps({
                "iteration": 200, "norm_return": .5, "door_tracking": {
                    "reference_completion_rate": .8, "angle_rmse_rad": .1,
                    "angular_velocity_rmse_radps": .2,
                },
            }))
        elif "tools.render" in command:
            attempts["render"] += 1
            if failure == "render" and attempts["render"] == 1:
                raise subprocess.CalledProcessError(1, command)
            Path(command[4]).write_bytes(b"video")
        else:
            assert command[0] == "ffprobe"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(monitor, "download", download)
    monkeypatch.setattr(monitor, "run", execute)
    marker = archive.directory / "evaluations/model_200/complete.json"
    receipt = archive.directory / "checkpoints/model_200.download.json"

    first = monitor.check(archive.cfg)
    assert len(first["errors"]) == 1
    assert first["new_models"] == []
    assert not marker.exists()
    assert receipt.exists() == (failure == "render")

    second = monitor.check(archive.cfg)
    assert second["errors"] == []
    assert len(second["new_models"]) == 1
    report = json.loads(marker.read_text())
    assert all(Path(video).stat().st_size > 0 for video in report["videos"])
    assert attempts["download"] == (2 if failure == "download" else 1)
    previous = attempts.copy()

    third = monitor.check(archive.cfg)
    assert third["new_models"] == []
    assert third["errors"] == []
    assert attempts == previous
    summary = (archive.directory / "evaluation_summary.csv").read_text().splitlines()
    assert len(summary) == 2
    assert summary[1].startswith("200,0.5,0.8,0.1,0.2,")


@pytest.mark.parametrize("failure", ["size", "checksum", "transport"])
def test_download_failure_preserves_existing_file_and_never_promotes_partial(tmp_path, monkeypatch, failure):
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"previous")

    def execute(command, **kwargs):
        Path(command[-1]).write_bytes(b"fresh")
        if failure == "transport":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(monitor, "run", execute)
    monkeypatch.setattr(monitor, "remote", lambda *args: "wrong-checksum")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        monitor.download({"host": "test-host"}, "/remote/model.pt", destination,
                         verify=True, expected_size=99 if failure == "size" else 5)
    assert destination.read_bytes() == b"previous"
    assert destination.with_suffix(".pt.partial").read_bytes() == b"fresh"
