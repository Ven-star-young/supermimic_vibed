"""Two real Gloo processes exercise learner collectives without launching CUDA."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_process_statistics_gradients_shared_geometry_and_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "-m", "tests.distributed_training_worker", str(tmp_path)]
    result = subprocess.run(command, cwd=root, env=environment, text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-10000:] + result.stderr[-10000:]
    first, second = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert first == second
    assert first["normalizer_count"] == 5
    assert first["rejected_nonfinite"] == 2
    assert "rank 1: ValueError: fixture physics overflow" in first["phase_error"]
    assert first["shared_encoder_wm_only"]
    assert first["checkpoint_restored"]
    assert first["actor_only_keeps_scene_geometry"]
