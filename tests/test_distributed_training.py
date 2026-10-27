"""Real two-process synchronization and checkpoint/replay ownership."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def test_two_process_training_norms_gradients_checkpoints_and_resume(tmp_path, motion_dir):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    output = tmp_path / "distributed_run"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(root / "tests/distributed_training_worker.py"), str(output), str(motion_dir)]
    result = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout[-12000:] + result.stderr[-12000:]
    first, second = [json.loads((output / f"rank{i}.json").read_text()) for i in range(2)]
    assert first["hashes"] == second["hashes"]
    assert first["iterations"] == second["iterations"] == 3
    assert first["sim_steps"] == second["sim_steps"] == 28
    assert first["normalizer_count"] == second["normalizer_count"] == 28
    with (output / "train_log.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["iteration"]) for r in rows] == [1, 2, 3]  # No duplicate rank writes.
    for iteration in (1, 2, 3):
        assert (output / f"model_{iteration}.pt").is_file()
        assert (output / f"world_models/model_{iteration}.pt").is_file()
        for rank in (0, 1):
            assert (output / f"replays/model_{iteration}.rank{rank:04d}.pt").is_file()
    checkpoint = torch.load(output / "model_3.pt", map_location="cpu", weights_only=False)
    assert checkpoint["distributed"]["world_size"] == 2
    assert checkpoint["distributed"]["batch_sizes"] == "per_rank"
    archives = [torch.load(output / f"replays/model_2.rank{rank:04d}.pt", weights_only=True) for rank in (0, 1)]
    assert not torch.equal(archives[0]["transitions"]["xi"], archives[1]["transitions"]["xi"])
