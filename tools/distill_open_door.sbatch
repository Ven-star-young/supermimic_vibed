#!/usr/bin/env bash
# Submit: mkdir -p logs/slurm && sbatch tools/distill_open_door.sbatch path/to/config.json
#SBATCH --job-name=door-dagger
#SBATCH --account=interns
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/wangfeng/supermimic_vibed
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

source_config="${1:-configs/open_door_dagger_tactile_counts_only.json}"
run_config="logs/slurm/distill-${SLURM_JOB_ID}.json"
.venv/bin/python - "$source_config" "$run_config" <<'PY'
import json
import os
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
config.update(
    output_dir=str(Path(config["output_dir"]) / f"slurm-{os.environ['SLURM_JOB_ID']}"),
    device="cuda:0",
)
for field, variable in (("expert_checkpoint", "SUPERMIMIC_EXPERT_CHECKPOINT"),
                        ("motion_dir", "SUPERMIMIC_MOTION_DIR")):
    if os.environ.get(variable):
        config[field] = os.environ[variable]
for source in (Path(config["expert_checkpoint"]), Path(config["motion_dir"]),
               Path(config["motion_dir"]).with_suffix(".xml")):
    if not source.is_file():
        raise FileNotFoundError(source)
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + "\n")
print(f"Configuration: {sys.argv[2]}\nOutput: {config['output_dir']}", flush=True)
PY

# Slurm owns CUDA_VISIBLE_DEVICES; this entrypoint runs one process on its GPU.
.venv/bin/python -m supermimic.scripts.distill_open_door --config "$run_config" --check
srun --ntasks=1 --cpu-bind=none .venv/bin/python -u -m supermimic.scripts.distill_open_door --config "$run_config"
