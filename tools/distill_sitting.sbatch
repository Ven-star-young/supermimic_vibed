#!/usr/bin/env bash
# Submit from the CUHK repository root: mkdir -p logs/slurm && sbatch tools/distill_sitting.sbatch
#SBATCH --job-name=sitting-depth-distill
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

source_config="${1:-configs/sitting_dagger_depth.json}"
run_config="logs/slurm/sitting-depth-${SLURM_JOB_ID}.json"
.venv/bin/python - "$source_config" "$run_config" <<'PY'
import json
import os
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
config.update(
    expert_checkpoint="logs/sitting_task_2gpu/model_10000.pt",
    output_dir=str(Path(config["output_dir"]) / f"slurm-{os.environ['SLURM_JOB_ID']}"),
    device="cuda:0",
)
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + "\n")
print(f"Configuration: {sys.argv[2]}\nOutput: {config['output_dir']}", flush=True)
PY

# Slurm selects the physical GPU; cuda:0 is the one visible to this process.
.venv/bin/python -m supermimic.scripts.distill_sitting --config "$run_config" --check
srun --ntasks=1 .venv/bin/python -u -m supermimic.scripts.distill_sitting --config "$run_config"
