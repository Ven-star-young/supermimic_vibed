#!/bin/bash
# Same multi-chair learner, using only the final manually selected references.
# Usage: [CPUS=0-23] ./run_train_references.sh [chair_experiment train arguments...]
#        ./run_train_references.sh --check
set -euo pipefail
cd "$(dirname "$0")"

uv run python tools/prepare_reference_training.py \
  --selection data/selections/sitting_back.json --data-root data/motions/sitting

MODE=train
if [ "${1:-}" = "--check" ]; then MODE=check; shift; fi
PREFIX=()
if [ -n "${CPUS:-}" ]; then PREFIX=(taskset -c "$CPUS"); fi
exec "${PREFIX[@]}" uv run python -m supermimic.scripts.chair_experiment "$MODE" \
  --split data/splits/sitting_back_references_seed0.json \
  --run-dir logs/sitting_back_references "$@"
