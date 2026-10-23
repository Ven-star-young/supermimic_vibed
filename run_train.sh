#!/bin/bash
# Run native mjlab training and save stdout alongside checkpoints.
# Usage: [CPUS=0-23] ./run_train.sh <run_name> [train.py args...]
# Run inside tmux yourself if a detached session is wanted.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-lafan1_main}
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Run name must contain only letters, numbers, underscore, dot, or dash." >&2
  exit 1
fi
PREFIX=()
if [ -n "${CPUS:-}" ]; then PREFIX=(taskset -c "$CPUS"); fi
mkdir -p "logs/$RUN"
"${PREFIX[@]}" uv run python -m supermimic.train --log_dir "logs/$RUN" "$@" \
  2>&1 | tee -a "logs/$RUN/stdout.log"
