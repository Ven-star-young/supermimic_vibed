#!/bin/bash
# Convenience alias for mjlab's stock train executable.
# Usage: [CPUS=0-23] ./run_train.sh <experiment_name> [mjlab train args...]
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-lafan1_mjlab}
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Experiment name must contain only letters, numbers, underscore, dot, or dash." >&2
  exit 1
fi
PREFIX=()
if [ -n "${CPUS:-}" ]; then PREFIX=(taskset -c "$CPUS"); fi
exec "${PREFIX[@]}" uv run train Mjlab-SuperMimic-Tracking-Unitree-G1 \
  --log-root logs --agent.experiment-name "$RUN" "$@"
