#!/usr/bin/env bash
# Use this project's code and environment, preserving caller-relative paths.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    echo "Python environment missing. Run: cd '$project_dir' && uv sync --extra motion" >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 /path/to/model.pt [options]"
    echo "Options: --init clean|perturbed | --random-start | --partition test | --reference /path/to/motion.npz"
    echo "Evaluate: --evaluate --partition all --seeds 0 1 2 3 4 --output results/evaluation.json"
    echo "Browser: --browse --partition all --port 8088"
    echo "Requires a checkpoint from the task-specific sitting branch."
    echo "Full help: $0 --help"
    exit 0
fi
if [[ "$1" != -* ]]; then
    set -- --checkpoint-file "$@"
fi
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$project_dir/.venv/bin/python" -m supermimic.scripts.play_policy "$@"
