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
    echo "Options: --start first | --seed 42 | --training-conditions | --object-motion-file /path/to/chair_motion.npz"
    echo "Full help: $0 --help"
    exit 0
fi
if [[ "$1" != -* ]]; then
    set -- --checkpoint-file "$@"
fi
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$project_dir/.venv/bin/python" -m supermimic.scripts.play_policy "$@"
