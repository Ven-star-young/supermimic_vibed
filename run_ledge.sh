#!/usr/bin/env bash
# Edit configs/ledge.json; launch options match run_open_door.sh.
set -euo pipefail
export TRAIN_MODULE=supermimic.scripts.train_ledge
exec "$(dirname -- "$0")/run_open_door.sh" "$@"
