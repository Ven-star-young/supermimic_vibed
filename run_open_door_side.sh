#!/usr/bin/env bash
# Edit configs/open_door_right_side.json; launch options match run_open_door.sh.
set -euo pipefail
export TRAIN_MODULE=supermimic.scripts.train_open_door_side
exec "$(dirname -- "$0")/run_open_door.sh" "$@"
