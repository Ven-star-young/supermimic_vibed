#!/usr/bin/env bash
# Edit configs/yogaball.json; launch options match run_open_door.sh.
set -euo pipefail
export TRAIN_MODULE=supermimic.scripts.train_yogaball
exec "$(dirname -- "$0")/run_open_door.sh" "$@"
