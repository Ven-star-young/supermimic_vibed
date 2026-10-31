#!/usr/bin/env bash
# Edit configs/open_door.json; CUDA_VISIBLE_DEVICES selects physical GPUs.
# Example: CUDA_VISIBLE_DEVICES=1,2 ./run_open_door.sh --config configs/open_door.json
set -euo pipefail
cd -- "$(dirname -- "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# A deployed source snapshot may reuse a venv installed in another checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Keep earlier launch commands usable. Only explicitly set variables override
# JSON values; command-line options at the end take precedence over both.
overrides=()
for mapping in \
  SUPERMIMIC_MOTION_DIR:env.commands.motion.motion-dir LOG_ROOT:log-root \
  NUM_ENVS:env.scene.num-envs WM_BATCH:agent.wm-batch PI_BATCH:agent.pi-batch \
  BUFFER_STEPS:agent.buffer-steps WM_ONLY_ITERS:agent.wm-only-iters \
  ITERATIONS:agent.max-iterations SAVE_INTERVAL:agent.save-interval \
  LOG_EVERY:agent.log-every RUN_NAME:agent.run-name; do
  variable=${mapping%%:*}
  if [[ -n "${!variable:-}" ]]; then
    overrides+=("--${mapping#*:}" "${!variable}")
  fi
done

exec "${TRAIN_PYTHON:-.venv/bin/python}" -m supermimic.scripts.train_open_door \
  "${overrides[@]}" "$@"
