#!/usr/bin/env bash
# One native mjlab launcher; CUDA_VISIBLE_DEVICES selects physical GPUs.
# Example: CUDA_VISIBLE_DEVICES=1,2 ./run_open_door.sh
set -euo pipefail
cd -- "$(dirname -- "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# A deployed source snapshot may reuse a venv installed in another checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

exec "${TRAIN_PYTHON:-.venv/bin/python}" -m mjlab.scripts.train \
  Mjlab-SuperMimic-OpenDoor-Unitree-G1 \
  --gpu-ids all \
  --log-root "${LOG_ROOT:-logs/rsl_rl}" \
  --env.scene.num-envs "${NUM_ENVS:-1024}" \
  --env.commands.motion.sampling-mode adaptive \
  --env.commands.motion.uniform-weight 0.5 \
  --env.commands.motion.current-weight 0.4 \
  --env.commands.motion.history-weight 0.1 \
  --agent.wm-batch "${WM_BATCH:-512}" \
  --agent.pi-batch "${PI_BATCH:-512}" \
  --agent.buffer-steps "${BUFFER_STEPS:-256}" \
  --agent.wm-only-iters "${WM_ONLY_ITERS:-500}" \
  --agent.max-iterations "${ITERATIONS:-10000}" \
  --agent.save-interval "${SAVE_INTERVAL:-200}" \
  --agent.log-every "${LOG_EVERY:-10}" \
  --agent.run-name "${RUN_NAME:-back_tracking_rsi}" \
  "$@"
