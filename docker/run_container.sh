#!/bin/bash
# Start the long-lived 'supermimic' container with one GPU and the repository mounted.
# Usage: docker/run_container.sh [GPU_INDEX] [MEMORY_LIMIT]
set -euo pipefail
SUPERMIMIC_GPU=${1:-0}
SUPERMIMIC_MEMORY=${2:-128g}
SUPERMIMIC_REPO="$(cd "$(dirname "$0")/.." && pwd)"
docker run -d --name supermimic --gpus "\"device=${SUPERMIMIC_GPU}\"" --memory="$SUPERMIMIC_MEMORY" --shm-size=16g \
  -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e MPLCONFIGDIR=/tmp/mpl \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
  -v "$SUPERMIMIC_REPO":/workspace/supermimic -w /workspace/supermimic supermimic:mjlab sleep infinity
