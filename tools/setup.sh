#!/bin/bash
# Install the native mjlab runtime into the repository's uv environment.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
uv sync
