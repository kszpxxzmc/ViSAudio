#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
PYTHON="$(command -v -- "$PYTHON")"
export PATH="$(dirname -- "$PYTHON"):$PATH"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Use the active environment; the Python loader also discovers ../perception_models.
if [[ -n "${PE_ROOT:-}" ]]; then
    export PE_ROOT
fi

exec "$PYTHON" -B -u "$SCRIPT_DIR/demo_batch.py" --gpus "${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}" "$@"
