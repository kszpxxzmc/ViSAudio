#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone \
    --nproc_per_node="${GPUS:-1}" training/extract_video_training_latents.py "$@"
