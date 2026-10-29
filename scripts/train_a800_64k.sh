#!/usr/bin/env bash
# One physical A800; two PhysX scenes avoid the per-scene material limit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
exec "${POISE_PYTHON:-python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train.py --distributed --physical-gpus 1 --backend gloo \
  --headless --recipe hexprism40 --num-envs 32000 --iterations 150000 \
  --seed 42 "$@"
