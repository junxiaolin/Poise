#!/usr/bin/env bash
# Two physical GPUs, one Isaac/PhysX scene per GPU, GF20k curriculum.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
exec "${POISE_PYTHON:-python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train.py --distributed --physical-gpus 2 --backend gloo \
  --headless --recipe hexprism40 --num-envs "${POISE_ENVS_PER_GPU:-32000}" \
  --global-frontier-min 20000 --iterations 150000 --seed 42 "$@"
