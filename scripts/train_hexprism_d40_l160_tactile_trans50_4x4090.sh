#!/usr/bin/env bash
# D40 x L160 original mesh: OmniWrist, tactile, FullPose 15 degrees / 15 mm; translation curriculum 10-50 mm.
set -euo pipefail
cd "$(dirname "$0")/.."
export POISE_ROOT="$PWD"
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
PYTHON="${POISE_PYTHON:-python}"
RUN_NAME="${RUN_NAME:-HexPrismD40L160_OmniWrist_Tactile_FullPose15_Trans50_4x4090_96k_GF20k_S42_Scratch}"
EXPERIENCE=()
if [[ -n "${POISE_EXPERIENCE:-}" ]]; then
    EXPERIENCE=(--experience "$POISE_EXPERIENCE")
fi
"$PYTHON" scripts/check_hexprism_d40_l160.py
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train.py --distributed --physical-gpus 4 --backend gloo --headless \
    --recipe hexprism_d40_l160_omniwrist_tactile_fullpose15_trans50_v1 \
    --num-envs "${POISE_ENVS_PER_RANK:-24000}" --iterations "${POISE_ITERATIONS:-150000}" \
    --global-frontier-min 20000 --seed "${POISE_BASE_SEED:-42}" \
    --output "outputs/training/$RUN_NAME" "${EXPERIENCE[@]}"
