#!/usr/bin/env bash
# Continue the Trans50 experiment; only the global curriculum gate changes.
set -euo pipefail
cd "$(dirname "$0")/.."
export POISE_ROOT="$PWD"
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1
PYTHON="${POISE_PYTHON:-python}"
RUN_NAME="${RUN_NAME:-HexPrismD40L160_OmniWrist_Tactile_FullPose15_Trans50_4x4090_96k_GF160k_S42_Resume11000}"
CHECKPOINT="${POISE_RESUME_CHECKPOINT:-outputs/training/HexPrismD40L160_OmniWrist_Tactile_FullPose15_Trans50_4x4090_96k_GF20k_S42_Scratch/model_11000.pt}"
EXPERIENCE=()
if [[ -n "${POISE_EXPERIENCE:-}" ]]; then
    EXPERIENCE=(--experience "$POISE_EXPERIENCE")
fi
"$PYTHON" scripts/check_hexprism_d40_l160.py
# model_11000 continues at 11001; 138999 further updates retain the original
# total budget of 150000 updates (last update index 149999).
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train.py --distributed --physical-gpus 4 --backend gloo --headless \
    --recipe hexprism_d40_l160_omniwrist_tactile_fullpose15_trans50_v1 \
    --num-envs 24000 --iterations 138999 --seed 42 \
    --global-frontier-min 160000 --resume-global-frontier-from 20000 \
    --resume "$CHECKPOINT" --output "outputs/training/$RUN_NAME" "${EXPERIENCE[@]}"
