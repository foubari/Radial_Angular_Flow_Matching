#!/usr/bin/env bash
# Additive preparation/sanity wrapper. No active runtime or configuration edits.
set -euo pipefail
: "${SLURM_JOB_ID:?A compute-node allocation is required}"
STUDY_ROOT=/mnt/vast01/users/fouad.oubari/msgm/rafm-additions
STUDY_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
cd "$STUDY_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$STUDY_ROOT/tools/validation_deps:$STUDY_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MIOPEN_USER_DB_PATH="$STUDY_ROOT/outputs_rafm_input_study/recovered/cache/$SLURM_JOB_ID/performance_db"
export MIOPEN_CUSTOM_CACHE_DIR="$STUDY_ROOT/outputs_rafm_input_study/recovered/cache/$SLURM_JOB_ID/kernel_cache"
export TORCHINDUCTOR_CACHE_DIR="$STUDY_ROOT/outputs_rafm_input_study/recovered/cache/$SLURM_JOB_ID/inductor"
export TRITON_CACHE_DIR="$STUDY_ROOT/outputs_rafm_input_study/recovered/cache/$SLURM_JOB_ID/triton"
export MPLCONFIGDIR="$STUDY_ROOT/outputs_rafm_input_study/recovered/cache/$SLURM_JOB_ID/matplotlib"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
exec "$STUDY_PYTHON" "$@"
