#!/usr/bin/env bash
#SBATCH --partition=hermes-2
#SBATCH --nodelist=auh7-3b-gpu-008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=tflow-entrypoint-check
#SBATCH --output=outputs_tflow_full/v2/logs/%x-%j.out
#SBATCH --error=outputs_tflow_full/v2/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/tools/validation_deps:$PWD"
export MIOPEN_USER_DB_PATH="$PWD/outputs_tflow_full/v2/cache/$SLURM_JOB_ID/performance_db"
export MIOPEN_CUSTOM_CACHE_DIR="$PWD/outputs_tflow_full/v2/cache/$SLURM_JOB_ID/kernel_cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs_tflow_full/v2/cache/$SLURM_JOB_ID/inductor"
export TRITON_CACHE_DIR="$PWD/outputs_tflow_full/v2/cache/$SLURM_JOB_ID/triton"
export MPLCONFIGDIR="$PWD/outputs_tflow_full/v2/cache/matplotlib"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
STUDY_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
"$STUDY_PYTHON" tools/experiment_entrypoint.py --module tools.validate_tflow_fresh_process --config configs/rafm_input_study/prepared/audiomnist_stft.json
"$STUDY_PYTHON" tools/experiment_entrypoint.py --module tools.validate_tflow_fresh_process --config configs/rafm_input_study/prepared/gaussian_aniso_d16_cor.json
