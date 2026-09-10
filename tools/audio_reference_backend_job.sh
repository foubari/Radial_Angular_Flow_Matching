#!/usr/bin/env bash
#SBATCH --partition=hermes-2
#SBATCH --nodelist=auh7-3b-gpu-008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=audio-reference-backend
#SBATCH --output=outputs_rafm_input_study/v1/logs/%x-%j.out
#SBATCH --error=outputs_rafm_input_study/v1/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/outputs_rafm_input_study/v1/cache/audio_reference_backend/$SLURM_JOB_ID"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/performance_db"
export MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/kernel_cache"
mkdir -p "$XDG_CACHE_HOME" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python tools/check_audio_reference_backend.py
