#!/usr/bin/env bash
# Checkpoint-only evaluation authorized by the user; no optimizer or training.
#SBATCH --partition=hermes-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=audio-empirical-gain
#SBATCH --output=outputs_audio_gain/logs/%x-%j.out
#SBATCH --error=outputs_audio_gain/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
AUDIO_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/tools/validation_deps:$PWD"
export MIOPEN_USER_DB_PATH="$PWD/outputs_audio_gain/cache/$SLURM_JOB_ID/performance_db"
export MIOPEN_CUSTOM_CACHE_DIR="$PWD/outputs_audio_gain/cache/$SLURM_JOB_ID/kernel_cache"
export MPLCONFIGDIR="$PWD/outputs_audio_gain/cache/matplotlib"
export AUDIO_GAIN_OUTPUT_DIR="$PWD/outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$MPLCONFIGDIR"
"$AUDIO_PYTHON" -m pytest -q tests/test_audio_empirical_gain.py tests/test_audio_gain_reporting.py \
  --junitxml=outputs_audio_gain/release_verification/audio_checks.xml
"$AUDIO_PYTHON" tools/evaluate_recovered_audio.py
