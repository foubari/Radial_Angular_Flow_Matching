#!/usr/bin/env bash
#SBATCH --partition=hermes-2
#SBATCH --nodelist=auh7-3b-gpu-015
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=none
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=rafm-tflow-report
#SBATCH --output=outputs_rafm_input_study/v1/logs/%x-%j.out
#SBATCH --error=outputs_rafm_input_study/v1/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/outputs_rafm_input_study/v1/cache/report_matplotlib"
mkdir -p "$MPLCONFIGDIR"
/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python tools/report_shared_study.py --plots
