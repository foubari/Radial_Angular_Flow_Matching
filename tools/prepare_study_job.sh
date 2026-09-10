#!/usr/bin/env bash
#SBATCH --partition=hermes-2
#SBATCH --nodelist=auh7-3b-gpu-015
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=none
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=rafm-study-prepare
#SBATCH --output=outputs_rafm_input_study/v1/logs/%x-%j.out
#SBATCH --error=outputs_rafm_input_study/v1/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/tools/validation_deps:$PWD"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
STUDY_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
"$STUDY_PYTHON" tools/prepare_shared_study.py --materialize
