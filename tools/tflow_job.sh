#!/usr/bin/env bash
# Submit only after explicit approval of the corresponding phase in docs/run_plan.md.
#SBATCH --partition=hermes-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=tflow-prepared
set -euo pipefail
: "${TFLOW_PHASE:?Set sanity, tuning or final after approval}"
: "${TFLOW_CONDITION:?Set one reviewed condition id}"
if [[ "$TFLOW_CONDITION" == *[!a-zA-Z0-9_.-]* || "$TFLOW_CONDITION" == . || "$TFLOW_CONDITION" == .. ]]; then
  echo 'Invalid condition id' >&2
  exit 2
fi
TFLOW_ROOT=/mnt/vast01/users/fouad.oubari/msgm/rafm-additions
TFLOW_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
cd "$TFLOW_ROOT"
mkdir -p outputs_tflow/cache/{torch,inductor,triton,miopen,matplotlib}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME="$TFLOW_ROOT/outputs_tflow/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$TFLOW_ROOT/outputs_tflow/cache/inductor"
export TRITON_CACHE_DIR="$TFLOW_ROOT/outputs_tflow/cache/triton"
export MIOPEN_USER_DB_PATH="$TFLOW_ROOT/outputs_tflow/cache/miopen"
export MPLCONFIGDIR="$TFLOW_ROOT/outputs_tflow/cache/matplotlib"
TFLOW_CONFIG="configs/tflow/prepared/$TFLOW_CONDITION.json"
case "$TFLOW_PHASE" in
  sanity)
    exec "$TFLOW_PYTHON" -m experiments.tflow.sanity --config "$TFLOW_CONFIG"
    ;;
  tuning)
    exec "$TFLOW_PYTHON" -m experiments.tflow.tune --config "$TFLOW_CONFIG"
    ;;
  final)
    : "${TFLOW_SEED:?Set one original model seed}"
    exec "$TFLOW_PYTHON" -m experiments.tflow.run train-evaluate --config "$TFLOW_CONFIG" \
      --selection "outputs_tflow/tuning/$TFLOW_CONDITION/selection.json" --seed "$TFLOW_SEED"
    ;;
  *) echo 'TFLOW_PHASE must be sanity, tuning or final' >&2; exit 2 ;;
esac
