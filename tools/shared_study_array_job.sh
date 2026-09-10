#!/usr/bin/env bash
# Resource/node/array flags are supplied by the reviewed Python planner.
# One process and one visible GPU per array task. No model or protocol overrides.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Expected immutable task manifest path, SHA-256, and phase' >&2
  exit 2
fi
: "${SLURM_JOB_ID:?A compute-node Slurm allocation is required}"
: "${SLURM_ARRAY_TASK_ID:?An array task ID is required}"
STUDY_ROOT=/mnt/vast01/users/fouad.oubari/msgm/rafm-additions
STUDY_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
cd "$STUDY_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$STUDY_ROOT/tools/validation_deps:$STUDY_ROOT"
exec "$STUDY_PYTHON" tools/launch_shared_study.py \
  --execute-worker "$SLURM_ARRAY_TASK_ID" --phase "$3" --manifest "$1" --manifest-sha256 "$2"
