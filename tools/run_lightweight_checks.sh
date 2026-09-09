#!/usr/bin/env bash
# CPU-only correctness/document-render checks. No optimizer loop or benchmark.
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
TFLOW_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/tools/validation_deps:$PWD"
export MPLCONFIGDIR="$PWD/tools/validation_deps/matplotlib_cache"
"$TFLOW_PYTHON" -m pytest -q tests/test_tflow_*.py tests/test_audio*gain*.py \
  --junitxml=docs/artifact_audit/unit_checks.xml
"$TFLOW_PYTHON" tools/check_backbone_contracts.py
"$TFLOW_PYTHON" -m experiments.poc_audio.render_gain_results \
  --reference-aggregate experiments/poc_audio/stage2_3seed.json \
  --pending --output-dir docs/proposed_audio
