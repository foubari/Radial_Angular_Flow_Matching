#!/usr/bin/env bash
#SBATCH --partition=hermes-2
#SBATCH --nodelist=auh7-3b-gpu-008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=tflow-study-sanity
#SBATCH --output=outputs_tflow_full/v1/logs/%x-%j.out
#SBATCH --error=outputs_tflow_full/v1/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/tools/validation_deps:$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MIOPEN_USER_DB_PATH="$PWD/outputs_tflow_full/v1/cache/$SLURM_JOB_ID/performance_db"
export MIOPEN_CUSTOM_CACHE_DIR="$PWD/outputs_tflow_full/v1/cache/$SLURM_JOB_ID/kernel_cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs_tflow_full/v1/cache/$SLURM_JOB_ID/inductor"
export TRITON_CACHE_DIR="$PWD/outputs_tflow_full/v1/cache/$SLURM_JOB_ID/triton"
export MPLCONFIGDIR="$PWD/outputs_tflow_full/v1/cache/matplotlib"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
STUDY_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
"$STUDY_PYTHON" -m pytest -q tests/test_tflow_core.py tests/test_tflow_runtime.py tests/test_tflow_tuning.py tests/test_tflow_validation_policy.py tests/test_tflow_reporting.py --junitxml=outputs_tflow_full/v1/unit_checks.xml
"$STUDY_PYTHON" - <<'PY'
import json,subprocess,sys
from pathlib import Path
rows=[]
configs=sorted(Path('configs/rafm_input_study/prepared').glob('*.json'))
for p in configs:
 print('SANITY',p.stem,flush=True)
 r=subprocess.run([sys.executable,'-m','experiments.tflow.sanity','--config',str(p),'--output','outputs_tflow_full/v1/sanity'])
 rows.append({'condition':p.stem,'exit_code':r.returncode})
 Path('outputs_tflow_full/v1/sanity_jobs.json').write_text(json.dumps(rows,indent=2)+'\n')
if not configs or any(x['exit_code'] for x in rows): sys.exit(1)
PY
