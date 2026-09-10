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
/usr/bin/python3 - <<'PY'
import datetime,json,subprocess
from pathlib import Path
roots=[Path('outputs_tflow_full/v2'),Path('outputs_rafm_input_study/v1')]
jobs=[]
for root in roots:
 for path in (root/'launch').glob('*.submission.json'):
  jobs.extend(str(row['job_id']) for row in json.loads(path.read_text())['arrays'] if row.get('job_id'))
if jobs:
 result=subprocess.run(['sacct','-j',','.join(jobs),'--parsable2','--noheader','--format=JobID,JobName,State,Elapsed,ExitCode,NodeList'],capture_output=True,text=True)
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 record={'captured_utc':stamp,'job_ids':jobs,'command_exit_code':result.returncode,'accounting':result.stdout,'stderr':result.stderr}
 for root in roots:
  target=root/'scheduler_snapshots'/'final_accounting.json'
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(record,indent=2)+'\n')
PY
/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python tools/report_shared_study.py --plots
