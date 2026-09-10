#!/usr/bin/env bash
# Submit after the recovered array; this job requests CPU resources only.
set -euo pipefail
: "${SLURM_JOB_ID:?A compute-node Slurm allocation is required}"
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/outputs_rafm_input_study/complete_report/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
/usr/bin/python3 - <<'PY'
import datetime,json,subprocess
from pathlib import Path
jobs=set()
for name in ('outputs_tflow_full/v2','outputs_rafm_input_study/v1','outputs_rafm_input_study/recovered'):
    for path in (Path(name)/'launch').glob('*.submission.json'):
        record=json.loads(path.read_text())
        jobs.update(str(row['job_id']) for row in record.get('arrays',[]) if row.get('job_id'))
        if record.get('job_id'): jobs.add(str(record['job_id']))
command=['sacct','-j',','.join(sorted(jobs)),'--parsable2','--noheader',
         '--format=JobID,JobName,State,Elapsed,ExitCode,NodeList,AllocTRES']
result=subprocess.run(command,capture_output=True,text=True,check=False)
record={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_ids':sorted(jobs),'command':command,'exit_code':result.returncode,
        'accounting':result.stdout,'stderr':result.stderr}
target=Path('outputs_rafm_input_study/complete_report/final_scheduler_accounting.json')
target.write_text(json.dumps(record,indent=2)+'\n')
PY
/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python tools/report_shared_study.py \
  --config-root configs/rafm_input_study_complete --output outputs_rafm_input_study/complete_report --plots
/usr/bin/python3 tools/summarize_shared_study.py \
  --report outputs_rafm_input_study/complete_report/report.json
/usr/bin/python3 tools/report_rafm_completion.py --require-finished
