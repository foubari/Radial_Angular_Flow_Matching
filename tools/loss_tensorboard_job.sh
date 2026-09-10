#!/usr/bin/env bash
#SBATCH --partition=hermes-2
#SBATCH --nodelist=auh7-3b-gpu-015
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --job-name=rafm-loss-tensorboard
#SBATCH --output=outputs_monitoring/logs/%x-%j.out
#SBATCH --error=outputs_monitoring/logs/%x-%j.err
set -euo pipefail
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MONITOR_PYTHON="$PWD/outputs_monitoring/.venv/bin/python"
MONITOR_SESSION="$PWD/outputs_monitoring/sessions/$SLURM_JOB_ID"
mkdir -p "$MONITOR_SESSION/events"
"$MONITOR_PYTHON" tools/check_loss_tensorboard_bridge.py
MONITOR_PORT="$($MONITOR_PYTHON - <<'PY'
import socket
for port in range(6010,6020):
    with socket.socket() as sock:
        try: sock.bind(('0.0.0.0',port))
        except OSError: continue
        print(port); break
else: raise SystemExit('No free TensorBoard port in 6010..6019')
PY
)"
"$MONITOR_PYTHON" - "$MONITOR_SESSION" "$MONITOR_PORT" <<'PY'
import datetime,json,os,pathlib,socket,sys
session=pathlib.Path(sys.argv[1]);port=int(sys.argv[2])
record={'slurm_job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'port':port,
        'session':str(session),'logdir':str(session/'events'),'gpus':0,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'ssh_tunnel_command':f'ssh -N -L {port}:{socket.gethostname()}:{port} m3',
        'browser_url':f'http://localhost:{port}/#scalars','poll_seconds':15,
        'training_files_modified':False}
(session/'service.json').write_text(json.dumps(record,indent=2)+'\n')
target=pathlib.Path('outputs_monitoring/service.json');temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(target)
print(json.dumps(record,indent=2),flush=True)
PY
"$MONITOR_PYTHON" tools/loss_tensorboard_bridge.py --output "$MONITOR_SESSION" --interval 15 &
BRIDGE_PID=$!
"$MONITOR_PYTHON" -m tensorboard.main --logdir "$MONITOR_SESSION/events" --host 0.0.0.0 --port "$MONITOR_PORT" --reload_interval 15 --load_fast=false --samples_per_plugin=scalars=20000 &
TENSORBOARD_PID=$!
cleanup() { kill "$BRIDGE_PID" "$TENSORBOARD_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
wait -n "$BRIDGE_PID" "$TENSORBOARD_PID"
