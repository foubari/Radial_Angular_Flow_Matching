"""Run recovered-condition A/B/C sanity and the offline image evaluator on Slurm."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Recovered smokes require a compute-node GPU allocation')
    rows = []
    for condition in ('piv_d32', 'imagenette_dcae'):
        cfg = ROOT / 'configs/rafm_input_study_recovered/prepared' / (condition + '.json')
        command = [sys.executable, str(ROOT / 'tools/experiment_entrypoint.py'), '--module',
                   'experiments.rafm_inputs.run', 'sanity', '--config', str(cfg),
                   '--output', str(ROOT / 'outputs_rafm_input_study/v1')]
        print('RECOVERED_SANITY', condition, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        rows.append({'condition': condition, 'command': command, 'exit_code': result.returncode})
    if rows[-1]['exit_code'] == 0:
        command = [sys.executable, str(ROOT / 'tools/check_recovered_image_evaluation.py'),
                   '--config', str(ROOT / 'configs/rafm_input_study_recovered/prepared/imagenette_dcae.json')]
        result = subprocess.run(command, cwd=ROOT)
        rows.append({'condition': 'image_evaluator', 'command': command, 'exit_code': result.returncode})
    output = ROOT / 'outputs_rafm_input_study/recovered' / ('smokes_' + os.environ['SLURM_JOB_ID'] + '.json')
    output.write_text(json.dumps({'rows': rows, 'status': 'passed' if all(x['exit_code'] == 0 for x in rows) else 'failed'}, indent=2) + '\n')
    raise SystemExit(0 if all(x['exit_code'] == 0 for x in rows) else 1)


if __name__ == '__main__':
    main()
