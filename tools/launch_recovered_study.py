"""Add only verified recovered A/B/C conditions to the existing frozen study.

Uses the original task executor, locks, model commands, and sample auditor.
Never submits the already-launched 26-condition sweep again.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import launch_shared_study as launch

ROOT = launch.ROOT
CONFIG_ROOT = ROOT / 'configs/rafm_input_study_recovered'
RECOVERY_ROOT = ROOT / 'outputs_rafm_input_study/recovered'
BASE_PLAN = ROOT / 'outputs_rafm_input_study/v1/launch/tasks_20260910T080615_c1cc6a41.json'
BASE_SHA = 'd94e88fa8135d786d7457e72644a797840272e614485cb11cb647ca9da482d41'
EXTRA_FILES = ['tools/launch_recovered_study.py', 'tools/recovered_study_job.sh',
               'tools/verify_recovered_release.py', 'tools/prepare_recovered_study.py',
               'tools/check_recovered_conditions.py']


def make_plan(conditions, workers, time_limit, node):
    if launch.digest(BASE_PLAN) != BASE_SHA:
        raise ValueError('Original study manifest changed')
    plan = copy.deepcopy(launch.load(BASE_PLAN))
    preparation_path = CONFIG_ROOT / 'materialization.json'
    preparation = launch.load(preparation_path)
    verified = {row['condition_id']: row for row in preparation['rows'] if row['status'] == 'verified'}
    if any(condition not in verified for condition in conditions):
        raise ValueError('Requested condition has no successful recovered-data audit')
    receipt = ROOT / 'docs/recovered_artifacts/release_verification.json'
    if launch.load(receipt).get('status') != 'passed':
        raise ValueError('Recovered release checksum audit has not passed')
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), node=node,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                materialization=launch.file_record(preparation_path), blocked_conditions={},
                tasks=[], blocking_issues=[], submitted=False, time_limit=time_limit,
                recovery={'original_manifest': launch.file_record(BASE_PLAN),
                          'scope': 'A/B/C only, newly recovered conditions; existing results preserved',
                          'conditions': conditions, 'previous_outcomes': []})
    evidence = ROOT / 'outputs_rafm_input_study/v1'
    plan['launch_gates'] = launch.unit_gate('rafm_inputs', evidence) + launch.entrypoint_gate()
    plan['launch_gates'] += [launch.file_record(receipt), launch.file_record(preparation_path)]
    for name in EXTRA_FILES:
        plan['frozen_files'][name] = launch.digest(ROOT / name)
    for condition in sorted(conditions, key=lambda c: launch.priority(CONFIG_ROOT / 'prepared' / f'{c}.json')):
        path = CONFIG_ROOT / 'prepared' / f'{condition}.json'
        cfg = launch.load(path)
        if cfg['condition_id'] != condition or cfg['protocol_status'] != 'resolved' or cfg['blocking_issues']:
            raise ValueError('Recovered condition is not resolved')
        if launch.digest(path) != verified[condition]['config']['sha256']:
            raise ValueError('Prepared configuration differs from successful audit')
        implementation = launch.runtime_manifest('rafm_inputs', cfg)
        impl_hash = launch.canonical_hash(implementation)
        gates = launch.sanity_gate('rafm_inputs', evidence, cfg, impl_hash)
        if cfg['kind'] == 'image':
            image_gate = ROOT / 'docs/recovered_artifacts/image_evaluation_smoke.json'
            check = launch.load(image_gate)
            if check.get('status') != 'passed' or check.get('config_file_sha256') != launch.digest(path):
                raise ValueError('Recovered image evaluator smoke has not passed for this configuration')
            if check.get('source_sha256') != launch.digest(ROOT / 'tools/check_recovered_image_evaluation.py'):
                raise ValueError('Image evaluator smoke script changed after its check')
            if (check.get('reference_unchanged') is not True
                    or check.get('all_image_and_latent_metrics_finite') is not True
                    or check.get('original_reference', {}).get('sha256') != cfg['evaluation']['expected_reference_sha256']
                    or check.get('evaluation_proof', {}).get('protocol', {}).get('source_sha256')
                    != launch.digest(ROOT / 'experiments/image_latents/dit/dit_eval_sit.py')):
                raise ValueError('Image evaluator smoke reference, source, or numerical proof is incomplete')
            # The recovered branch intentionally keeps the original evaluator
            # bytes. Bind this smoke to that fixed implementation, not merely a
            # successful receipt left over after a future evaluator edit.
            original_plan = launch.load(BASE_PLAN)
            original = original_plan['frozen_files']
            for name in launch.IMAGE_FILES + ['baselines/tflow_downstream.py']:
                # Images were blocked in the original task list, so image-only
                # files were not in its runtime manifest. Check their exact
                # bytes at that pinned source commit instead.
                expected = original.get(name)
                if expected is None:
                    expected = hashlib.sha256(subprocess.check_output(
                        ['git', 'show', original_plan['source_commit'] + ':' + name], cwd=ROOT)).hexdigest()
                if launch.digest(ROOT / name) != expected:
                    raise ValueError('Recovered evaluator differs from original frozen implementation: ' + name)
            gates.append(launch.file_record(image_gate))
            plan['frozen_files']['tools/check_recovered_image_evaluation.py'] = check['source_sha256']
        plan['frozen_files'].update(implementation['files'])
        plan['frozen_files'][str(path.relative_to(ROOT))] = launch.digest(path)
        base = {'condition_id': condition, 'config_path': str(path),
                'config_sha256': launch.canonical_hash(cfg), 'config_file_sha256': launch.digest(path),
                'config': cfg, 'implementation_sha256': impl_hash, 'implementation': implementation,
                'sanity_gates': gates, 'phase': 'final'}
        for seed in cfg['seeds']:
            for arm in ('A', 'B', 'C'):
                previous = evidence / 'final' / condition / arm / f'seed_{seed}' / 'result.json'
                if previous.exists():
                    old = launch.load(previous)
                    plan['recovery']['previous_outcomes'].append({'path': str(previous), 'status': old.get('status')})
                    # Do not retry scientific failures or duplicate successes.
                    continue
                command = [str(launch.PYTHON), str(ROOT / 'tools/experiment_entrypoint.py'),
                           '--module', 'experiments.rafm_inputs.run', 'train-evaluate',
                           '--config', str(path), '--arm', arm, '--seed', str(seed), '--output', str(evidence)]
                plan['tasks'].append({**base, 'index': len(plan['tasks']), 'seed': seed, 'arm': arm,
                                      'commands': [command]})
    if not plan['tasks']:
        raise ValueError('No untouched recovered-condition tasks remain to submit')
    count = min(workers, len(plan['tasks']))
    phase = {'name': 'final', 'first': 0, 'last': len(plan['tasks']) - 1, 'worker_count': count,
             'shards': [{'worker_index': i, 'task_indices': list(range(i, len(plan['tasks']), count))}
                        for i in range(count)]}
    plan.update(array_phases=[phase], condition_count=len(conditions), task_count=len(plan['tasks']),
                max_concurrent_tasks=count)
    plan['blocking_issues'] = launch.frozen_files(plan['source_commit'], plan['frozen_files'])
    plan['ready_to_submit'] = not plan['blocking_issues']
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('piv_d32', 'imagenette_dcae'), action='append')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--node', choices=('auh7-3b-gpu-008', 'auh7-3b-gpu-015', 'auh7-3b-gpu-029'), default='auh7-3b-gpu-029',
                        help='One verified MI210 node; at most six GPUs total after the existing ABC array')
    parser.add_argument('--time-limit', default='48:00:00')
    parser.add_argument('--afterany', default='765748', help='Existing ABC array; retain node concurrency limit')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('Use one to six single-GPU workers')
    conditions = args.condition or ['piv_d32', 'imagenette_dcae']
    if len(conditions) != len(set(conditions)):
        parser.error('Duplicate conditions')
    if args.afterany and not all(x.isdigit() for x in args.afterany.split(':')):
        parser.error('Dependency must contain scheduler job IDs only')
    plan = make_plan(conditions, args.workers, args.time_limit, args.node)
    directory = RECOVERY_ROOT / 'launch'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('tasks_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.json')
    launch.write(path, plan, immutable=True)
    digest = launch.digest(path)
    print(json.dumps({'manifest': str(path), 'sha256': digest, 'tasks': plan['task_count'],
                      'ready_to_submit': plan['ready_to_submit'], 'issues': plan['blocking_issues']}), flush=True)
    if not args.submit:
        return
    launch.verify_plan(plan)
    # Claim each condition, including when a previous submission attempt failed;
    # explicit scheduler inspection is needed before any retry.
    for condition in conditions:
        claim = directory / (condition + '.submission_claim.json')
        if claim.exists():
            raise RuntimeError(f'Recovered condition already has a submission claim: {claim}')
    for condition in conditions:
        launch.write(directory / (condition + '.submission_claim.json'),
                     {'manifest': str(path), 'sha256': digest}, immutable=True)
    logs = RECOVERY_ROOT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    count = plan['array_phases'][0]['worker_count']
    command = ['sbatch', '--parsable', '--partition=hermes-2', '--nodelist=' + plan['node'],
               '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--gres=gpu:mi210:1', '--mem=32G',
               '--time=' + args.time_limit, f'--array=0-{count-1}%{count}', '--no-requeue',
               '--job-name=rafm-recovered-final', '--output=' + str(logs / '%x-%A_%a.out'),
               '--error=' + str(logs / '%x-%A_%a.err')]
    if args.afterany:
        command += ['--dependency=afterany:' + args.afterany]
    command += [str(ROOT / 'tools/shared_study_array_job.sh'), str(path), digest, 'final']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    submission = {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout.strip(),
                  'stderr': result.stderr.strip(), 'manifest': str(path), 'manifest_sha256': digest}
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unrecognized scheduler response; inspect scheduler before retry')
        submission['job_id'] = job
    launch.write(path.with_suffix('.submission.json'), submission, immutable=True)
    print(json.dumps(submission), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
