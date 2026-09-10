"""Prepare/submit only the two recovered t-Flow conditions using frozen runtime.

Planning is standard-library only. The original executor retains per-run locks,
validation-selection receipts, resume identities, and independent sample audits.
Old 26-condition tasks are never included. Submission requires --submit and all
gates plus committed sources; planning alone never requests scheduler resources.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

import launch_shared_study as launch

ROOT = launch.ROOT
CONFIG_ROOT = ROOT / 'configs/rafm_input_study_recovered'
RECOVERY_ROOT = ROOT / 'outputs_tflow_full/recovered'
OUTPUT_ROOT = ROOT / 'outputs_tflow_full/v2'
EVIDENCE_ROOT = ROOT / 'outputs_tflow_full/v1'
BASE_PLAN = OUTPUT_ROOT / 'launch/tasks_20260910T080611_91887409.json'
BASE_SHA = '4485325dbc659a74aa875d1a1e69a3061014808d844a5d2ad4f8f04128ef2a92'
CONDITIONS = ('imagenette_dcae', 'piv_d32')
EXTRA_FILES = ('tools/launch_recovered_tflow.py', 'tools/verify_recovered_release.py',
    'tools/prepare_recovered_study.py', 'tools/check_recovered_image_evaluation.py',
    'tools/check_recovered_tflow.py', 'tools/validate_recovered_tflow_fresh_process.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def original_source_digest(original, name):
    if name in original['frozen_files']:
        return original['frozen_files'][name]
    payload = subprocess.check_output(['git', 'show', original['source_commit'] + ':' + name], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def image_gate(cfg, config_path, original):
    path = ROOT / 'docs/recovered_artifacts/image_evaluation_smoke.json'
    row = launch.load(path)
    require(row.get('status') == 'passed' and row.get('smoke_only') is True
        and row.get('benchmark_result') is False, 'Image evaluator smoke has not passed')
    require(row.get('config_file_sha256') == launch.digest(config_path), 'Image evaluator smoke used another configuration')
    require(row.get('source_sha256') == launch.digest(ROOT / 'tools/check_recovered_image_evaluation.py'),
        'Image evaluator smoke source changed')
    require(row.get('reference_unchanged') is True and row.get('all_image_and_latent_metrics_finite') is True,
        'Image evaluator smoke lacks finite metrics or unchanged reference proof')
    require(row.get('original_reference', {}).get('sha256') == cfg['evaluation']['expected_reference_sha256']
        and row.get('original_reference', {}).get('count') == 3925, 'Image evaluator smoke reference differs')
    require(row.get('evaluation_proof', {}).get('protocol', {}).get('source_sha256')
        == launch.digest(ROOT / 'experiments/image_latents/dit/dit_eval_sit.py'), 'Image evaluator source proof differs')
    for name in launch.IMAGE_FILES + ['baselines/tflow_downstream.py']:
        require(launch.digest(ROOT / name) == original_source_digest(original, name),
            'Image evaluation differs from the original t-Flow implementation: ' + name)
    return launch.file_record(path)


def image_fresh_process_gate(cfg, implementation_hash):
    path = RECOVERY_ROOT / 'entrypoint_checks/imagenette_dcae/check.json'
    row = launch.load(path)
    require(row.get('status') == 'passed' and row.get('condition_id') == cfg['condition_id'],
        'Recovered image public-trainer/resume check has not passed')
    require(row.get('config_sha256') == launch.canonical_hash(cfg)
        and row.get('implementation_sha256') == implementation_hash, 'Image public-trainer check used another config/runtime')
    sources = {'entrypoint_sha256': 'tools/experiment_entrypoint.py',
        'validator_sha256': 'tools/validate_recovered_tflow_fresh_process.py',
        'original_validator_sha256': 'tools/validate_tflow_fresh_process.py'}
    for key, source in sources.items():
        require(row.get(key) == launch.digest(ROOT / source), 'Image public-trainer check source differs: ' + source)
    for key in ('gpu_initialized_before_training', 'checkpoint_load', 'resume_bitwise_identical', 'finite_samples'):
        require(row.get(key) is True, 'Image public-trainer check is missing ' + key)
    require(row.get('test_data_used') is False and row.get('source_selection_performed') is False
        and row.get('benchmark_metrics_computed') is False,
        'Image public-trainer check must not select a source or inspect test data')
    expected = {'initial_updates': 4, 'resumed_updates': 2, 'uninterrupted_updates': 6,
        'disposable_optimizer_updates_total': 12, 'n_generated': 40, 'class_counts': [4] * 10,
        'nfe': cfg['evaluation']['model_evaluations'], 'model_calls_total': 100, 'n_batches': 1,
        'batch_size': cfg['training']['batch_size'], 'precision': cfg['training']['precision'],
        'ema': cfg['training']['ema'], 'seed': 46021, 'sample_seed': 61717}
    for key, value in expected.items():
        require(row.get(key) == value, 'Image public-trainer check violates agreed tiny budget/count: ' + key)
    require(row.get('source', {}).get('nu') == 5
        and row.get('source_calibration', {}).get('scale_multiplier') == 1,
        'Image public-trainer check must use the fixed untuned source setting')
    return launch.file_record(path)


def phase(name, first, last, workers, *, dependency=None):
    indices = list(range(first, last + 1))
    count = min(workers, len(indices))
    row = {'name': name, 'first': first, 'last': last, 'worker_count': count,
        'shards': [{'worker_index': i, 'task_indices': indices[i::count]} for i in range(count)]}
    if dependency:
        row['dependency'] = dependency
    return row


def existing_outcomes(condition):
    """Preserve prior attempts; resume their reviewed manifest, never resubmit."""
    directories = (OUTPUT_ROOT / 'tuning' / condition, OUTPUT_ROOT / 'final' / condition)
    return [str(path) for directory in directories if directory.exists()
            for path in directory.rglob('*') if path.is_file()]


def make_plan(node, workers=6, time_limit='04:00:00', extra_sources=()):
    require(re.fullmatch(r'auh7-3b-gpu-[0-9]{3}', node) is not None, 'Node must be an explicit cluster GPU hostname')
    require(1 <= workers <= 6, 'Use one to six independent single-GPU workers')
    require(launch.digest(BASE_PLAN) == BASE_SHA, 'Original t-Flow manifest changed')
    original = launch.load(BASE_PLAN)
    require(original.get('study') == 'tflow', 'Original manifest is not t-Flow')
    for name, expected in original['frozen_files'].items():
        require(launch.digest(ROOT / name) == expected, 'Original frozen source/config changed: ' + name)
    preparation_path = CONFIG_ROOT / 'materialization.json'
    preparation = launch.load(preparation_path)
    verified = {row['condition_id']: row for row in preparation['rows'] if row.get('status') == 'verified'}
    require(all(name in verified for name in CONDITIONS), 'Both recovered inputs must have passed preparation')
    receipt_path = ROOT / 'docs/recovered_artifacts/release_verification.json'
    require(launch.load(receipt_path).get('status') == 'passed', 'Release checksum audit has not passed')
    require(preparation.get('release_verification', {}).get('sha256') == launch.digest(receipt_path),
        'Preparation used a different release-verification receipt')
    plan = copy.deepcopy(original)
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), node=node,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        materialization=launch.file_record(preparation_path), blocked_conditions={},
        tasks=[], blocking_issues=[], submitted=False, time_limit=time_limit,
        max_concurrent_tasks=workers, condition_count=2,
        recovery={'original_manifest': launch.file_record(BASE_PLAN),
            'scope': 'Only the two recovered t-Flow conditions; existing A/B/C and 26-condition t-Flow outcomes unchanged',
            'conditions': list(CONDITIONS), 'selection_policy': 'Original validation-only 0.55 full-run equivalents per condition',
            'no_test_based_retuning': True})
    plan['launch_gates'] = launch.unit_gate('tflow', EVIDENCE_ROOT) + launch.entrypoint_gate()
    plan['launch_gates'] += [launch.file_record(receipt_path), launch.file_record(preparation_path)]
    for name in (*EXTRA_FILES, *extra_sources):
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.suffix in ('.py', '.sh'),
            'Extra source must be a repository-relative Python/shell file')
        require((ROOT / path).is_file(), 'Missing extra source: ' + str(path))
        plan['frozen_files'][str(path)] = launch.digest(ROOT / path)
    finals = []
    for condition in CONDITIONS:
        previous = existing_outcomes(condition)
        require(not previous, 'Recovered t-Flow condition already has output; inspect/resume its existing manifest: '
                + condition + ' (' + ', '.join(previous[:3]) + ')')
        path = CONFIG_ROOT / 'prepared' / (condition + '.json')
        cfg = launch.load(path)
        require(cfg.get('condition_id') == condition and cfg.get('protocol_status') == 'resolved'
                and not cfg.get('blocking_issues'), 'Recovered configuration is unresolved: ' + condition)
        require(launch.digest(path) == verified[condition]['config']['sha256'], 'Recovered config differs from preparation audit')
        cache_path = Path(cfg['shared_cache']['manifest_path'])
        require(launch.digest(cache_path) == cfg['shared_cache']['manifest_sha256']
                == verified[condition]['shared_cache_manifest']['sha256'], 'Recovered shared-cache manifest changed')
        cache = launch.load(cache_path)
        recorded_cfg = copy.deepcopy(cache['prepared_config'])
        recorded_cfg['shared_cache']['manifest_sha256'] = cfg['shared_cache']['manifest_sha256']
        require(recorded_cfg == cfg and cache.get('status') == 'complete', 'Config does not match immutable recovered cache')
        implementation = launch.runtime_manifest('tflow', cfg)
        for name, expected in implementation['files'].items():
            if Path(name).is_absolute():
                continue  # External SiT revision/bytes are pinned below and by its builder.
            require(expected == original_source_digest(original, name), 'Recovered t-Flow runtime changed: ' + name)
        impl_hash = launch.canonical_hash(implementation)
        gates = launch.sanity_gate('tflow', EVIDENCE_ROOT, cfg, impl_hash)
        gates.append(launch.file_record(cache_path))
        if cfg['kind'] == 'image':
            gates.append(image_gate(cfg, path, original))
            gates.append(image_fresh_process_gate(cfg, impl_hash))
            backbone = launch.load(ROOT / 'docs/artifact_audit/backbone_contract_checks.json')
            image_rows = [row for row in backbone.get('checks', []) if row.get('kind') == 'image_sit']
            require(backbone.get('status') == 'passed' and len(image_rows) == 1 and image_rows[0].get('finite') is True,
                'Original raw SiT backbone interface check is absent')
            provenance = image_rows[0]['provenance']
            external = Path(cfg['model']['source_dir'])
            require(provenance['source_sha256'] == launch.digest(external / 'models.py')
                    and provenance['license_sha256'] == launch.digest(external / 'LICENSE.txt'), 'Original raw SiT source/license changed')
            require(provenance['commit'] == subprocess.check_output(['git', '-C', str(external), 'rev-parse', 'HEAD'], text=True).strip(),
                'Original raw SiT revision changed')
            gates.append(launch.file_record(ROOT / 'docs/artifact_audit/backbone_contract_checks.json'))
        plan['frozen_files'].update(implementation['files'])
        plan['frozen_files'][str(path.relative_to(ROOT))] = launch.digest(path)
        base = {'condition_id': condition, 'config_path': str(path),
            'config_sha256': launch.canonical_hash(cfg), 'config_file_sha256': launch.digest(path),
            'config': cfg, 'implementation_sha256': impl_hash, 'implementation': implementation,
            'sanity_gates': gates}
        tune = [str(launch.PYTHON), str(ROOT / 'tools/experiment_entrypoint.py'), '--module', 'experiments.tflow.tune',
            '--config', str(path), '--output', str(OUTPUT_ROOT / 'tuning')]
        plan['tasks'].append({**base, 'index': len(plan['tasks']), 'phase': 'tuning', 'commands': [tune]})
        for seed in cfg['seeds']:
            command = [str(launch.PYTHON), str(ROOT / 'tools/experiment_entrypoint.py'), '--module', 'experiments.tflow.run',
                'train-evaluate', '--config', str(path), '--selection', str(OUTPUT_ROOT / 'tuning' / condition / 'selection.json'),
                '--seed', str(seed), '--output', str(OUTPUT_ROOT / 'final')]
            finals.append({**base, 'phase': 'final', 'seed': seed, 'commands': [command]})
    for task in finals:
        plan['tasks'].append({**task, 'index': len(plan['tasks'])})
    require(len(plan['tasks']) == 8, 'Recovered t-Flow scope must be exactly two tuning and six final tasks')
    plan['array_phases'] = [phase('tuning', 0, 1, workers),
        phase('final', 2, 7, workers, dependency='afterany:tuning_array')]
    plan['task_count'] = len(plan['tasks'])
    plan['blocking_issues'] = launch.frozen_files(plan['source_commit'], plan['frozen_files'])
    plan['ready_to_submit'] = not plan['blocking_issues']
    return plan


def submit(plan, manifest_path, manifest_hash, afterany=''):
    launch.verify_plan(plan)
    directory = RECOVERY_ROOT / 'launch'
    # Exclusive per-condition claims are retained after partial submissions.
    # A concurrent invocation can never schedule the same condition twice.
    for condition in CONDITIONS:
        require(not (directory / (condition + '.submission_claim.json')).exists(),
            'Recovered t-Flow condition already has a submission claim: ' + condition)
        require(not existing_outcomes(condition), 'Recovered t-Flow output appeared after planning: ' + condition)
    for condition in CONDITIONS:
        launch.write(directory / (condition + '.submission_claim.json'),
            {'manifest': str(manifest_path), 'manifest_sha256': manifest_hash,
             'created_utc': datetime.now(timezone.utc).isoformat()}, immutable=True)
    logs = RECOVERY_ROOT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    record = {'manifest': str(manifest_path), 'manifest_sha256': manifest_hash,
        'submitted_utc': datetime.now(timezone.utc).isoformat(), 'arrays': []}
    receipt_path = manifest_path.with_suffix('.submission.json')
    launch.write(receipt_path, record, immutable=True)
    preceding = None
    for stage in plan['array_phases']:
        count = stage['worker_count']
        command = ['sbatch', '--parsable', '--partition=hermes-2', '--nodelist=' + plan['node'],
            '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--gres=gpu:mi210:1', '--mem=32G',
            '--time=' + plan['time_limit'], f'--array=0-{count-1}%{count}', '--no-requeue',
            '--job-name=tflow-recovered-' + stage['name'], '--output=' + str(logs / '%x-%A_%a.out'),
            '--error=' + str(logs / '%x-%A_%a.err')]
        dependency = preceding or afterany
        if dependency:
            command.append('--dependency=afterany:' + dependency)
        command += [str(ROOT / 'tools/shared_study_array_job.sh'), str(manifest_path), manifest_hash, stage['name']]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        row = {'phase': stage['name'], 'command': command, 'exit_code': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
        if result.returncode == 0:
            job = result.stdout.strip().split(';')[0]
            if job.isdigit():
                row['job_id'] = job
                preceding = job
            else:
                row['failure'] = 'Unrecognized scheduler response; inspect scheduler before retry'
        record['arrays'].append(row)
        launch.write(receipt_path, record)
        print(json.dumps(row), flush=True)
        if result.returncode or 'job_id' not in row:
            raise SystemExit('Partial/failed recovered t-Flow submission recorded; inspect existing jobs before any retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', default='auh7-3b-gpu-030', help='Verified MI210 hostname; root must check availability before submitting')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--time-limit', default='04:00:00')
    parser.add_argument('--afterany', default='', help='Optional existing scheduler jobs, separated by colons')
    parser.add_argument('--extra-source', action='append', default=[], help='Additional repository-relative check wrapper to freeze')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'auh7-3b-gpu-[0-9]{3}', args.node):
        parser.error('Node must match auh7-3b-gpu-NNN')
    if not 1 <= args.workers <= 6:
        parser.error('Use one to six independent single-GPU workers')
    if args.afterany and not re.fullmatch(r'[0-9]+(?::[0-9]+)*', args.afterany):
        parser.error('--afterany accepts scheduler job IDs separated by colons')
    plan = make_plan(args.node, args.workers, args.time_limit, args.extra_source)
    directory = RECOVERY_ROOT / 'launch'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('tasks_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.json')
    launch.write(path, plan, immutable=True)
    digest = launch.digest(path)
    print(json.dumps({'manifest': str(path), 'sha256': digest, 'tasks': plan['task_count'],
        'phase_workers': {stage['name']: stage['worker_count'] for stage in plan['array_phases']},
        'node': plan['node'], 'ready_to_submit': plan['ready_to_submit'], 'issues': plan['blocking_issues']}), flush=True)
    if args.submit:
        submit(plan, path, digest, args.afterany)


if __name__ == '__main__':
    main()
