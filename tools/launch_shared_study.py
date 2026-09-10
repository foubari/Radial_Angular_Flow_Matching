"""Prepare immutable Slurm task manifests; submit only with explicit --submit.

Planning uses only the standard library. Runtime implementation imports occur
only for --execute-task inside a Slurm compute allocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / 'msgm-sparse-control/.venv/bin/python'
CONFIG_DIR = ROOT / 'configs/rafm_input_study/prepared'
STUDIES = {
    'tflow': {'node': 'auh7-3b-gpu-008', 'output': 'outputs_tflow_full/v1'},
    'rafm_inputs': {'node': 'auh7-3b-gpu-015', 'output': 'outputs_rafm_input_study/v1'},
}
COMMON_FILES = [
    'baselines/tflow_core.py', 'baselines/tflow_downstream.py',
    'experiments/tflow/run.py', 'experiments/tflow/data.py',
    'experiments/tflow/validation.py', 'experiments/tflow/tune.py', 'rafm/utils/seeds.py',
]
VECTOR_FILES = ['rafm/models/mlp.py', 'rebuttal_experiments/lib/resmlp.py',
                'rafm/metrics/radial.py', 'rafm/metrics/distributional.py',
                'rafm/metrics/angular.py', 'rafm/metrics/stability.py']
AUDIO_FILES = ['experiments/poc_audio/audio_flow.py', 'experiments/poc_audio/audio_classifier.py',
               'experiments/poc_audio/audio_empirical_gain.py']
IMAGE_FILES = ['experiments/image_latents/dit/dit_eval_sit.py', 'rafm/metrics/radial.py',
               'rafm/metrics/distributional.py']
ABC_FILES = ['experiments/rafm_inputs/run.py', 'baselines/rafm_input_parameterization.py',
             'rafm/flow_matching/loss.py', 'rafm/flow_matching/sampler.py',
             'rafm/paths/spherical_geodesic.py', 'rafm/utils/sphere.py',
             'rafm/sources/radial_empirical.py']
ORCHESTRATION_FILES = ['tools/launch_shared_study.py', 'tools/shared_study_array_job.sh',
                       'tools/audit_study_samples.py']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if immutable:
        with path.open('x') as stream:
            stream.write(data)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(data)
        temporary.replace(path)


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def runtime_manifest(study, cfg):
    """Mirror the runtime's declarative source manifest without importing ML.

    The compute worker compares against the actual runtime function before any
    optimizer or sampler command, so drift in this list is a hard failure.
    """
    names = COMMON_FILES + {'vector': VECTOR_FILES, 'audio': AUDIO_FILES, 'image': IMAGE_FILES}[cfg['kind']]
    if study == 'rafm_inputs':
        names += ABC_FILES
    files = {name: digest(ROOT / name) for name in names}
    if cfg['kind'] == 'image':
        external = Path(cfg['model'].get('source_dir', cfg['model'].get('sit_repo', '')))
        for name in ('models.py', 'LICENSE.txt'):
            files[str((external / name).resolve())] = digest(external / name)
    return {'files': files, 'python': platform.python_version(),
            'torch': importlib.metadata.version('torch'), 'numpy': importlib.metadata.version('numpy')}


def unit_gate(study, output):
    path = output / 'unit_checks.xml'
    if not path.exists():
        raise ValueError(f'Missing unit report: {path}')
    document = ET.parse(path).getroot()
    cases = list(document.iter('testcase'))
    if not cases or any(list(document.iter(name)) for name in ('failure', 'error', 'skipped')):
        raise ValueError(f'Unit checks are incomplete, failed or skipped: {path}')
    for suite in document.iter('testsuite'):
        if any(int(suite.get(name, '0')) for name in ('errors', 'failures', 'skipped')):
            raise ValueError(f'Unit report records failure/skip: {path}')
    required = ('test_tflow_core', 'test_tflow_runtime', 'test_tflow_tuning',
                'test_tflow_validation_policy', 'test_tflow_reporting') if study == 'tflow' else (
                    'test_rafm_input_parameterization', 'test_rafm_input_runtime', 'test_prepare_shared_study')
    classes = [case.get('classname', '') for case in cases]
    if not all(any(name in classname for classname in classes) for name in required):
        raise ValueError(f'Unit report omits required suites: {required}')
    records = [file_record(path)]
    if study == 'rafm_inputs':
        path = output / 'full_backbone_checks.json'
        if not path.exists():
            raise ValueError(f'Missing/pending/failed complete backbone interface checks: {path}')
        report = load(path)
        if (report.get('status') != 'passed'
                or report.get('script_sha256') != digest(ROOT / 'tools/check_rafm_input_backbones.py')
                or report.get('b_c_counts_modules_and_initial_weights_identical') is not True):
            raise ValueError(f'Full backbone checks have incomplete or changed provenance: {path}')
        expected = {(kind, arm) for kind in ('mlp', 'audio_unet', 'image_sit') for arm in ('A', 'B', 'C')}
        rows = report.get('rows', [])
        if len(rows) != 9 or {(row.get('model_config', {}).get('kind'), row.get('arm')) for row in rows} != expected:
            raise ValueError('Full backbone checks must contain exactly all nine architecture/arm rows')
        for row in rows:
            metadata = row.get('model_metadata', {})
            implementation = metadata.get('implementation_sha256', {})
            required_files = {'baselines/rafm_input_parameterization.py', 'rafm/models/mlp.py',
                              'rafm/flow_matching/sampler.py'}
            if (row.get('status') != 'passed' or row.get('strict_checkpoint_reload', {}).get('passed') is not True
                    or metadata.get('arm') != row['arm'] or metadata.get('model_config') != row['model_config']
                    or set(implementation) != required_files
                    or any(digest(ROOT / name) != value for name, value in implementation.items())):
                raise ValueError(f'Backbone metadata or current implementation mismatch: {row.get("arm")}')
            provenance = metadata.get('original_backbone_provenance', {})
            kind = row['model_config']['kind']
            source_path = {'mlp': ROOT / 'rafm/models/mlp.py',
                           'audio_unet': ROOT / 'experiments/poc_audio/audio_flow.py'}.get(kind)
            if kind == 'image_sit':
                source_path = Path(row['model_config']['source_dir']) / 'models.py'
                actual_commit = subprocess.check_output(['git', '-C', str(source_path.parent),
                                                         'rev-parse', 'HEAD'], text=True).strip()
                if provenance.get('commit') != actual_commit:
                    raise ValueError('Backbone SiT source commit changed after interface checks')
            if provenance.get('source_sha256') != digest(source_path):
                raise ValueError(f'Original {kind} backbone source changed after interface checks')
        records.append(file_record(path))
    return records


def sanity_gate(study, output, cfg, implementation_hash):
    directory = output / 'sanity' / cfg['condition_id']
    path = directory / 'sanity.json'
    if not path.exists():
        raise ValueError(f'Missing condition sanity: {path}')
    report = load(path)
    if report.get('status') != 'passed' or report.get('condition_id') != cfg['condition_id']:
        raise ValueError(f'Condition sanity did not pass: {path}')
    records = [file_record(path)]
    if study == 'tflow':
        if (report.get('config_sha256') != canonical_hash(cfg)
                or report.get('implementation_sha256') != implementation_hash
                or report.get('finite_samples') is not True or report.get('test_data_used') is not False
                or report.get('nfe') != cfg['evaluation']['model_evaluations']
                or report.get('batch_size') != cfg['training']['batch_size']
                or report.get('precision') != cfg['training']['precision']):
            raise ValueError(f'Sanity configuration, implementation or numerical contract mismatch: {path}')
    else:
        rows = report.get('rows', [])
        if [row.get('arm') for row in rows] != ['A', 'B', 'C']:
            raise ValueError(f'Sanity omits an A/B/C arm: {path}')
        for row in rows:
            arm = row['arm']
            config_path = directory / arm / 'config.json'
            signature = load(config_path)
            if (row.get('status') != 'passed' or row.get('finite_samples') is not True
                    or row.get('checkpoint_strict_load') is not True or row.get('test_data_used') is not False
                    or row.get('nfe') != cfg['evaluation']['model_evaluations']
                    or signature.get('config') != cfg
                    or signature.get('implementation_sha256') != implementation_hash):
                raise ValueError(f'A/B/C sanity provenance or numerical contract mismatch: {config_path}')
            records.append(file_record(config_path))
    return records


def frozen_files(commit, files):
    failures = []
    for name, expected in sorted(files.items()):
        path = Path(name)
        if path.is_absolute():
            # External references are pinned by their independent hash/commit in
            # the runtime; they are not misrepresented as part of this checkout.
            continue
        result = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            failures.append(f'Uncommitted or changed source/config relative to {commit}: {name}')
    return failures


def priority(path):
    cfg = load(path)
    # The long downstream condition starts first. Larger vector dimensions then
    # precede smaller vectors; tie-breaking is deterministic and does not use scores.
    return (0 if cfg['kind'] == 'audio' else 1 if cfg['kind'] == 'image' else 2,
            -cfg['data']['shape'][1], cfg['condition_id'])


def make_plan(args):
    study = args.study
    output = ROOT / STUDIES[study]['output']
    materialization_path = ROOT / 'configs/rafm_input_study/materialization.json'
    materialization = load(materialization_path)
    configs = sorted(CONFIG_DIR.glob('*.json'), key=priority)
    if sorted(path.stem for path in configs) != sorted(materialization['conditions']):
        raise ValueError('Prepared configuration files differ from the explicit materialized inventory')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    plan = {'schema_version': 1, 'study': study, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_commit': commit, 'repository': str(ROOT), 'python': str(PYTHON),
            'output_root': str(output), 'partition': 'hermes-2', 'node': STUDIES[study]['node'],
            'gpus_per_task': 1, 'max_concurrent_tasks': 6, 'cpus_per_task': 4, 'host_memory': '32G',
            'time_limit': args.time_limit, 'requested_suite_conditions': 28,
            'blocked_conditions': materialization['blocked_conditions'],
            'materialization': file_record(materialization_path),
            'launch_gates': [], 'blocking_issues': [], 'tasks': [], 'frozen_files': {},
            'sampling_audit': {'command': [str(PYTHON), 'tools/audit_study_samples.py', '--result', 'RESULT_JSON']},
            'submitted': False}
    try:
        plan['launch_gates'] += unit_gate(study, output)
    except (ValueError, OSError, ET.ParseError) as exc:
        plan['blocking_issues'].append(str(exc))
    for name in ORCHESTRATION_FILES:
        if not (ROOT / name).is_file():
            plan['blocking_issues'].append(f'Missing required orchestration file: {name}')
        else:
            plan['frozen_files'][name] = digest(ROOT / name)
    validation_files = (['experiments/tflow/sanity.py', 'tests/test_tflow_core.py', 'tests/test_tflow_runtime.py',
                         'tests/test_tflow_tuning.py', 'tests/test_tflow_validation_policy.py', 'tests/test_tflow_reporting.py']
                        if study == 'tflow' else ['tools/check_rafm_input_backbones.py',
                            'tests/test_rafm_input_parameterization.py', 'tests/test_rafm_input_runtime.py',
                            'tests/test_prepare_shared_study.py', 'tools/prepare_shared_study.py'])
    for name in validation_files:
        plan['frozen_files'][name] = digest(ROOT / name)
    final_tasks = []
    for path in configs:
        cfg = load(path)
        if cfg.get('protocol_status') != 'resolved' or cfg.get('blocking_issues'):
            raise ValueError(f'Prepared configuration is unresolved: {path}')
        implementation = runtime_manifest(study, cfg)
        implementation_hash = canonical_hash(implementation)
        plan['frozen_files'].update(implementation['files'])
        plan['frozen_files'][str(path.relative_to(ROOT))] = digest(path)
        gates = []
        try:
            gates = sanity_gate(study, output, cfg, implementation_hash)
        except (ValueError, OSError, KeyError) as exc:
            plan['blocking_issues'].append(str(exc))
        task = {'condition_id': cfg['condition_id'], 'config_path': str(path),
                'config_sha256': canonical_hash(cfg), 'config_file_sha256': digest(path),
                'config': cfg, 'implementation_sha256': implementation_hash,
                'implementation': implementation, 'sanity_gates': gates}
        if study == 'tflow':
            command = [str(PYTHON), '-m', 'experiments.tflow.tune', '--config', str(path),
                       '--output', str(output / 'tuning')]
            plan['tasks'].append({**task, 'index': len(plan['tasks']), 'phase': 'tuning',
                                  'commands': [command]})
            for seed in cfg['seeds']:
                command = [str(PYTHON), '-m', 'experiments.tflow.run', 'train-evaluate',
                           '--config', str(path), '--selection', str(output / 'tuning' / path.stem / 'selection.json'),
                           '--seed', str(seed), '--output', str(output / 'final')]
                final_tasks.append({**task, 'phase': 'final', 'seed': seed, 'commands': [command]})
        else:
            # Put A/B/C for each seed adjacent, so the early array slots compare
            # arms on the same condition before moving to smaller datasets.
            for seed in cfg['seeds']:
                for arm in ('A', 'B', 'C'):
                    command = [str(PYTHON), '-m', 'experiments.rafm_inputs.run', 'train-evaluate',
                               '--config', str(path), '--arm', arm, '--seed', str(seed), '--output', str(output)]
                    plan['tasks'].append({**task, 'index': len(plan['tasks']), 'seed': seed, 'arm': arm, 'phase': 'final',
                                          'commands': [command]})
    for task in final_tasks:
        plan['tasks'].append({**task, 'index': len(plan['tasks'])})
    plan['array_phases'] = ([{'name': 'tuning', 'first': 0, 'last': len(configs)-1},
                             {'name': 'final', 'first': len(configs), 'last': len(plan['tasks'])-1,
                              'dependency': 'afterany:tuning_array'}] if study == 'tflow' else
                            [{'name': 'final', 'first': 0, 'last': len(plan['tasks'])-1}])
    for phase in plan['array_phases']:
        indices = list(range(phase['first'], phase['last'] + 1))
        phase['worker_count'] = min(6, len(indices))
        phase['shards'] = [{'worker_index': index, 'task_indices': indices[index::phase['worker_count']]}
                           for index in range(phase['worker_count'])]
    plan['blocking_issues'] += frozen_files(commit, plan['frozen_files'])
    plan['condition_count'] = len(configs)
    plan['task_count'] = len(plan['tasks'])
    plan['ready_to_submit'] = not plan['blocking_issues']
    return plan


def check_file(record):
    if digest(record['path']) != record['sha256']:
        raise ValueError(f'Changed reviewed evidence: {record["path"]}')


def verify_plan(plan, *, task=None):
    if not plan.get('ready_to_submit') or plan.get('blocking_issues'):
        raise ValueError('The immutable task manifest contains unresolved launch gates')
    for name, expected in plan['frozen_files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen source/config changed: {name}')
    failures = frozen_files(plan['source_commit'], plan['frozen_files'])
    if failures:
        raise ValueError('\n'.join(failures))
    check_file(plan['materialization'])
    for gate in plan['launch_gates']:
        check_file(gate)
    tasks = [task] if task is not None else plan['tasks']
    for item in tasks:
        if digest(item['config_path']) != item['config_file_sha256'] or canonical_hash(load(item['config_path'])) != item['config_sha256']:
            raise ValueError(f'Changed task configuration: {item["config_path"]}')
        for gate in item['sanity_gates']:
            check_file(gate)


def submit(plan, manifest_path, manifest_hash):
    verify_plan(plan)
    output = Path(plan['output_root'])
    for prior_path in (output / 'launch').glob('*.submission.json'):
        prior = load(prior_path)
        if any(row.get('job_id') or (row.get('exit_code') == 0) for row in prior.get('arrays', [])):
            raise RuntimeError(f'This study already has a submission record: {prior_path}. '
                               'Inspect scheduler state and resume only selected existing tasks; '
                               'do not submit a duplicate full sweep.')
    # Exclusive creation closes the race between two concurrent planners using
    # different manifests. A partial submission retains its claim for review.
    claim_path = output / 'launch' / 'submission_claim.json'
    write(claim_path, {'manifest': str(manifest_path), 'manifest_sha256': manifest_hash,
                      'created_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()}, immutable=True)
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    record = {'manifest': str(manifest_path), 'manifest_sha256': manifest_hash,
              'submitted_utc': datetime.now(timezone.utc).isoformat(), 'arrays': []}
    record_path = manifest_path.with_suffix('.submission.json')
    write(record_path, record, immutable=True)
    previous_job = None
    for phase in plan['array_phases']:
        command = ['sbatch', '--parsable', '--partition=hermes-2', f'--nodelist={plan["node"]}',
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--gres=gpu:mi210:1', '--mem=32G',
                   f'--time={plan["time_limit"]}', f'--array=0-{phase["worker_count"] - 1}%6',
                   '--no-requeue', f'--job-name={plan["study"]}-{phase["name"]}-v1',
                   f'--output={output}/logs/%x-%A_%a.out', f'--error={output}/logs/%x-%A_%a.err']
        if previous_job is not None:
            # A failed tuning condition must not prevent successful conditions'
            # final jobs. Each final task independently requires its own frozen
            # selection. The phase barrier also prevents >6 jobs on this node.
            command.append(f'--dependency=afterany:{previous_job}')
        command += [str(ROOT / 'tools/shared_study_array_job.sh'), str(manifest_path), manifest_hash, phase['name']]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        row = {'phase': phase['name'], 'command': command, 'exit_code': completed.returncode,
               'stdout': completed.stdout.strip(), 'stderr': completed.stderr.strip()}
        if completed.returncode == 0:
            job_id = completed.stdout.strip().split(';')[0]
            if job_id.isdigit():
                row['job_id'] = job_id
                previous_job = job_id
            else:
                row['failure'] = 'Unrecognized sbatch job ID; inspect scheduler before retrying'
        record['arrays'].append(row)
        write(record_path, record)
        print(json.dumps(row, indent=2))
        if completed.returncode or 'job_id' not in row:
            # A successfully submitted preceding phase remains explicitly
            # recorded. Never resubmit it automatically after a partial failure.
            raise SystemExit('Partial/failed submission recorded; inspect before retrying')


def task_output(plan, task, seed):
    root = Path(plan['output_root']) / 'final' / task['condition_id']
    if plan['study'] == 'rafm_inputs':
        root /= task['arm']
    return root / f'seed_{seed}' / 'result.json'


def selection_receipt(plan, task, *, create=False, execution_path=None):
    directory = Path(plan['output_root']) / 'tuning' / task['condition_id']
    selection_path = directory / 'selection.json'
    choice = load(selection_path)
    if (choice.get('status') != 'frozen' or choice.get('selection_split') != 'validation'
            or choice.get('test_data_used_for_selection') is not False
            or choice.get('config_sha256') != task['config_sha256']
            or choice.get('implementation_sha256') != task['implementation_sha256']):
        raise ValueError('Source selection is incomplete or has changed provenance')
    expected = {'schema_version': 1, 'status': 'frozen', 'condition_id': task['condition_id'],
                'selection': file_record(selection_path), 'selected': choice['selected'],
                'config_sha256': task['config_sha256'], 'implementation_sha256': task['implementation_sha256'],
                'source_commit': plan['source_commit']}
    path = directory / 'selection_receipt.json'
    if path.exists():
        prior = load(path)
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError('Frozen selection receipt mismatch; source choice changed between tasks')
    elif create:
        write(path, {**expected, 'created_utc': datetime.now(timezone.utc).isoformat(),
                     'tuning_execution': str(execution_path)}, immutable=True)
    else:
        raise FileNotFoundError(f'Final training requires the tuning selection receipt: {path}')
    return file_record(path)


def execute_task(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Array execution requires a cluster scheduler allocation')
    manifest_path = Path(args.manifest).resolve()
    if not args.manifest_sha256 or digest(manifest_path) != args.manifest_sha256:
        raise ValueError('Task manifest SHA-256 mismatch')
    plan = load(manifest_path)
    if not 0 <= args.execute_task < len(plan['tasks']):
        raise ValueError('Task index is outside the immutable manifest')
    task = plan['tasks'][args.execute_task]
    worker_key = f'{os.environ["SLURM_JOB_ID"]}_{os.environ.get("SLURM_ARRAY_TASK_ID", "single")}'
    job_key = f'{worker_key}_task_{args.execute_task}'
    record_path = Path(plan['output_root']) / 'executions' / f'{job_key}.json'
    record = {'status': 'starting', 'study': plan['study'], 'task_index': task['index'],
              'condition_id': task['condition_id'], 'phase': task['phase'], 'arm': task.get('arm'), 'seed': task.get('seed'),
              'source_commit': plan['source_commit'], 'manifest': str(manifest_path),
              'manifest_sha256': args.manifest_sha256, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'started_utc': datetime.now(timezone.utc).isoformat(), 'host': platform.node(), 'commands': []}
    write(record_path, record, immutable=True)
    lock_stream = None
    try:
        if args.execute_task < 0 or task['index'] != args.execute_task:
            raise ValueError('Task index does not match immutable manifest')
        if platform.node().split('.')[0] != plan['node']:
            raise ValueError(f'Task is on {platform.node()}, expected separate study node {plan["node"]}')
        verify_plan(plan, task=task)
        lock_key = '_'.join(str(value) for value in (task['condition_id'], task['phase'],
                                                    task.get('arm', 'all'), task.get('seed', 'selection')))
        lock_path = Path(plan['output_root']) / 'locks' / f'{lock_key}.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_stream = lock_path.open('a+')
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another worker owns this exact condition/arm/seed: {lock_path}') from exc
        record['run_lock'] = str(lock_path)
        # Logical tasks on this worker run sequentially and share writable
        # compiler/kernel caches. GPU model memory is released with each child.
        cache = Path(plan['output_root']) / 'cache' / worker_key
        env = dict(os.environ)
        for key, suffix in {'MIOPEN_USER_DB_PATH': 'performance_db', 'MIOPEN_CUSTOM_CACHE_DIR': 'kernel_cache',
                            'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
                            'MPLCONFIGDIR': 'matplotlib', 'TORCH_HOME': 'torch'}.items():
            path = cache / suffix
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=f'{ROOT}/tools/validation_deps:{ROOT}')
        # Runtime import/check is allowed here, on compute. It catches any drift
        # between this stdlib planner's file inventory and the actual runner.
        os.environ.update(env)
        from experiments.tflow import run as runtime
        runtime.compute_device()
        cfg = task['config']
        if plan['study'] == 'tflow':
            actual = runtime.implementation_sha256(cfg)
        else:
            from experiments.rafm_inputs.run import implementation
            actual = runtime.json_hash(implementation(cfg))
        if actual != task['implementation_sha256']:
            raise ValueError('Actual runtime implementation differs from planned manifest')
        record['status'] = 'running'
        record['hardware'] = runtime.hardware()
        write(record_path, record)

        def run_command(command, phase, seed=None):
            row = {'phase': phase, 'seed': seed, 'command': command,
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
            record['commands'].append(row)
            write(record_path, record)
            completed = subprocess.run(command, cwd=ROOT, env=env)
            row.update(exit_code=completed.returncode,
                       status='complete' if completed.returncode == 0 else 'failed',
                       ended_utc=datetime.now(timezone.utc).isoformat())
            write(record_path, record)
            return completed.returncode

        failed = False
        if task['phase'] == 'tuning':
            failed = bool(run_command(task['commands'][0], 'validation_only_tuning'))
            if not failed:
                record['selection_receipt'] = selection_receipt(plan, task, create=True, execution_path=record_path)
        else:
            seed = task['seed']
            if plan['study'] == 'tflow':
                selection = Path(plan['output_root']) / 'tuning' / task['condition_id'] / 'selection.json'
                # The phase barrier uses afterany so a failed condition does not
                # cancel others. This condition's source must still be frozen.
                record['selection_receipt'] = selection_receipt(plan, task)
                runtime.source_from_selection(cfg, selection)
            if run_command(task['commands'][0], 'final_train_evaluate', seed):
                failed = True
            else:
                result_path = task_output(plan, task, seed)
                audit_command = [str(PYTHON), 'tools/audit_study_samples.py', '--result', str(result_path)]
                failed = bool(run_command(audit_command, 'independent_sample_audit', seed))
            if plan['study'] == 'tflow' and selection_receipt(plan, task) != record['selection_receipt']:
                raise ValueError('Frozen selection receipt changed while this final seed ran')
        record['status'] = 'failed' if failed else 'complete'
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(record_path, record)
        return 1 if failed else 0
    except Exception as exc:
        record.update(status='failed', ended_utc=datetime.now(timezone.utc).isoformat(),
                      failure={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        write(record_path, record)
        raise
    finally:
        if lock_stream is not None:
            lock_stream.close()


def execute_worker(args):
    """One persistent GPU allocation, sequential isolated logical-task children."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Persistent workers require a Slurm compute allocation')
    path = Path(args.manifest).resolve()
    if not args.manifest_sha256 or digest(path) != args.manifest_sha256:
        raise ValueError('Worker manifest SHA-256 mismatch')
    plan = load(path)
    phases = [phase for phase in plan['array_phases'] if phase['name'] == args.phase]
    if len(phases) != 1 or not 0 <= args.execute_worker < phases[0]['worker_count']:
        raise ValueError('Unknown phase or worker index')
    phase = phases[0]
    shard = phase['shards'][args.execute_worker]
    if shard['worker_index'] != args.execute_worker:
        raise ValueError('Worker shard identity mismatch')
    worker_key = f'{os.environ["SLURM_JOB_ID"]}_{os.environ.get("SLURM_ARRAY_TASK_ID", args.execute_worker)}'
    record_path = Path(plan['output_root']) / 'executions' / f'{worker_key}_worker.json'
    record = {'status': 'running', 'study': plan['study'], 'phase': args.phase,
              'worker_index': args.execute_worker, 'task_indices': shard['task_indices'],
              'manifest': str(path), 'manifest_sha256': args.manifest_sha256,
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'host': platform.node(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'tasks': []}
    write(record_path, record, immutable=True)
    failed = False
    for index in shard['task_indices']:
        if plan['tasks'][index]['phase'] != args.phase:
            raise ValueError('A logical task belongs to a different worker phase')
        command = [str(PYTHON), str(ROOT / 'tools/launch_shared_study.py'),
                   '--execute-task', str(index), '--manifest', str(path),
                   '--manifest-sha256', args.manifest_sha256]
        row = {'task_index': index, 'command': command, 'status': 'running',
               'started_utc': datetime.now(timezone.utc).isoformat()}
        record['tasks'].append(row)
        write(record_path, record)
        completed = subprocess.run(command, cwd=ROOT)
        row.update(exit_code=completed.returncode, status='complete' if completed.returncode == 0 else 'failed',
                   ended_utc=datetime.now(timezone.utc).isoformat())
        failed |= completed.returncode != 0
        write(record_path, record)
        # An unrelated failed condition/seed never prevents later shard tasks.
    record.update(status='failed' if failed else 'complete', ended_utc=datetime.now(timezone.utc).isoformat())
    write(record_path, record)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES)
    parser.add_argument('--manifest', help='New immutable manifest path; default uses a unique filename')
    parser.add_argument('--submit', action='store_true', help='Submit only after all concrete gates pass')
    parser.add_argument('--time-limit', default='48:00:00')
    parser.add_argument('--execute-task', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--execute-worker', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('tuning', 'final'), help=argparse.SUPPRESS)
    parser.add_argument('--manifest-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_worker is not None:
        if args.submit or not args.manifest or not args.phase or args.execute_task is not None:
            parser.error('Worker execution requires manifest/phase and cannot submit or select another task')
        raise SystemExit(execute_worker(args))
    if args.execute_task is not None:
        if args.submit or not args.manifest:
            parser.error('Task execution requires --manifest and cannot submit')
        raise SystemExit(execute_task(args))
    if not args.study:
        parser.error('Planning requires --study')
    plan = make_plan(args)
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    path = Path(args.manifest).resolve() if args.manifest else Path(plan['output_root']) / 'launch' / f'tasks_{identifier}.json'
    write(path, plan, immutable=True)
    manifest_hash = digest(path)
    print(json.dumps({'manifest': str(path), 'sha256': manifest_hash, 'study': args.study,
                      'tasks': plan['task_count'], 'conditions': plan['condition_count'],
                      'node': plan['node'], 'concurrency': 6, 'ready_to_submit': plan['ready_to_submit'],
                      'blocking_issues': plan['blocking_issues'], 'blocked_conditions': plan['blocked_conditions']}, indent=2))
    if args.submit:
        if not plan['ready_to_submit']:
            raise SystemExit('No jobs submitted: launch gates are recorded in the manifest')
        submit(plan, path, manifest_hash)


if __name__ == '__main__':
    main()
