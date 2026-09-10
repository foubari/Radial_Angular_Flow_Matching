"""Bounded CPU-only monitoring and reporting of recovered t-Flow outcomes.

Reads immutable launch/config/source records, JSON training logs, results,
sample audits and scheduler state. Never launches, retries or evaluates models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import time

import monitor_recovered_study as common
import report_shared_study as shared

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs_tflow_full/recovered/monitor'
REPORT = ROOT / 'outputs_rafm_input_study/complete_report_tflow'
PLOT_PYTHON = ROOT.parent / 'msgm-sparse-control/.venv/bin/python'
CONDITIONS = {'imagenette_dcae', 'piv_d32'}
INTERVAL = 60


def scheduler(job, workers=None):
    queue = common.command_result(['squeue', '--noheader', '--array', '--jobs=' + job,
                                   '--format=%i|%T|%N|%M'])
    accounting = common.command_result(['sacct', '--noheader', '--parsable2', '--jobs=' + job,
                                        '--format=JobID%32,State%32,ExitCode,Elapsed,NodeList'])
    pattern = re.escape(job) + (r'_\d+' if workers else '')
    rows = {}
    for line in accounting['stdout'].splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) >= 3 and re.fullmatch(pattern, parts[0]):
            rows[parts[0]] = {'state': parts[1].split()[0].rstrip('+'), 'exit_code': parts[2]}
    accounting_terminal = (accounting['exit_code'] == 0 and len(rows) == (workers or 1)
                           and all(row['state'] in common.TERMINAL for row in rows.values()))
    # Completed jobs age out of the controller before they age out of sacct.
    # Accept only this exact squeue error, and only with positive, complete
    # accounting evidence. Authentication/network errors remain real failures.
    queue_purged = (queue['exit_code'] == 1 and not queue['stdout'] and accounting_terminal
                    and queue['stderr'] == 'slurm_load_jobs error: Invalid job id specified')
    query_failure = (queue['exit_code'] != 0 and not queue_purged) or accounting['exit_code'] != 0
    ended = not query_failure and not queue['stdout'] and accounting_terminal
    return {'job_id': job, 'queue': queue, 'accounting': accounting, 'tasks': rows,
            'ended': ended, 'query_failure': query_failure,
            'queue_purged_after_accounted_completion': queue_purged}


def load_plan(args):
    if shared.sha256(args.manifest) != args.manifest_sha256:
        raise ValueError('Recovered t-Flow manifest checksum mismatch')
    plan = shared.read_json(args.manifest)
    tasks = plan['tasks']
    tuning = [t for t in tasks if t['phase'] == 'tuning']
    finals = [t for t in tasks if t['phase'] == 'final']
    if (plan.get('study') != 'tflow' or len(tasks) != 8 or len(tuning) != 2 or len(finals) != 6
            or {t['condition_id'] for t in tuning} != CONDITIONS
            or {t['condition_id'] for t in finals} != CONDITIONS
            or len({(t['condition_id'], t['seed']) for t in finals}) != 6):
        raise ValueError('Expected exactly two recovered conditions, two tuning tasks and six unique finals')
    phases = {p['name']: p['worker_count'] for p in plan['array_phases']}
    if phases != {'tuning': 2, 'final': 6}:
        raise ValueError('Unexpected recovered array worker counts')
    for task in tasks:
        if (shared.config_hash(task['config']) != task['config_sha256']
                or shared.sha256(task['config_path']) != task['config_file_sha256']
                or shared.read_json(task['config_path']) != task['config']):
            raise ValueError('Manifest/config identity mismatch: ' + task['condition_id'])
    for name, expected in plan['frozen_files'].items():
        if shared.sha256(ROOT / name) != expected:
            raise ValueError('Frozen source/config checksum mismatch: ' + name)
    receipt = shared.read_json(args.manifest.with_suffix('.submission.json'))
    recorded = {row['phase']: str(row.get('job_id')) for row in receipt['arrays'] if row['exit_code'] == 0}
    if (receipt.get('manifest_sha256') != args.manifest_sha256
            or Path(receipt['manifest']).resolve() != args.manifest.resolve()
            or recorded != {'tuning': args.tuning_job, 'final': args.final_job}):
        raise ValueError('Submission receipt does not match manifest and requested array IDs')
    return plan


def training_log(path):
    out = {'path': str(path), 'logged_rows': 0, 'training_eta': None,
           'eta_limitation': 't-Flow training logs lack elapsed times; no throughput or runtime is inferred'}
    if not path.exists():
        return out
    with path.open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                break
            row = shared.json_safe(json.loads(raw))
            if not shared.numeric(row.get('step')):
                raise ValueError('Missing/nonfinite logged training step: ' + str(path))
            out.setdefault('first_logged', row)
            out.update(latest_logged=row, logged_rows=out['logged_rows'] + 1)
    out['file_modified_utc'] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return out


def tuning_row(task, base):
    directory = base / 'tuning' / task['condition_id']
    selection_path = directory / 'selection.json'
    row = {'condition': task['condition_id'], 'status': 'pending', 'selection_path': str(selection_path),
           'candidate_logs': [training_log(p) for p in sorted(directory.glob('candidate_*/training.jsonl'))],
           'failures': []}
    for path in sorted(directory.glob('candidate_*/failed.json')):
        failed = shared.read_json(path)
        if (failed.get('config_sha256') != task['config_sha256']
                or failed.get('implementation_sha256') != task['implementation_sha256']
                or failed.get('stage') != 'tuning' or failed.get('status') != 'failed'):
            raise ValueError('Tuning failure identity mismatch: ' + str(path))
        row['failures'].append({'path': str(path), 'sha256': shared.sha256(path), 'record': shared.json_safe(failed)})
    if row['failures']:
        row['status'] = 'failed'
    if selection_path.exists():
        selection = shared.read_json(selection_path)
        required = {'status': 'frozen', 'condition_id': task['condition_id'],
                    'config_sha256': task['config_sha256'], 'implementation_sha256': task['implementation_sha256'],
                    'selection_split': 'validation', 'test_data_used_for_selection': False,
                    'tuning_seed': task['config']['tuning']['seed'], 'full_training_equivalents': 0.55}
        if (any(selection.get(k) != v for k, v in required.items())
                or len(selection.get('stage1_trials', [])) != 9 or len(selection.get('finalist_trials', [])) != 2
                or row['failures']):
            raise ValueError('Source-selection provenance mismatch: ' + str(selection_path))
        row.update(status='frozen', selection_sha256=shared.sha256(selection_path), selected=selection['selected'])
    elif row['candidate_logs'] and not row['failures']:
        row['status'] = 'updates_logged_or_validation_in_progress'
    return row


def final_row(task, base, selection):
    cfg = task['config']
    directory = base / 'final' / task['condition_id'] / f"seed_{task['seed']}"
    cache, issue = shared.load_cache_manifest(cfg)
    if issue:
        raise ValueError(task['condition_id'] + ': ' + issue)
    row = shared.collect_seed(cfg, 'tflow', task['seed'], directory, cache)
    if row.get('result_sha256'):
        result = shared.read_json(directory / 'result.json')
        expected = {'condition_id': task['condition_id'], 'method': 'tflow', 'stage': 'final',
                    'seed': task['seed'], 'config': cfg, 'config_sha256': task['config_sha256'],
                    'implementation_sha256': task['implementation_sha256']}
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError('Final result identity/source mismatch: ' + str(directory))
        source = selection.get('selected', {})
        if selection['status'] != 'frozen' or result.get('source') != {k: source[k] for k in ('nu', 'scale')}:
            raise ValueError('Final source differs from frozen validation selection: ' + str(directory))
        # Exception envelopes may legitimately lack dataset/checkpoint records.
        # When present, even failed metric outcomes must retain matching splits.
        dataset = result.get('dataset_manifest')
        if dataset is not None:
            for split in ('train', 'val', 'test'):
                if (dataset.get('splits', {}).get(split, {}).get('sha256') != cache['splits'][split]['values']['sha256_contiguous_bytes']
                        or dataset.get('split_indices', {}).get(split, {}).get('sha256') != cache['splits'][split]['indices']['sha256_contiguous_bytes']):
                    raise ValueError('Failed/complete result dataset identity mismatch: ' + str(directory))
    row.update(condition=task['condition_id'], training=training_log(directory / 'training.jsonl'))
    if row['status'] in ('complete', 'failed', 'awaiting_sample_audit', 'incompatible'):
        row['execution_stage'] = row['status']
    elif row['training']['logged_rows']:
        row['execution_stage'] = ('sampling_or_evaluation_after_final_logged_update'
            if row['training']['latest_logged']['step'] >= cfg['training']['steps'] else 'training_updates_logged')
    else:
        row['execution_stage'] = 'no_training_log_yet; consult scheduler allocation/prolog state'
    return row


def snapshot(plan, args):
    base = Path(plan['output_root'])
    tuning = [tuning_row(t, base) for t in plan['tasks'] if t['phase'] == 'tuning']
    by_condition = {row['condition']: row for row in tuning}
    rows = [final_row(t, base, by_condition[t['condition_id']]) for t in plan['tasks'] if t['phase'] == 'final']
    return {'time': common.timestamps(), 'status': 'monitoring', 'manifest': str(args.manifest),
            'manifest_sha256': args.manifest_sha256, 'source_commit': plan['source_commit'],
            'expected_final_tasks': 6, 'counts': dict(Counter(row['status'] for row in rows)),
            'tasks': rows, 'tuning': tuning,
            'additional_final_arrays': args.additional_final_array,
            'scheduler': {'tuning': scheduler(args.tuning_job, 2), 'final': scheduler(args.final_job, 6),
                          **{'additional_final_' + job: scheduler(job, workers)
                             for job, workers in args.additional_final_array}},
            'policy': 'Read-only; no retries, submissions, training, tuning, sampling or metric recomputation'}


def combined_completion():
    report = shared.collect(ROOT / 'configs/rafm_input_study_complete',
        ROOT / 'outputs_rafm_input_study/v1/final', ROOT / 'outputs_tflow_full/v2/final',
        ROOT / 'outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json',
        ROOT / 'outputs_audio_gain/delivery_manifest.json')
    if report['expected_final_runs'] != 336 or len(report['conditions']) != 28:
        raise ValueError('Expected the complete 28-condition, 336-outcome report scope')
    rows = [row for condition in report['conditions'] for group in condition['methods'].values() for row in group['seeds']]
    return {'counts': report['seed_status_counts'], 'all_terminal': all(r['status'] in ('complete', 'failed') for r in rows),
            'unresolved': [{'path': r['result_path'], 'status': r['status'], 'issues': r['issues']}
                           for r in rows if r['status'] not in ('complete', 'failed')]}


def report_once(args):
    receipt = args.output / 'reporter.json'
    if receipt.exists():
        prior = shared.read_json(receipt)
        if prior.get('status') != 'finished':
            raise RuntimeError('Prior report attempt unresolved; inspect ' + str(receipt))
        return prior
    if args.report_output.resolve() != REPORT.resolve():
        raise ValueError('Combined report must use the separate complete_report_tflow directory')
    record = {'status': 'started', 'time': common.timestamps(), 'commands': []}
    with receipt.open('x') as stream:
        json.dump(record, stream)
    config_root = ROOT / 'configs/rafm_input_study_complete'
    commands = [
        [str(PLOT_PYTHON), '-B', str(ROOT / 'tools/report_shared_study.py'), '--config-root', str(config_root), '--output', str(args.report_output), '--plots'],
        ['/usr/bin/python3', '-B', str(ROOT / 'tools/report_rafm_completion.py'), '--report', str(args.report_output / 'report.json'), '--output', str(args.report_output), '--require-finished'],
        ['/usr/bin/python3', '-B', str(ROOT / 'tools/summarize_shared_study.py'), '--report', str(args.report_output / 'report.json')]]
    for index, command in enumerate(commands):
        done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
        record['commands'].append({'command': command, 'exit_code': done.returncode,
                                   'stdout': done.stdout, 'stderr': done.stderr})
        shared.write_json(receipt, record)
        if done.returncode:
            break
        if index == 0:
            report = shared.read_json(args.report_output / 'report.json')
            counts = report['seed_status_counts']
            if (report['expected_final_runs'] != 336 or len(report['conditions']) != 28
                    or sum(counts.get(k, 0) for k in ('complete', 'failed')) != 336):
                raise ValueError('Combined report still has unresolved outcomes; exact counts: ' + str(counts))
            # Keep the frozen collector and its generated legacy prose unchanged.
            # This separate note explicitly supersedes only its obsolete blocker sentence.
            (args.report_output / 'recovered_provenance_note.md').write_text(
                '# Recovered-condition reporting note\n\n'
                'The generated manuscript_addition.md retains a legacy sentence saying that PIV d32 and image provenance are blockers. '
                'That sentence is obsolete for this complete 28-condition report: both recovered conditions use the verified release, pinned caches and recorded protocols. '
                'Use findings.md and report.json for measured outcomes and remaining scientific failures. The original main manuscript has not been modified.\n\n'
                'The historical 3,925-image FID reference is retained by authorization; it overlaps generator train/validation/test by 2,339/775/811 images and is not a held-out reference. '
                'New shared synthetic caches remain labelled new matched comparisons, not recovered historical realizations. '
                'See docs/rafm_toy_numerical_failure_analysis.md and docs/tflow_matched_backbone_failure_analysis.md for numerical and architecture qualifications. '
                'Means and population standard deviations require all three compatible, finite final seeds; failed or missing seeds are never averaged away.\n')
    record.update(status='finished', finished=common.timestamps(), exit_code=record['commands'][-1]['exit_code'])
    shared.write_json(receipt, record)
    return record


def monitor(plan, args):
    began = time.monotonic()
    ended_since = None
    query_failures = 0
    signatures = {}
    common.event(args.output, 'monitor_started', manifest_sha256=args.manifest_sha256,
                 tuning_job=args.tuning_job, final_job=args.final_job,
                 additional_final_arrays=args.additional_final_array, max_hours=args.hours, once=args.once)
    while True:
        tick = time.monotonic()
        data = snapshot(plan, args)
        data.update(monitor_elapsed_s=tick - began, max_monitor_hours=args.hours,
                    monitor_mode='single_snapshot' if args.once else 'persistent')
        for row in data['tasks'] + data['tuning']:
            key = row['condition'] + '/' + str(row.get('seed', 'tuning'))
            signature = (row['status'], row.get('result_sha256'), row.get('selection_sha256'))
            if signatures.get(key) != signature:
                common.event(args.output, 'task_status_changed', task=key, status=row['status'],
                             failure=row.get('failure'), failures=row.get('failures'), issues=row.get('issues'),
                             nonfinite_metric_fields=row.get('nonfinite_metric_fields'))
                signatures[key] = signature
        terminal = all(row['status'] in ('complete', 'failed') for row in data['tasks'])
        schedules = data['scheduler']
        query_failures = query_failures + 1 if any(s['query_failure'] for s in schedules.values()) else 0
        data['consecutive_scheduler_query_failures'] = query_failures
        code = None
        if terminal and not args.once:
            data['combined_completion'] = combined_completion()
        if terminal and not args.once and data['combined_completion']['all_terminal']:
            load_plan(args)  # Recheck immutable source/config bytes immediately before reporting.
            data['reporter'] = report_once(args)
            code = (1 if data['combined_completion']['counts'].get('failed') else 0) if data['reporter']['exit_code'] == 0 else 2
            data['status'] = 'complete' if code == 0 else 'finished_with_failures' if code == 1 else 'blocked'
            if code == 2:
                data['blocker'] = 'Combined reporting command failed; inspect reporter.json'
        elif terminal:
            data['status'] = 'waiting_for_remaining_combined_outcomes'
        finals_ended = all(value['ended'] for name, value in schedules.items()
                           if name == 'final' or name.startswith('additional_final_'))
        if finals_ended and not terminal:
            ended_since = tick if ended_since is None else ended_since
        else:
            ended_since = None
        if code is None and (query_failures >= 3 or (ended_since is not None and tick - ended_since >= 60)
                             or tick - began >= args.hours * 3600):
            code = 2
            if query_failures >= 3:
                data.update(status='blocked', blocker='Three consecutive scheduler query failures; exact errors recorded')
            elif ended_since is not None and tick - ended_since >= 60:
                data.update(status='blocked', blocker='Original and additional final arrays ended, but results/audits remain unresolved after 60-second grace',
                    unresolved=[{'condition': r['condition'], 'seed': r['seed'], 'status': r['status'],
                                 'path': r['result_path'], 'issues': r['issues']}
                                for r in data['tasks'] if r['status'] not in ('complete', 'failed')])
            else:
                data.update(status='timeout', blocker='Bounded monitoring deadline reached')
        if args.once:
            data['status'] = 'single_snapshot'
        shared.write_json(args.output / 'latest.json', data)
        if args.once or code is not None:
            common.event(args.output, 'monitor_stopped', status=data['status'], counts=data['counts'], blocker=data.get('blocker'))
            print(json.dumps({'status': data['status'], 'time': data['time'], 'counts': data['counts'],
                              'latest': str(args.output / 'latest.json'), 'blocker': data.get('blocker')}), flush=True)
            return 0 if args.once else code
        time.sleep(max(0, min(INTERVAL - (time.monotonic() - tick), began + args.hours * 3600 - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--tuning-job', required=True)
    parser.add_argument('--final-job', required=True)
    parser.add_argument('--additional-final-array', action='append', default=[], metavar='JOB:WORKER_COUNT',
                        help='Root-authorized replacement array to monitor; does not submit or retry any task')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--report-output', type=Path, default=REPORT)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--once', action='store_true', help='Single snapshot, no report invocation')
    args = parser.parse_args()
    if (not math.isfinite(args.hours) or not 0 < args.hours <= 8
            or not re.fullmatch('[a-f0-9]{64}', args.manifest_sha256)
            or not all(re.fullmatch(r'\d+', j) for j in (args.tuning_job, args.final_job))):
        parser.error('Require a SHA-256, numeric job IDs and a positive maximum of eight hours')
    additional = []
    for spec in args.additional_final_array:
        if not re.fullmatch(r'[1-9]\d*:[1-9]\d*', spec):
            parser.error('--additional-final-array requires positive JOB:WORKER_COUNT integers')
        job, count = spec.split(':')
        if int(count) > 6 or job in {args.tuning_job, args.final_job, *(j for j, _ in additional)}:
            parser.error('Additional final arrays require unique job IDs and at most six workers')
        additional.append((job, int(count)))
    args.additional_final_array = additional
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.monitor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            code = monitor(load_plan(args), args)
        except Exception as error:
            record = {'status': 'blocked', 'time': common.timestamps(), 'blocker': str(error),
                      'error_type': type(error).__name__, 'manifest': str(args.manifest)}
            shared.write_json(args.output / 'latest.json', record)
            common.event(args.output, 'monitor_execution_failure', **record)
            print(json.dumps(record), flush=True)
            code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
