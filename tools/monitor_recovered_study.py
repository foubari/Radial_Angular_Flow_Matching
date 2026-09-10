"""CPU-only, read-only experiment monitor for the 18 recovered A/B/C runs.

Writes monitoring/report artifacts only. Never submits, resumes, cancels, trains
or evaluates experiments. Timestamps include UTC and Europe/Paris.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

import report_shared_study as shared

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs_rafm_input_study/recovered/launch/tasks_20260910T141752_da3e0210.json'
MANIFEST_SHA = '76cbb3534bc5182bc65e79580afbe344980ea0c52f2bddee8763905dc1200546'
OUTPUT = ROOT / 'outputs_rafm_input_study/recovered/monitor'
JOB = '765988'
INTERVAL = 60
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def timestamps():
    now = datetime.now(timezone.utc)
    return {'utc': now.isoformat(), 'paris': now.astimezone(ZoneInfo('Europe/Paris')).isoformat()}


def event(output, kind, **details):
    # A single O_APPEND write under the process-wide lock preserves whole JSONL
    # records; latest.json uses an atomic replacement through shared.write_json.
    data = (json.dumps({'time': timestamps(), 'event': kind, **details}, allow_nan=False) + '\n').encode()
    fd = os.open(output / 'events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError('Incomplete monitoring event write')
        os.fsync(fd)
    finally:
        os.close(fd)


def read_training(path, budget):
    if not path.exists():
        return {'logged_rows': 0, 'training_eta': None}
    records, segment = [], []
    with path.open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                break  # A concurrent writer has not completed this line yet.
            row = json.loads(raw)
            if not all(shared.numeric(row.get(key)) for key in ('step', 'elapsed_s')):
                raise ValueError('Training log lacks finite step/elapsed_s: ' + str(path))
            if segment and (row['step'] <= segment[-1]['step'] or row['elapsed_s'] <= segment[-1]['elapsed_s']):
                segment = []
            segment.append(row)
            records.append(row)
    out = {'path': str(path), 'logged_rows': len(records), 'training_eta': None,
           'file_modified_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    if not records:
        return out
    out.update(first_logged=records[0], latest_logged=records[-1], latest_contiguous_segment_rows=len(segment))
    if len(segment) >= 2:
        first, last = segment[0], segment[-1]
        updates, seconds = last['step'] - first['step'], last['elapsed_s'] - first['elapsed_s']
        if updates > 0 and seconds > 0:
            remaining = max(0, budget - last['step']) * seconds / updates
            finish = datetime.now(timezone.utc) + timedelta(seconds=remaining)
            out['training_eta'] = {'kind': 'estimate_from_measured_log_deltas',
                'first_step_used': first['step'], 'latest_step_used': last['step'],
                'measured_updates': updates, 'measured_elapsed_s': seconds,
                'measured_seconds_per_update': seconds / updates,
                'estimated_remaining_training_s_from_latest_logged_step': remaining,
                'estimated_training_finish_utc': finish.isoformat(),
                'estimated_training_finish_paris': finish.astimezone(ZoneInfo('Europe/Paris')).isoformat(),
                'excludes': 'Queued jobs, sampling, metrics and time since the last log; not a whole-study ETA'}
    return out


def command_result(command):
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {'command': command, 'exit_code': done.returncode,
                'stdout': done.stdout.strip(), 'stderr': done.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'exit_code': None, 'stdout': '', 'stderr': str(error)}


def scheduler_state():
    queue = command_result(['squeue', '--noheader', '--array', '--jobs=' + JOB, '--format=%i|%T|%N|%M'])
    accounting = command_result(['sacct', '--noheader', '--parsable2', '--jobs=' + JOB,
        '--format=JobID%32,State%32,ExitCode,Elapsed,NodeList'])
    tasks = {}
    for line in accounting['stdout'].splitlines():
        fields = [field.strip() for field in line.split('|')]
        if len(fields) >= 3 and re.fullmatch(JOB + r'_\d+', fields[0]):
            tasks[fields[0]] = {'state': fields[1].split()[0].rstrip('+'), 'exit_code': fields[2]}
    ended = (queue['exit_code'] == 0 and not queue['stdout'] and accounting['exit_code'] == 0
             and len(tasks) == 6 and all(row['state'] in TERMINAL for row in tasks.values()))
    return {'queue': queue, 'accounting': accounting, 'array_tasks': tasks, 'array_ended': ended,
            'query_failure': queue['exit_code'] != 0 or accounting['exit_code'] != 0}


def load_plan():
    if shared.sha256(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Immutable recovered-task manifest SHA-256 mismatch')
    plan = shared.read_json(MANIFEST)
    identities = [(t['condition_id'], t['arm'], t['seed']) for t in plan['tasks']]
    if plan['study'] != 'rafm_inputs' or len(identities) != 18 or len(set(identities)) != 18:
        raise ValueError('Expected exactly 18 unique recovered A/B/C tasks')
    return plan


def snapshot(plan):
    rows = []
    for task in plan['tasks']:
        cfg = task['config']
        directory = Path(plan['output_root']) / 'final' / task['condition_id'] / task['arm'] / f"seed_{task['seed']}"
        cache, issue = shared.load_cache_manifest(cfg)
        if issue:
            raise ValueError(task['condition_id'] + ': ' + issue)
        row = shared.collect_seed(cfg, task['arm'], task['seed'], directory, cache)
        if row['status'] == 'failed':
            failed = shared.read_json(directory / 'result.json')
            if (failed.get('condition_id'), failed.get('arm'), failed.get('seed')) != (task['condition_id'], task['arm'], task['seed']):
                raise ValueError('Failed result has incorrect task identity: ' + str(directory))
        rows.append({'condition': task['condition_id'], 'arm': task['arm'], 'seed': task['seed'],
            'status': row['status'], 'result_path': row['result_path'], 'result_sha256': row.get('result_sha256'),
            'issues': row['issues'], 'failure': row.get('failure'),
            'nonfinite_metric_fields': row.get('nonfinite_metric_fields'),
            'training': read_training(directory / 'training.jsonl', cfg['training']['steps'])})
    result = {'time': timestamps(), 'array_job_id': JOB, 'manifest': str(MANIFEST),
        'manifest_sha256': MANIFEST_SHA, 'expected_tasks': 18,
        'counts': dict(Counter(row['status'] for row in rows)), 'tasks': rows,
        'scheduler': scheduler_state(), 'status': 'monitoring',
        'policy': 'Read-only monitoring; no experiment retries, source edits, tuning, training or evaluation'}
    queue = {}
    for line in result['scheduler']['queue']['stdout'].splitlines():
        parts = line.split('|')
        if len(parts) >= 2:
            queue[parts[0].strip()] = parts[1].strip()
    for shard in plan['array_phases'][0]['shards']:
        first_unfinished = True
        for index in shard['task_indices']:
            row = rows[index]
            row['worker_index'] = shard['worker_index']
            row['allocation_state'] = queue.get(JOB + '_' + str(shard['worker_index']), 'not_in_queue')
            if row['status'] in ('complete', 'failed', 'awaiting_sample_audit', 'incompatible'):
                row['execution_stage'] = row['status']
            elif row['training']['logged_rows']:
                step = row['training']['latest_logged']['step']
                row['execution_stage'] = ('sampling_or_evaluation_after_final_logged_update'
                    if step >= plan['tasks'][index]['config']['training']['steps'] else 'training_updates_logged')
            elif not first_unfinished:
                row['execution_stage'] = 'waiting_for_previous_task_on_worker'
            elif row['allocation_state'] == 'PENDING':
                row['execution_stage'] = 'waiting_for_scheduler_allocation'
            elif row['allocation_state'] in ('RUNNING', 'CONFIGURING'):
                row['execution_stage'] = 'allocated_prolog_or_startup_no_training_log_yet'
            else:
                row['execution_stage'] = 'no_training_log_allocation_unconfirmed'
            if row['status'] not in ('complete', 'failed'):
                first_unfinished = False
    return result


def run_reporter_once(output):
    path = output / 'reporter.json'
    if path.exists():
        prior = shared.read_json(path)
        if prior.get('status') != 'finished':
            raise RuntimeError('Prior reporter attempt is unresolved; inspect ' + str(path))
        return prior
    command = ['/usr/bin/python3', '-B', str(ROOT / 'tools/report_rafm_completion.py'), '--require-finished']
    record = {'status': 'started', 'time': timestamps(), 'command': command}
    with path.open('x') as stream:
        json.dump(record, stream)
        stream.write('\n')
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    record.update(status='finished', finished=timestamps(), exit_code=completed.returncode,
                  stdout=completed.stdout, stderr=completed.stderr)
    shared.write_json(path, record)
    return record


def monitor(plan, output, hours, once):
    began = time.monotonic()
    deadline = began + hours * 3600
    terminal_since = None
    query_failures = 0
    prior = {}
    event(output, 'monitor_started', array_job_id=JOB, max_hours=hours, interval_s=INTERVAL, once=once)
    while True:
        tick = time.monotonic()
        data = snapshot(plan)
        data.update(monitor_mode='single_snapshot' if once else 'persistent',
                    monitor_elapsed_s=tick - began, max_monitor_hours=hours)
        for row in data['tasks']:
            key = f"{row['condition']}/{row['arm']}/{row['seed']}"
            signature = (row['status'], row['result_sha256'])
            if prior.get(key) != signature:
                event(output, 'task_status_changed', task=key, status=row['status'],
                      result_path=row['result_path'], failure=row['failure'],
                      nonfinite_metric_fields=row['nonfinite_metric_fields'], issues=row['issues'])
                prior[key] = signature
        terminal = all(row['status'] in ('complete', 'failed') for row in data['tasks'])
        if terminal and not once:
            reporter = run_reporter_once(output)
            data['reporter'] = reporter
            code = (0 if not data['counts'].get('failed') else 1) if reporter['exit_code'] == 0 else 2
            data['status'] = 'complete' if code == 0 else 'finished_with_failures' if code == 1 else 'blocked'
            if code == 2:
                data['blocker'] = 'Completion reporter failed or found unresolved A/B/C outcomes; see reporter.json'
        else:
            code = None
            query_failures = query_failures + 1 if data['scheduler']['query_failure'] else 0
            data['consecutive_scheduler_query_failures'] = query_failures
            if data['scheduler']['array_ended']:
                terminal_since = terminal_since if terminal_since is not None else tick
            else:
                terminal_since = None
            if query_failures >= 3:
                data.update(status='blocked', blocker='Three consecutive scheduler query failures; exact command errors recorded')
                code = 2
            elif terminal_since is not None and tick - terminal_since >= 60:
                missing = [{'task': f"{r['condition']}/{r['arm']}/{r['seed']}", 'status': r['status'],
                            'result_path': r['result_path'], 'issues': r['issues']}
                           for r in data['tasks'] if r['status'] not in ('complete', 'failed')]
                data.update(status='blocked', blocker='Scheduler array ended but results/audits remain unresolved after 60-second grace', unresolved=missing)
                code = 2
            elif tick >= deadline:
                data.update(status='timeout', blocker='Eight-hour maximum or configured shorter monitoring deadline reached')
                code = 2
        if once:
            data['status'] = 'single_snapshot'
        shared.write_json(output / 'latest.json', data)
        if once or code is not None:
            event(output, 'monitor_stopped', status=data['status'], counts=data['counts'], blocker=data.get('blocker'))
            print(json.dumps({'status': data['status'], 'time': data['time'], 'latest': str(output / 'latest.json'),
                              'counts': data['counts'], 'blocker': data.get('blocker')}, allow_nan=False), flush=True)
            return 0 if once else code
        time.sleep(max(0, min(INTERVAL - (time.monotonic() - tick), deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--once', action='store_true', help='One read-only snapshot for a launch check; does not run the completion reporter')
    args = parser.parse_args()
    if not math.isfinite(args.hours) or not 0 < args.hours <= 8:
        parser.error('--hours must be positive and at most eight')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.monitor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            code = monitor(load_plan(), args.output, args.hours, args.once)
        except Exception as error:
            record = {'status': 'blocked', 'time': timestamps(), 'array_job_id': JOB,
                      'blocker': str(error), 'error_type': type(error).__name__}
            shared.write_json(args.output / 'latest.json', record)
            event(args.output, 'monitor_execution_failure', **record)
            print(json.dumps(record), flush=True)
            code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
