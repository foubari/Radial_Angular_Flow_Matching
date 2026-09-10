#!/usr/bin/env python3
"""Check the read-only loss bridge using isolated fixtures on a compute node."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs_monitoring/bridge_checks.json')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this check through the cluster scheduler on a compute node')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from loss_tensorboard_bridge import Bridge

    began = time.perf_counter()
    checks = []
    report = {
        'schema_version': 1, 'status': 'running', 'fixture_data_only': True,
        'host': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'checker_sha256': sha256(__file__),
        'bridge_sha256': sha256(ROOT / 'tools/loss_tensorboard_bridge.py'),
        'checks': checks,
    }
    cache = ROOT / 'outputs_monitoring/cache'
    cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def require(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    try:
        with tempfile.TemporaryDirectory(prefix='bridge_check_', dir=cache) as temporary:
            fixture = Path(temporary)
            paths = [
                fixture / 'outputs_rafm_input_study/v1/final/toy/A/seed_8925/training.jsonl',
                fixture / 'outputs_tflow_full/v2/final/toy/seed_8925/training.jsonl',
                fixture / 'outputs_tflow_full/v2/tuning/toy/candidate_00/training.jsonl',
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = [{'step': 100, 'loss': 2.0 + index}, {'step': 200, 'loss': 1.0 + index}]
                if index == 0:
                    for row in rows:
                        row.update(arm='A', seed=8925, elapsed_s=row['step'] / 10)
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            bridge = Bridge(fixture / 'monitor')

            def poll(expected_added, expected_total):
                before = {str(path): path.read_bytes() for path in paths}
                status = bridge.poll(fixture)
                if before != {str(path): path.read_bytes() for path in paths}:
                    raise AssertionError('Bridge modified source bytes')
                if status['errors'] or status['rows_added_last_poll'] != expected_added or status['rows_imported'] != expected_total:
                    raise AssertionError(f'Unexpected poll counts/errors: {status}')
                return status

            def scalar_values(name, tag):
                accumulator = EventAccumulator(str(bridge.events / name), size_guidance={'scalars': 0})
                accumulator.Reload()
                return [(event.step, event.value) for event in accumulator.Scalars(tag)], accumulator.Tags()['scalars']

            angular_name = 'RAFM_inputs/toy/A_original/seed_8925'
            angular_tag = 'loss/angular_mse'
            noise_tag = 'loss/noise_prediction_mse'
            poll(6, 6)
            poll(0, 6)
            values, tags = scalar_values(angular_name, angular_tag)
            require(values == [(100, 2.0), (200, 1.0)], 'backfill_and_repeated_poll_have_no_duplicate_events')
            require('time/training_elapsed_seconds' in tags, 'saved_training_elapsed_time_is_available')
            for name in ['tFlow/final/toy/seed_8925', 'tFlow/tuning/toy/candidate_00']:
                values, tags = scalar_values(name, noise_tag)
                require(len(values) == 2 and angular_tag not in tags, f'distinct_noise_objective_tag:{name}')
            _, tags = scalar_values(angular_name, angular_tag)
            require(noise_tag not in tags, 'angular_run_does_not_expose_noise_loss_tag')

            with paths[0].open('ab') as stream:
                stream.write(json.dumps({'step': 300, 'loss': 0.5}).encode())
            poll(0, 6)
            values, _ = scalar_values(angular_name, angular_tag)
            require(len(values) == 2, 'incomplete_final_jsonl_row_is_deferred')
            with paths[0].open('ab') as stream:
                stream.write(b'\n')
            poll(1, 7)
            poll(0, 7)
            values, _ = scalar_values(angular_name, angular_tag)
            require(values == [(100, 2.0), (200, 1.0), (300, 0.5)], 'completed_partial_row_is_imported_exactly_once')

            with paths[0].open('a') as stream:
                stream.write(json.dumps({'step': 150, 'loss': 0.75}) + '\n')
            poll(1, 8)
            poll(0, 8)
            original, _ = scalar_values(angular_name, angular_tag)
            resumed, _ = scalar_values(angular_name + '/resume_1', angular_tag)
            require(original == [(100, 2.0), (200, 1.0), (300, 0.5)] and resumed == [(150, 0.75)],
                    'decreasing_step_creates_resume_1_and_preserves_original_history')
            require(all(path.exists() for path in paths), 'source_files_preserved')
            checks.append('source_bytes_unchanged_across_all_bridge_polls')
            report['fixture_source_sha256'] = {str(path.relative_to(fixture)): sha256(path) for path in paths}
            report['verified_rows'] = 8
        require('torch' not in sys.modules, 'torch_was_not_imported')
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        report['runtime_s'] = time.perf_counter() - began
        report['torch_imported'] = 'torch' in sys.modules
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
