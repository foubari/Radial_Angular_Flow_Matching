#!/usr/bin/env python3
"""Read existing JSONL losses into an independent, live TensorBoard directory.

Training files are opened read-only. Each service start uses a fresh session and
backfills all saved rows. Decreasing/repeated steps create a separate resume
segment rather than purging history. Event wall times are ingestion times; use
the Step x-axis for reconstructed history. No torch/TensorFlow runtime needed.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import time

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.record_writer import RecordWriter

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'A': 'A_original', 'B': 'B_unit_radius', 'C': 'C_unit_no_radius'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sources(root):
    for path in sorted((root / 'outputs_rafm_input_study/v1/final').glob('*/*/seed_*/training.jsonl')):
        condition, arm, seed = path.parts[-4:-1]
        if arm in ARMS:
            yield path, f'RAFM_inputs/{condition}/{ARMS[arm]}/{seed}', 'loss/angular_mse'
    for path in sorted((root / 'outputs_tflow_full/v2/final').glob('*/seed_*/training.jsonl')):
        condition, seed = path.parts[-3:-1]
        yield path, f'tFlow/final/{condition}/{seed}', 'loss/noise_prediction_mse'
    for path in sorted((root / 'outputs_tflow_full/v2/tuning').glob('*/candidate_*/training.jsonl')):
        condition, candidate = path.parts[-3:-1]
        yield path, f'tFlow/tuning/{condition}/{candidate}', 'loss/noise_prediction_mse'


class Bridge:
    def __init__(self, output):
        self.output = Path(output)
        self.events = self.output / 'events'
        self.events.mkdir(parents=True, exist_ok=True)
        self.cursors = {}
        self.started = time.time()
        self.errors = {}

    def append_event(self, name, tag, row, cursor):
        suffix = '' if cursor['segment'] == 0 else f'/resume_{cursor["segment"]}'
        folder = self.events / (name + suffix)
        folder.mkdir(parents=True, exist_ok=True)
        event_path = folder / f'events.out.tfevents.{int(self.started):010d}.{socket.gethostname()}.loss_bridge'
        new_file = not event_path.exists()
        with event_path.open('ab') as stream:
            writer = RecordWriter(stream)
            if new_file:
                header = event_pb2.Event(wall_time=time.time(), file_version='brain.Event:2')
                writer.write(header.SerializeToString())
            values = [summary_pb2.Summary.Value(tag=tag, simple_value=row['loss'])]
            if 'elapsed_s' in row and math.isfinite(float(row['elapsed_s'])):
                values.append(summary_pb2.Summary.Value(tag='time/training_elapsed_seconds', simple_value=float(row['elapsed_s'])))
            event = event_pb2.Event(wall_time=time.time(), step=row['step'], summary=summary_pb2.Summary(value=values))
            writer.write(event.SerializeToString())
            writer.flush()

    def consume(self, path, name, tag):
        path = Path(path)
        stat = path.stat()
        cursor = self.cursors.setdefault(str(path), {
            'offset': 0, 'inode': stat.st_ino, 'last_step': None, 'segment': 0,
            'rows': 0, 'run_name': name, 'tag': tag})
        if stat.st_ino != cursor['inode'] or stat.st_size < cursor['offset']:
            cursor.update(offset=0, inode=stat.st_ino, last_step=None, segment=cursor['segment'] + 1)
        consumed = 0
        with path.open('rb') as stream:
            stream.seek(cursor['offset'])
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break  # An in-progress append is retried on the next poll.
                if not line.strip():
                    cursor['offset'] = stream.tell()
                    continue
                row = json.loads(line)
                if (isinstance(row.get('step'), bool) or not isinstance(row.get('step'), int)
                        or row['step'] < 0 or isinstance(row.get('loss'), bool)
                        or not isinstance(row.get('loss'), (int, float)) or not math.isfinite(row['loss'])):
                    raise ValueError(f'Invalid/nonfinite loss row at byte {cursor["offset"]}')
                if cursor['last_step'] is not None and row['step'] <= cursor['last_step']:
                    cursor['segment'] += 1
                self.append_event(name, tag, row, cursor)
                cursor.update(offset=stream.tell(), last_step=row['step'], rows=cursor['rows'] + 1)
                consumed += 1
        return consumed

    def poll(self, root):
        added = 0
        for path, name, tag in sources(Path(root)):
            try:
                added += self.consume(path, name, tag)
                self.errors.pop(str(path), None)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.errors[str(path)] = f'{type(error).__name__}: {error}'
        status = {'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'status': 'running' if not self.errors else 'running_with_log_errors',
                  'host': socket.gethostname(), 'pid': os.getpid(), 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
                  'logdir': str(self.events.resolve()), 'source_files': len(self.cursors),
                  'rows_imported': sum(c['rows'] for c in self.cursors.values()), 'rows_added_last_poll': added,
                  'sources': self.cursors, 'errors': self.errors,
                  'training_files_modified': False, 'wall_time_semantics': 'ingestion time; historical plots use Step',
                  'loss_semantics': 'saved instantaneous batch losses; angular and noise losses are distinct objectives'}
        write_json(self.output / 'bridge_status.json', status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=15.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'bridge.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'bridge_status.json').exists():
            raise FileExistsError('Use a fresh monitoring session to backfill on restart; existing events are preserved')
        bridge = Bridge(args.output)
        stopping = False

        def stop(*_):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not stopping:
            report = bridge.poll(args.root)
            if args.once or report['rows_added_last_poll']:
                print(json.dumps({k: report[k] for k in ('updated_utc', 'status', 'source_files', 'rows_imported', 'rows_added_last_poll')}), flush=True)
            if args.once:
                break
            for _ in range(max(1, math.ceil(args.interval))):
                if stopping:
                    break
                time.sleep(1)


if __name__ == '__main__':
    main()
