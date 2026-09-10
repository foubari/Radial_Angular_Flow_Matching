"""Exercise the unchanged offline image evaluator on 40 cached validation rows.

This is a dependency/numerical smoke test, never a generative-model benchmark.
It trains no model, draws no generator samples, and does not modify the FID cache.
Run only in a single-GPU Slurm allocation after recovered-input preparation passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/mnt/vast01/users/fouad.oubari/data/rafm_input_study/recovered_eval_smoke')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
        parser.error('Image-evaluator smoke requires a single-GPU Slurm compute allocation')
    if not args.config.is_absolute():
        parser.error('--config must be absolute')
    identifier = os.environ['SLURM_JOB_ID'] + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir or ROOT / 'docs/recovered_artifacts' / ('image_evaluation_smoke_' + identifier)
    if not output.is_absolute():
        parser.error('--output-dir must be absolute')
    output.mkdir(parents=True, exist_ok=False)
    generated = DATA_ROOT / identifier
    if generated.exists():
        raise FileExistsError('Preserve an existing evaluator smoke directory: ' + str(generated))
    report = {'schema_version': 1, 'status': 'running', 'smoke_only': True,
        'benchmark_result': False, 'no_model_training': True, 'no_generator_sampling': True,
        'generator_network_evaluations': 0, 'n_cached_validation_rows': 40,
        'generated_images_directory': str(generated), 'output_directory': str(output),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'host': platform.node(),
        'config_path': str(args.config), 'config_file_sha256': sha256(args.config),
        'source_path': str(Path(__file__).resolve()), 'source_sha256': sha256(__file__),
        'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'smoke.json', report)
    began = time.perf_counter()
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import torch
        from experiments.tflow import run as common
        from experiments.tflow.data import load_data
        from baselines.tflow_downstream import evaluate_image, image_reference_manifest
        cfg = json.loads(args.config.read_text())
        if cfg['condition_id'] != 'imagenette_dcae' or cfg['kind'] != 'image':
            raise ValueError('This smoke accepts only the recovered ImageNette condition')
        common.validate_config(cfg)
        device = common.compute_device()
        torch.cuda.init()
        torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
        manifest_path = Path(cfg['shared_cache']['manifest_path'])
        if sha256(manifest_path) != cfg['shared_cache']['manifest_sha256']:
            raise ValueError('Shared-cache manifest identity mismatch')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'complete' or manifest.get('condition_id') != 'imagenette_dcae':
            raise ValueError('Recovered shared cache is not complete')
        data = load_data(cfg)
        # Same evaluator API and frozen resources as the benchmark, but neither
        # generated-model outputs nor the final latent test split are used here.
        validation = data.split('val')
        labels = data.split_labels('val')
        if len(validation) < 40 or labels is None:
            raise ValueError('Need 40 original validation rows and their class labels')
        samples = validation[:40].clone()
        sample_labels = labels[:40].clone()
        settings = dict(cfg['evaluation'])
        settings['device'] = str(device)
        settings['generated_image_dir'] = str(generated)
        before = image_reference_manifest(settings['real_reference_dir'])
        if before['sha256'] != settings['expected_reference_sha256'] or before['count'] != 3925:
            raise ValueError('Original reference identity mismatch before smoke')
        torch.cuda.reset_peak_memory_stats(device)
        common._synchronize(device)
        evaluation = evaluate_image(samples, validation, train_mean=data.mean, cfg=settings,
            output_dir=output / 'evaluator', labels=sample_labels)
        common._synchronize(device)
        after = image_reference_manifest(settings['real_reference_dir'])
        if after != before:
            raise ValueError('Reference cache changed during evaluator smoke')
        pngs = sorted(generated.glob('*.png'))
        if len(pngs) != 40:
            raise ValueError('Evaluator did not write exactly 40 decoded validation PNGs')
        clean, bad = common.sanitize_metrics({'image': evaluation['image'], 'latent': evaluation['latent']})
        if bad:
            raise FloatingPointError('Nonfinite evaluator smoke fields: ' + str(bad))
        report.update(status='passed', all_image_and_latent_metrics_finite=True,
            reference_unchanged=True,
            exact_data=common.dataset_manifest(data),
            validation_rows=common.tensor_fingerprint(data.indices['val'][:40]),
            latent_reference_split='validation_only_for_this_smoke',
            class_counts=torch.bincount(sample_labels, minlength=10).tolist(),
            hardware=common.hardware(), peak_memory=common.peak_memory(device),
            evaluation_proof=evaluation, smoke_metrics=clean,
            settings_note='Unchanged decoder feature batches and precision; input consists of cached validation rows, not generator outputs',
            original_reference={key: before[key] for key in ('path', 'count', 'sha256')},
            decoded_pngs=[{'name': path.name, 'sha256': sha256(path)} for path in pngs])
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc())
    report.update(elapsed_s=time.perf_counter() - began, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'smoke.json', report)
    print(json.dumps({'status': report['status'], 'smoke_only': True, 'report': str(output / 'smoke.json'),
        'elapsed_s': report['elapsed_s'], 'error': report.get('error')}, allow_nan=False), flush=True)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
