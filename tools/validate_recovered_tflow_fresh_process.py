"""Disposable public-trainer check for recovered ImageNette, on one scheduled GPU.

Adapted from validate_tflow_fresh_process.py at the pinned hash below. Preserves
its 4 -> 6 resumed versus 6 uninterrupted update check (12 actual updates).
Necessary differences: accept the image condition, sample 40 balanced examples,
initialize through the existing allocator helper directly, and use isolated
output/provenance. No selection, benchmark evaluation, or final training runs.

Invoke directly through tools/recovered_study_job.sh. The active frozen
experiment_entrypoint.py module whitelist is deliberately unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'tools/validate_tflow_fresh_process.py'
ORIGINAL_SHA256 = '14b7a6ded9550caa459925abfc6863e29af0cb1e44f0a88ca98fc627cc7ced27'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(cfg, root):
    # Imports occur only after main() calls the existing compute guard/allocator.
    import torch
    from baselines.tflow_core import TFlowSourceConfig
    from experiments.tflow import run
    from experiments.tflow.data import load_data, sha256
    from experiments.tflow.validation import source_candidates

    output = Path(root) / cfg['condition_id']
    if output.exists():
        raise FileExistsError('Preserve previous recovered fresh-process validation')
    output.mkdir(parents=True)
    initialized_before_metadata = torch.cuda.is_initialized()
    row = {'status': 'running', 'purpose': 'disposable_public_trainer_validation',
        'condition_id': cfg['condition_id'], 'test_data_used': False,
        'source_selection_performed': False, 'benchmark_metrics_computed': False,
        'config_sha256': run.json_hash(cfg), 'implementation_sha256': run.implementation_sha256(cfg),
        'entrypoint_sha256': sha256(ROOT / 'tools/experiment_entrypoint.py'),
        'validator_sha256': sha256(Path(__file__)),
        'original_validator_path': str(ORIGINAL), 'original_validator_sha256': ORIGINAL_SHA256,
        'adaptations': ['Image condition only', '40 samples instead of vector fallback 128; four per class',
            'Direct call to existing initialize_accelerator, without changing its whitelist',
            'Dedicated recovered-entrypoint-check output'],
        'hardware': run.hardware(), 'gpu_initialized_before_training': initialized_before_metadata,
        'batch_size': cfg['training']['batch_size'], 'precision': cfg['training']['precision'],
        'ema': cfg['training']['ema'],
        'source_policy': 'Fixed nu=5 and scale_multiplier=1 training-median calibration; no candidate scoring or validation/test selection',
        'data_use_note': 'Unchanged cached tensor is loaded and public-trainer provenance hashes its splits; '
            'held-out rows are not used for optimization, source calibration or quality metrics.'}
    try:
        if not initialized_before_metadata:
            raise RuntimeError('GPU not initialized before trainer')
        if (cfg['condition_id'] != 'imagenette_dcae' or cfg['kind'] != 'image'
                or cfg['model']['kind'] != 'image_sit'
                or cfg['training']['batch_size'] != 64
                or cfg['training']['precision'] != 'bfloat16_autocast'
                or cfg['training']['steps'] != 40000
                or cfg['training']['ema'] != 0.9999
                or cfg['evaluation']['model_evaluations'] != 100):
            raise ValueError('Expected unchanged recovered ImageNette SiT/batch64/bfloat16/EMA/100-call configuration')
        data = load_data(cfg, include_external_test=False)
        choice = next(x for x in source_candidates(data.split('train'))
                      if x['nu'] == 5 and x['scale_multiplier'] == 1)
        source = TFlowSourceConfig(choice['nu'], choice['scale'])
        seed = 46021
        first = run.train(cfg, data, output / 'resumed', seed, source, budget=4, stage='tuning')
        model, saved = run.load_trained(cfg, data, output / 'resumed', seed, source, stage='tuning')
        if saved['step'] != 4:
            raise RuntimeError('Wrong disposable checkpoint budget')
        generated = run.sample(cfg, model, source, 40, seed=61717)
        if not bool(torch.isfinite(generated['samples']).all()):
            raise FloatingPointError('Nonfinite public-trainer samples')
        if generated['samples'].shape != (40, cfg['data']['shape'][1]):
            raise RuntimeError('Wrong disposable sample count or shape')
        labels = generated['labels']
        if labels is None or labels.shape != (40,):
            raise RuntimeError('Image class labels missing or malformed')
        counts = torch.bincount(labels, minlength=10).tolist()
        if counts != [4] * 10:
            raise RuntimeError('Disposable image samples are not class-balanced')
        if (generated['nfe'] != 100 or generated['n_batches'] != 1
                or generated['model_calls_total'] != 100):
            raise RuntimeError('Actual image sampling network budget mismatch')
        sampler_proof = {key: generated[key] for key in
            ('nfe', 'n_batches', 'model_calls_total', 'sample_time_s', 'peak_memory')}
        del model, saved, generated
        resumed = run.train(cfg, data, output / 'resumed', seed, source, budget=6, stage='tuning')
        direct = run.train(cfg, data, output / 'uninterrupted', seed, source, budget=6, stage='tuning')
        a = torch.load(output / 'resumed/checkpoint.pt', map_location='cpu', weights_only=False)
        b = torch.load(output / 'uninterrupted/checkpoint.pt', map_location='cpu', weights_only=False)
        errors = {}
        for name in ('model', 'ema'):
            if a[name] is None:
                if b[name] is not None:
                    raise RuntimeError('EMA checkpoint mismatch')
                continue
            if b[name] is None or set(a[name]) != set(b[name]):
                raise RuntimeError('Checkpoint state keys differ: ' + name)
            for key in a[name]:
                if not torch.equal(a[name][key], b[name][key]):
                    errors[name + '.' + key] = float((a[name][key] - b[name][key]).abs().max())
        if errors:
            raise RuntimeError('Resume differs from uninterrupted trajectory: ' + str(errors))
        if a['step'] != 6 or b['step'] != 6:
            raise RuntimeError('Wrong final disposable checkpoint budget')
        row.update(status='passed', initial_updates=4, resumed_updates=2, uninterrupted_updates=6,
            disposable_optimizer_updates_total=12, checkpoint_load=True, resume_bitwise_identical=True,
            finite_samples=True, n_generated=40, class_counts=counts, sample_seed=61717, seed=seed,
            nfe=100, n_batches=1, model_calls_total=100, sampler=sampler_proof,
            source=vars(source), source_calibration=choice,
            first_training=first, resumed_training=resumed, uninterrupted_training=direct)
        run.write_json(output / 'check.json', row)
        return row
    except Exception as error:
        row.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                           'traceback': traceback.format_exc()})
        run.write_json(output / 'check.json', row)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default=str(ROOT / 'outputs_tflow_full/recovered/entrypoint_checks'))
    args = parser.parse_args()
    if source_hash(ORIGINAL) != ORIGINAL_SHA256:
        raise ValueError('Original helper changed; re-review recovered adaptation provenance')
    from tools.experiment_entrypoint import initialize_accelerator
    initialize_accelerator()
    print(json.dumps(check(json.loads(Path(args.config).read_text()), args.output), indent=2))


if __name__ == '__main__':
    main()
