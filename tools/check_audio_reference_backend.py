#!/usr/bin/env python3
"""Re-evaluate saved fixed-spherical Y/X with the new study's classifier path.

No generation, retraining, source draws, gain rescaling or data preprocessing.
Every saved sample file is checked before use. This measures whether moving
normalization to CPU and using the study's deterministic backend changes the
previously saved digit predictions or reference distribution metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from experiments.poc_audio.audio_classifier import Clf
from experiments.poc_audio.audio_empirical_gain import summarize, flat_metrics
from experiments.tflow import run as common
from experiments.tflow.data import sha256
from rafm.utils.seeds import set_all_seeds

SEEDS = [8925, 1234, 7]
N, DIM = 2000, 16254
DEFAULT_AGGREGATE = ROOT / 'outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json'
DEFAULT_MANIFEST = ROOT / 'outputs_audio_gain/delivery_manifest.json'
DATA_OUTPUT = Path('/mnt/vast01/users/fouad.oubari/data/rafm_input_study/reference_backend_check')
ENERGY_COMPATIBILITY_KEYS = ('energy_KS', 'cov>q95', 'cov>q99', 'cov<q10', 'PIT', 'cov>q90')


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_file(spec, records):
    path = Path(spec['path'])
    if not path.is_absolute():
        path = ROOT / path
    actual = sha256(path)
    if actual != spec['sha256']:
        raise ValueError(f'SHA-256 mismatch for {path}: expected {spec["sha256"]}, got {actual}')
    if 'size_bytes' in spec and path.stat().st_size != spec['size_bytes']:
        raise ValueError(f'Byte-size mismatch for {path}')
    record = {'path': str(path.resolve()), 'sha256': actual, 'size_bytes': path.stat().st_size, 'status': 'passed'}
    records.append(record)
    return path


def manifest_entry(manifest, path):
    path = Path(path).resolve()
    matches = [entry for entry in manifest['files'] if (ROOT / entry['path']).resolve() == path]
    if len(matches) != 1:
        raise ValueError(f'Delivery manifest must list exactly one entry for {path}')
    return matches[0]


def verify_delivery(path, aggregate_path, records):
    """Anchor the delivery manifest to its existing local result commit."""
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT))
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', relative], cwd=ROOT, text=True).strip()
    if not commit:
        raise ValueError('Delivery manifest has no committed provenance')
    committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT)
    expected = hashlib.sha256(committed).hexdigest()
    verify_file({'path': str(path), 'sha256': expected}, records)
    manifest = load_json(path)
    if manifest.get('status') != 'verified_report_consistency':
        raise ValueError('Original delivery manifest is not a verified report')
    verify_file(manifest_entry(manifest, aggregate_path), records)
    return manifest, {'path': str(path), 'sha256': expected, 'result_commit': commit,
                      'commit_role': 'Recorded evaluation delivery; not a training commit'}


def finite_tensor(tensor, name, shape=None, dtype=None):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'{name} is not a tensor')
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f'{name} has wrong shape: {tuple(tensor.shape)}')
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f'{name} has wrong dtype: {tensor.dtype}')
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f'Nonfinite tensor: {name}')


def backend_flags():
    return {'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'normalization_device': 'cpu', 'normalization_dtype': 'float32',
            'classifier_device': 'cuda:0', 'classifier_dtype': 'float32',
            'autocast': False, 'classifier_batch_size': N, 'metric_seed': 0}


def evaluate_saved(samples, labels, cached_logits, classifier, test_gains, old_summary, device):
    """Literal classifier path used by experiments.rafm_inputs.run.quality."""
    begin = time.perf_counter()
    with torch.no_grad():
        norms = samples.norm(dim=1, keepdim=True)
        if bool((norms <= 0).any().item()) or not bool(torch.isfinite(norms).all().item()):
            raise FloatingPointError('Nonpositive or nonfinite saved audio radius')
        logits = classifier((samples / norms).reshape(-1, 2, 129, 63).to(device)).cpu()
    finite_tensor(logits, 'new classifier logits', (N, 10), torch.float32)
    common._synchronize(device)
    classifier_s = time.perf_counter() - begin
    summary = summarize(samples, logits, labels, test_gains)
    metrics = flat_metrics(summary, full_precision=True)
    old_metrics = flat_metrics(old_summary, full_precision=True)
    cached_predictions = cached_logits.argmax(1)
    if int((cached_predictions == labels).sum()) != old_summary['unrounded']['correct_count']:
        raise ValueError('Saved classifier logits disagree with the original measured correct count')
    if old_summary['unrounded']['digit_acc'] != int((cached_predictions == labels).sum()) / N:
        raise ValueError('Original per-seed accuracy does not match its saved logits')
    comparison = {
        'prediction_disagreements_vs_cached': int((logits.argmax(1) != cached_predictions).sum()),
        'maximum_logit_absolute_difference': float((logits - cached_logits).abs().max()),
        'mean_logit_absolute_difference': float((logits - cached_logits).abs().mean()),
        'metric_deltas': {key: metrics[key] - old_metrics[key] for key in metrics},
        'old_metrics': old_metrics,
        'energy_ks_tail_metrics_exactly_equal': all(metrics[key] == old_metrics[key] for key in ENERGY_COMPATIBILITY_KEYS),
        'energy_compatibility_keys': list(ENERGY_COMPATIBILITY_KEYS),
        'all_unrounded_energy_deltas': {key: value - old_summary['unrounded']['energy'][key]
                                      for key, value in summary['unrounded']['energy'].items()},
        'mean_confidence_delta': summary['unrounded']['mean_confidence'] - old_summary['unrounded']['mean_confidence'],
        'rounded_energy_exactly_equal': summary['energy'] == old_summary['energy'],
        'energy_bin_accuracies_exactly_equal': summary['content']['acc_by_energy'] == old_summary['content']['acc_by_energy'],
    }
    clean, bad = common.sanitize_metrics({'summary': summary, 'metrics': metrics, 'comparison': comparison})
    if bad:
        raise FloatingPointError(f'Nonfinite/undefined measured reference fields: {bad}')
    return {**clean, 'classifier_time_s': classifier_s}, logits


def aggregate_metrics(rows):
    return {arm: {key: {'mean': statistics.mean(values), 'std_population': statistics.pstdev(values),
                       'values': values, 'n': len(values)}
                  for key in rows[0][arm]['metrics']
                  for values in [[row[arm]['metrics'][key] for row in rows]]}
            for arm in ('baseline', 'posthoc')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument('--delivery-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/rafm_input_study/prepared/audiomnist_stft.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs_rafm_input_study/v1/audio_reference_backend_check.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Preserve previous audio reference backend check: {output}')
    result = {'schema_version': 1, 'status': 'running', 'verification': [], 'runs': [],
              'generation_performed': False, 'training_updates': 0, 'gain_draws': 0,
              'gain_rescaling_performed': False, 'existing_artifacts_modified': False,
              'purpose': 'Measured compatibility of saved fixed-spherical Y/X under the new study classifier path'}
    begin = time.perf_counter()
    try:
        device = common.compute_device()
        torch.cuda.init()
        result['hardware'] = common.hardware()
        manifest, proof = verify_delivery(args.delivery_manifest, args.aggregate, result['verification'])
        result['delivery_manifest'] = proof
        aggregate = load_json(args.aggregate)
        result['aggregate'] = {'path': str(args.aggregate.resolve()), 'sha256': sha256(args.aggregate),
                               'original_status': aggregate['status']}
        if aggregate.get('status') not in ('complete', 'baseline_mismatch') or [row['seed'] for row in aggregate['runs']] != SEEDS:
            raise ValueError('Expected all three completed saved fixed-spherical seeds')
        cfg = load_json(args.config)
        result['config_sha256'] = common.json_hash(cfg)
        result['config'] = cfg
        result['original_historical_reproduction_discrepancy_preserved'] = aggregate['status'] == 'baseline_mismatch'
        protocol = aggregate['protocol']
        if (cfg['kind'] != 'audio' or cfg['seeds'] != SEEDS or cfg['evaluation']['metric_seed'] != 0
                or cfg['evaluation']['n_samples'] != N or cfg['evaluation']['sample_seed'] != 0
                or cfg['evaluation']['model_evaluations'] != 160 or cfg['data']['transform'] != 'as_cached'
                or protocol['n_gen'] != N or protocol['class_counts'] != [200] * 10
                or protocol['sample_seed'] != 0 or protocol['model_evaluations'] != 160
                or protocol['classifier_batch_size'] != N):
            raise ValueError('Prepared evaluation and saved reference protocols differ')
        specs = {'classifier': cfg['evaluation']['classifier'], 'test_file': cfg['data']['external_test'],
                 'train_file': cfg['data']['input']}
        for name, spec in specs.items():
            old = aggregate['inputs'][name]
            if old['sha256'] != spec['sha256'] or Path(old['path']).resolve() != Path(spec['path']).resolve():
                raise ValueError(f'Prepared/reference input identity mismatch: {name}')
            verify_file(spec, result['verification'])
        result['classifier'] = dict(specs['classifier'])
        result['external_test'] = dict(specs['test_file'])
        source_files = ['tools/check_audio_reference_backend.py', 'experiments/rafm_inputs/run.py',
                        'experiments/tflow/run.py', 'rafm/utils/seeds.py',
                        'experiments/poc_audio/audio_classifier.py', 'experiments/poc_audio/audio_empirical_gain.py']
        result['source_sha256'] = {name: sha256(ROOT / name) for name in source_files}
        for name in ('experiments/poc_audio/audio_classifier.py', 'experiments/poc_audio/audio_empirical_gain.py'):
            expected = aggregate['source_sha256'][str(ROOT / name)]
            verify_file({'path': str(ROOT / name), 'sha256': expected}, result['verification'])
        split_path = verify_file(cfg['data']['split']['file'], result['verification'])
        split = torch.load(split_path, map_location='cpu', weights_only=True)
        for key, old_key in (('train', 'training_indices_sha256'), ('val', 'internal_validation_indices_sha256')):
            if common.tensor_fingerprint(split[key])['sha256'] != aggregate['split'][old_key]:
                raise ValueError(f'Training/validation split identity mismatch: {key}')
        result['generator_split_indices_match'] = True
        for row in aggregate['runs']:
            seed_json = args.aggregate.parent / f'seed_{row["seed"]}' / 'eval.json'
            verify_file(manifest_entry(manifest, seed_json), result['verification'])
            if load_json(seed_json) != row:
                raise ValueError(f'Per-seed JSON differs from delivered aggregate: seed {row["seed"]}')
            verify_file(row['samples'], result['verification'])
        # Check and reuse the saved paired-gain artifact; no ECDF draw occurs.
        paired_specs = {row['paired_gains']['path']: row['paired_gains'] for row in aggregate['runs']}
        if len(paired_specs) != 1:
            raise ValueError('Expected one shared saved gain vector for all three seeds')
        paired_path = verify_file(next(iter(paired_specs.values())), result['verification'])
        paired = torch.load(paired_path, map_location='cpu', weights_only=True)
        finite_tensor(paired['gains'], 'saved paired gains', (N,), torch.float32)
        test = torch.load(specs['test_file']['path'], map_location='cpu', weights_only=True)
        finite_tensor(test['g'], 'external test gains', (3000,), torch.float32)
        if bool((test['g'] <= 0).any()):
            raise ValueError('External test gains must be positive')
        test_gains = test['g'].numpy()
        del test
        identifier = common.json_hash({'aggregate': result['aggregate']['sha256'],
            'config': result['config_sha256'], 'source': result['source_sha256'], 'report': str(output)})
        logits_dir = DATA_OUTPUT / identifier
        if logits_dir.exists():
            raise FileExistsError(f'Preserve previous backend-check logits: {logits_dir}')
        logits_dir.mkdir(parents=True)
        result['logits_directory'] = str(logits_dir)
        result['backend_before_study_seed_helper'] = backend_flags()
        for original in aggregate['runs']:
            seed = original['seed']
            print(f'Re-evaluating saved AudioMNIST Y/X seed {seed}, no generation', flush=True)
            seed_begin = time.perf_counter()
            payload = torch.load(original['samples']['path'], map_location='cpu', weights_only=True)
            for name in ('Y', 'X'):
                finite_tensor(payload[name], f'seed {seed} {name}', (N, DIM), torch.float32)
            for name in ('logits_before', 'logits_after'):
                finite_tensor(payload[name], f'seed {seed} cached {name}', (N, 10), torch.float32)
            finite_tensor(payload['digit'], f'seed {seed} labels', (N,), torch.int64)
            finite_tensor(payload['gains'], f'seed {seed} gains', (N,), torch.float32)
            finite_tensor(payload['initial_radii'], f'seed {seed} initial radii')
            labels = payload['digit']
            if not torch.equal(labels, torch.arange(10).repeat_interleave(200)):
                raise ValueError(f'Saved seed {seed} labels do not match the class-balanced protocol')
            if not torch.equal(payload['gains'], paired['gains']):
                raise ValueError(f'Saved seed {seed} gains differ from shared gains')
            set_all_seeds(cfg['evaluation']['metric_seed'])
            result['backend_flags'] = backend_flags()
            classifier = Clf(ncls=10).to(device).eval()
            classifier.load_state_dict(torch.load(specs['classifier']['path'], map_location='cpu', weights_only=True), strict=True)
            torch.cuda.reset_peak_memory_stats(device)
            baseline, before = evaluate_saved(payload['Y'], labels, payload['logits_before'], classifier,
                test_gains, original['baseline'], device)
            posthoc, after = evaluate_saved(payload['X'], labels, payload['logits_after'], classifier,
                test_gains, original['posthoc'], device)
            disagreements = int((before.argmax(1) != after.argmax(1)).sum())
            invariance = {'prediction_disagreements': disagreements,
                'accuracy_difference': posthoc['metrics']['digit_acc'] - baseline['metrics']['digit_acc'],
                'passed': disagreements == 0}
            logits_path = logits_dir / f'seed_{seed}_logits.pt'
            torch.save({'seed': seed, 'digit': labels, 'logits_before': before, 'logits_after': after,
                        'input_samples_sha256': original['samples']['sha256'], 'config_sha256': result['config_sha256']}, logits_path)
            compatible = (disagreements == 0 and all(value['comparison']['prediction_disagreements_vs_cached'] == 0
                and value['comparison']['energy_ks_tail_metrics_exactly_equal'] for value in (baseline, posthoc)))
            row = {'seed': seed, 'n': N, 'status': 'passed' if compatible else 'compatibility_discrepancy',
                'input_samples': original['samples'], 'baseline': baseline, 'posthoc': posthoc,
                'invariance': invariance, 'new_logits': {'path': str(logits_path), 'sha256': sha256(logits_path)},
                'runtime': {'total_s': time.perf_counter() - seed_begin,
                            'classifier_s': baseline['classifier_time_s'] + posthoc['classifier_time_s'],
                            'peak_memory': common.peak_memory(device), 'timing_kind': 'measured'}}
            result['runs'].append(row)
            common.write_json(output, result)
            del payload, classifier, before, after
            torch.cuda.empty_cache()
        result['aggregate_metrics'] = aggregate_metrics(result['runs'])
        result['totals'] = {
            'old_vs_new_prediction_disagreements': sum(row[arm]['comparison']['prediction_disagreements_vs_cached']
                for row in result['runs'] for arm in ('baseline', 'posthoc')),
            'postrescale_prediction_disagreements': sum(row['invariance']['prediction_disagreements'] for row in result['runs']),
            'saved_generated_examples': 6000, 'classified_versions': 12000,
            'energy_ks_tail_metrics_exactly_equal': all(row[arm]['comparison']['energy_ks_tail_metrics_exactly_equal']
                for row in result['runs'] for arm in ('baseline', 'posthoc'))}
        result['status'] = 'passed' if all(row['status'] == 'passed' for row in result['runs']) else 'compatibility_discrepancy'
        result['runtime_s'] = time.perf_counter() - begin
        result['interpretation'] = ('Measured compatibility of these same saved outputs; this does not resolve the separate '
            'historical fixed-spherical accuracy reproduction discrepancy, and does not test a new generative model.')
        common.write_json(output, result)
        print(json.dumps({'status': result['status'], 'totals': result['totals'], 'output': str(output)}), flush=True)
        if result['status'] != 'passed':
            raise SystemExit(1)
    except Exception as error:
        result.update(status='failed', runtime_s=time.perf_counter() - begin,
                      failure={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        common.write_json(output, result)
        raise


if __name__ == '__main__':
    main()
