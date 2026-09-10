"""Freeze the newly authorized shared RAFM-Ang / t-Flow study inputs.

Planning uses stdlib only. Materialization imports numerical libraries only after
checking a Slurm allocation, never changes existing cached inputs, and publishes
immutable data directories by atomic rename. New synthetic tensors are explicitly
labelled new realizations, not recovered historical data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/mnt/vast01/users/fouad.oubari/data/rafm_input_study/v1')
DEFAULT_CONFIG = ROOT / 'configs/rafm_input_study'
STUDY_ID = 'rafm_input_parameterization_v1'
AUTHORIZED_DATE = '2026-09-10'
GENERATION_NAMESPACE = 'rafm_input_parameterization_v1:20260910:'
KAPPAS = (1, 3, 10, 30, 100, 300)
BLOCKED = {
    'piv_d32': ['piv32_identity_missing'],
    'imagenette_dcae': ['image_split_conflict', 'historical_eval_provenance'],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def generation_spec(condition, shape):
    seed = int.from_bytes(hashlib.sha256((GENERATION_NAMESPACE + condition).encode()).digest()[:4], 'big') % (2**31)
    common = {'realization': 'new_matched_comparison_not_historical',
              'sample_seed': seed, 'split_seed': 0, 'shape': shape,
              'device': 'cpu', 'dtype': 'float32'}
    match = re.fullmatch(r'student_t_d(\d+)_df([\d.]+)_cor', condition)
    if match:
        return dict(common, family='student_t', dim=int(match[1]), df=float(match[2]),
                    matrix_seed=42, correlated=True,
                    source_files=['rafm/data/student_t.py', 'rafm/data/base.py'],
                    target_noise='independent univariate Student-t coordinates, then z @ A.T; unchanged target generator')
    if condition == 'gaussian_aniso_d16_cor':
        return dict(common, family='gaussian_aniso', dim=16, matrix_seed=42, correlated=True,
                    source_files=['rafm/data/gaussian_aniso.py', 'rafm/data/base.py'])
    if condition == 'toy_radial_angular':
        return dict(common, family='toy_radial_angular', dim=2, df=3.0, n_modes=4,
                    kappa=5.0, scale=1.0,
                    source_files=['rafm/data/toy_radial_angular.py', 'rafm/data/base.py'],
                    angular_generator='existing Gaussian angle approximation; not exact von Mises')
    match = re.fullmatch(r'aniso_k(\d+)', condition)
    if match:
        return dict(common, family='aniso_sweep', dim=32, kappa=int(match[1]),
                    sample_seed=42, numpy_generator='numpy.random.default_rng / PCG64',
                    matrix_seed=42, kappa_order=list(KAPPAS),
                    source_files=['rebuttal_experiments/scripts/gen_aniso.py'],
                    replay_rule='U,V then one Nxd normal array for each kappa in fixed order; discard preceding arrays',
                    historical_identity_claim=False)
    return None


def make_draft(original, data_root):
    cfg = copy.deepcopy(original)
    condition = cfg['condition_id']
    generation = generation_spec(condition, cfg['data']['shape'])
    original_batch = cfg['training']['batch_size']
    if condition.startswith('student_t_') or condition.startswith('piv_'):
        cfg['training']['batch_size'] = 256
    # The supplied PDF C.7 explicitly assigns the other vector benchmarks 4096,
    # except Weather 2048. Do not import the sparse-MSGM Weather 4096 exception.
    if generation:
        cfg['evaluation']['sample_seed'] = 0
    cfg['evaluation']['metric_seed_note'] = 'Common new seed-0 metric projections across A/B/C and t-Flow; historical projections unavailable.'
    cfg['provenance'].update(
        study_id=STUDY_ID, cached_input_only=True,
        historical_result_reuse=False,
        comparison_label='new_matched_realization' if generation else 'new_matched_runs_on_verified_existing_cache',
        original_prepared_config_sha256=canonical_sha(original),
        batch_resolution={'prepared_value': original_batch, 'selected_value': cfg['training']['batch_size'],
                          'evidence': 'Supplied PDF p34 C.7: Student-t/PIV256; Weather2048; other vectors4096; downstream p35 C.10'},
        original_split_spec=copy.deepcopy(cfg['data']['split']),
        generation=generation,
    )
    cfg['rafm_sampling_source'] = {'kind': 'radial_empirical_ecdf', 'fit_split': 'generator_train_only',
        'implementation': 'rafm/sources/radial_empirical.py:RadialEmpiricalSource(mode=ecdf)',
        'interpolation': 'torch.quantile linear, independent uniform radius draw',
        'log_radius_source': False,
        'evidence': ['rebuttal_experiments/configs/E1_studentt_d16.yaml',
                     'rebuttal_experiments/configs/E1_gaussian_d16.yaml',
                     'rebuttal_experiments/configs/E5_toy2d.yaml',
                     'rebuttal_experiments/configs/E6_dim16.yaml',
                     'rebuttal_experiments/configs/E8_df3.0.yaml',
                     'rebuttal_experiments/scripts/run_real.py',
                     'experiments/poc_audio/audio_flow.py',
                     'experiments/image_latents/dit/dit_train_sit.py'],
        'note': 'Explicit angular_rafm entries use empirical ECDF; rafm_oracle is a separate reference method.'}
    cfg['study'] = {'id': STUDY_ID, 'methods': ['A', 'B', 'C', 'tflow'],
                    'training_seeds_shared': True, 'new_A_rerun': True,
                    'B_C_additional_tuning_steps': 0,
                    'final_checkpoint_selection': 'prescribed_final_update_or_final_EMA',
                    'radius_stat_fit_split': 'generator_train_only',
                    'authorization': 'User authorized full t-Flow and A/B/C study on 2026-09-10, after small correctness runs'}
    cfg['experiment_launch_status'] = 'authorized_after_correctness_checks_and_verified_cache'
    issues = BLOCKED.get(condition, [])
    if generation:
        cfg['data']['input'] = {'path': str(data_root / condition / 'values.pt'), 'sha256': None}
    if not issues:
        cfg['data']['split'] = {**cfg['data']['split'], 'kind': 'indices',
                                'file': {'path': str(data_root / condition / 'split_indices.pt'), 'sha256': None}}
    cfg['protocol_status'] = 'blocked'
    cfg['blocking_issues'] = list(issues) if issues else ['shared_cache_not_materialized']
    cfg['shared_cache'] = {'directory': str(data_root / condition), 'manifest_path': str(data_root / condition / 'manifest.json'),
                           'manifest_sha256': None}
    return cfg


def make_plan(prepared_root=ROOT / 'configs/tflow/prepared', data_root=DEFAULT_DATA):
    originals = [json.loads(path.read_text()) for path in sorted(Path(prepared_root).glob('*.json'))]
    if len(originals) != 28:
        raise ValueError(f'Expected complete 28-condition suite; found {len(originals)}')
    drafts = [make_draft(original, Path(data_root)) for original in originals]
    return {'schema_version': 1, 'study_id': STUDY_ID, 'authorization_date': AUTHORIZED_DATE,
            'data_root': str(data_root), 'complete_scope_conditions': len(drafts),
            'materializable_conditions': [c['condition_id'] for c in drafts if c['condition_id'] not in BLOCKED],
            'blocked_conditions': BLOCKED,
            'new_synthetic_realizations': [c['condition_id'] for c in drafts if c['provenance']['generation']],
            'per_condition_final_runs': {'A': 3, 'B': 3, 'C': 3, 'tflow': 3},
            'tflow_additional_tuning': 'existing 0.55 full-run equivalents per condition, validation only',
            'drafts': drafts}


def tensor_record(tensor):
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes(order='C')
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256_contiguous_bytes': hashlib.sha256(raw).hexdigest(),
            'md5_contiguous_bytes': hashlib.md5(raw).hexdigest()}


def _generate(spec):
    import numpy as np
    import torch
    if spec['family'] == 'aniso_sweep':
        rng = np.random.default_rng(spec['sample_seed'])
        d, n = spec['dim'], spec['shape'][0]
        U, _ = np.linalg.qr(rng.standard_normal((d, d)))
        V, _ = np.linalg.qr(rng.standard_normal((d, d)))
        for kappa in spec['kappa_order']:
            singular = np.logspace(0, np.log10(kappa), d)
            A = U @ np.diag(singular) @ V.T
            z = rng.standard_normal((n, d))
            if kappa == spec['kappa']:
                return torch.tensor(z @ A.T, dtype=torch.float32), {'mixing_matrix': torch.from_numpy(A.copy()),
                        'left_orthogonal': torch.from_numpy(U.copy()), 'right_orthogonal': torch.from_numpy(V.copy())}
        raise ValueError('Unrecognized anisotropy kappa')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec['sample_seed'])
        if spec['family'] == 'student_t':
            from rafm.data.student_t import StudentT
            dataset = StudentT(dim=spec['dim'], df=spec['df'], n_samples=spec['shape'][0],
                               correlated=True, matrix_seed=spec['matrix_seed'], split_seed=spec['split_seed'])
        elif spec['family'] == 'gaussian_aniso':
            from rafm.data.gaussian_aniso import GaussianAniso
            dataset = GaussianAniso(dim=spec['dim'], n_samples=spec['shape'][0], correlated=True,
                                    matrix_seed=spec['matrix_seed'], split_seed=spec['split_seed'])
        elif spec['family'] == 'toy_radial_angular':
            from rafm.data.toy_radial_angular import ToyRadialAngular
            dataset = ToyRadialAngular(n_samples=spec['shape'][0], df=spec['df'], n_modes=spec['n_modes'],
                                      kappa=spec['kappa'], scale=spec['scale'], split_seed=spec['split_seed'])
        else:
            raise ValueError('Unrecognized synthetic family')
        matrices = {'mixing_matrix': dataset.A} if hasattr(dataset, 'A') else {}
        return dataset._data, matrices


def validate_existing_manifest(directory, identity_sha):
    manifest_path = Path(directory) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete' or manifest['preparation_identity_sha256'] != identity_sha:
        raise ValueError(f'Existing cache has different preparation identity: {directory}')
    for asset in manifest['assets']:
        path = Path(asset['path'])
        if not path.is_file() or sha256(path) != asset['sha256']:
            raise ValueError(f'Frozen shared cache missing or checksum mismatch: {path}')
    return manifest


def materialize(draft):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Data generation/deserialization requires a Slurm compute allocation')
    condition = draft['condition_id']
    if condition in BLOCKED:
        raise ValueError(f'Refusing unresolved input: {condition}: {BLOCKED[condition]}')
    import numpy as np
    import torch
    from experiments.tflow.data import load_data
    generation = draft['provenance']['generation']
    source_paths = ['tools/prepare_shared_study.py', 'experiments/tflow/data.py']
    source_paths += generation['source_files'] if generation else []
    source_hashes = {name: sha256(ROOT / name) for name in source_paths}
    identity = {'condition': condition, 'generation': generation, 'source_sha256': source_hashes,
                'original_config_sha256': draft['provenance']['original_prepared_config_sha256']}
    identity_sha = canonical_sha(identity)
    directory = Path(draft['shared_cache']['directory'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        manifest = validate_existing_manifest(directory, identity_sha)
        return finalize_config(draft, manifest)
    lock_path = directory.parent / ('.' + condition + '.lock')
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    tmp = Path(tempfile.mkdtemp(prefix='.' + condition + '.building-', dir=directory.parent))
    try:
        started = time.time()
        working = copy.deepcopy(draft)
        working['data']['split'] = copy.deepcopy(draft['provenance']['original_split_spec'])
        matrices = {}
        if generation:
            values, matrices = _generate(generation)
            if values.device.type != 'cpu' or values.dtype != torch.float32:
                raise ValueError('Replacement generation must produce CPU float32 tensors')
            torch.save(values.contiguous(), tmp / 'values.pt')
            working['data']['input'] = {'path': str(tmp / 'values.pt'), 'sha256': sha256(tmp / 'values.pt')}
        data = load_data(working)
        indices = data.indices
        torch.save(indices, tmp / 'split_indices.pt')
        torch.save(data.mean, tmp / 'preprocessing_mean.pt')
        if matrices:
            torch.save(matrices, tmp / 'generator_matrices.pt')
        split_records = {}
        for name in ('train', 'val', 'test'):
            values = data.split(name).contiguous()
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError(f'Nonfinite frozen {condition}/{name}')
            torch.save(values, tmp / f'{name}.pt')
            split_records[name] = {'values': tensor_record(values), 'indices': tensor_record(indices[name]),
                                   'external_test': name == 'test' and data.external_test is not None}
            labels = data.split_labels(name)
            if labels is not None:
                torch.save(labels.contiguous(), tmp / f'{name}_labels.pt')
                split_records[name]['labels'] = tensor_record(labels)
        historical = working['data'].get('historical_split_hashes')
        if historical:
            for name in ('train', 'test'):
                key = name + '_md5'
                if key in historical and split_records[name]['values']['md5_contiguous_bytes'] != historical[key]:
                    raise ValueError(f'Historical {condition}/{name} tensor hash mismatch')
        inputs = [working['data']['input']]
        for key in ('external_test', 'labels'):
            if key in working['data']:
                inputs.append(working['data'][key])
        if draft['kind'] == 'audio':
            classifier = draft['evaluation']['classifier']
            if sha256(classifier['path']) != classifier['sha256']:
                raise ValueError('Audio classifier checksum mismatch')
            inputs.append(classifier)
        assets = [{'path': str(directory / p.name), 'sha256': sha256(p), 'size_bytes': p.stat().st_size}
                  for p in sorted(tmp.iterdir()) if p.is_file()]
        for item in inputs:
            if Path(item['path']).parent != tmp:
                assets.append({'path': item['path'], 'sha256': item['sha256'], 'existing_input_unchanged': True})
        input_record = {'path': str(directory / 'values.pt'), 'sha256': working['data']['input']['sha256']} if generation else working['data']['input']
        manifest = {'schema_version': 1, 'study_id': STUDY_ID, 'condition_id': condition,
                    'status': 'complete', 'preparation_identity_sha256': identity_sha,
                    'preparation_identity': identity, 'input': input_record,
                    'split_file': {'path': str(directory / 'split_indices.pt'), 'sha256': sha256(tmp / 'split_indices.pt')},
                    'split_rule': draft['provenance']['original_split_spec'],
                    'split_rng': 'local CPU torch.Generator().manual_seed(0), torch.randperm; chronological cases torch.arange',
                    'splits': split_records, 'preprocessing_mean': tensor_record(data.mean),
                    'generation_matrices': {k: tensor_record(v) for k, v in matrices.items()},
                    'historical_split_hashes_verified': bool(historical),
                    'comparison_label': draft['provenance']['comparison_label'],
                    'historical_synthetic_identity_claim': False,
                    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                    'numpy': str(np.__version__), 'host': platform.node(), 'slurm_job_id': os.environ['SLURM_JOB_ID']},
                    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'source_worktree_hashes': source_hashes,
                    'duration_s': time.time() - started, 'assets': assets}
        write_json(tmp / 'manifest.json', manifest)
        os.rename(tmp, directory)
        validate_existing_manifest(directory, identity_sha)
        return finalize_config(draft, manifest)
    except BaseException as error:
        if tmp.exists():
            write_json(tmp / 'failure.json', {'condition': condition, 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def finalize_config(draft, manifest):
    cfg = copy.deepcopy(draft)
    cfg['data']['input'] = manifest['input']
    cfg['data']['split']['file'] = manifest['split_file']
    cfg['shared_cache']['manifest_sha256'] = sha256(cfg['shared_cache']['manifest_path'])
    cfg['protocol_status'] = 'resolved'
    cfg['blocking_issues'] = []
    cfg['provenance']['shared_preparation_identity_sha256'] = manifest['preparation_identity_sha256']
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan-only', action='store_true')
    action.add_argument('--materialize', action='store_true')
    parser.add_argument('--prepared-root', type=Path, default=ROOT / 'configs/tflow/prepared')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--condition', action='append', help='Materialize only these conditions; repeat for multiple')
    args = parser.parse_args()
    if not args.data_root.is_absolute():
        parser.error('--data-root must be absolute')
    plan = make_plan(args.prepared_root, args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'plan.json', {k: v for k, v in plan.items() if k != 'drafts'})
    for draft in plan['drafts']:
        write_json(args.output / 'drafts' / (draft['condition_id'] + '.json'), draft)
    if args.plan_only:
        print(json.dumps({k: v for k, v in plan.items() if k != 'drafts'}, indent=2))
        return
    selected = set(args.condition or plan['materializable_conditions'])
    unknown = selected - {c['condition_id'] for c in plan['drafts']}
    if unknown:
        parser.error('Unknown conditions: ' + ', '.join(sorted(unknown)))
    completed = []
    for draft in plan['drafts']:
        if draft['condition_id'] not in selected:
            continue
        print('Preparing ' + draft['condition_id'], flush=True)
        cfg = materialize(draft)
        write_json(args.output / 'prepared' / (cfg['condition_id'] + '.json'), cfg)
        completed.append(cfg['condition_id'])
        print('Verified and pinned ' + cfg['condition_id'], flush=True)
    write_json(args.output / 'materialization.json', {'status': 'complete_for_selected_conditions',
               'conditions': completed, 'blocked_conditions': BLOCKED, 'no_models_run': True})


if __name__ == '__main__':
    main()
