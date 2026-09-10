"""Audit and freeze the recovered PIV32/ImageNette inputs without changing runtime.

Only standard-library modules are imported before the Slurm guard. This command
does not train a model, generate data, decode images, or alter historical inputs.
Both conditions are attempted independently; any failed check makes the command
fail after recording the unaffected condition's result.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/mnt/vast01/users/fouad.oubari/data/rafm_input_study/v1')
CONFIG_ROOT = ROOT / 'configs/rafm_input_study_recovered'
REPORT_ROOT = ROOT / 'docs/recovered_artifacts'
CONDITIONS = ('piv_d32', 'imagenette_dcae')
STUDY = 'rafm_input_parameterization_v1'
REQUIRED_ASSETS = ('manifest.json', 'manifest_addendum.json', 'AUDIT_REPORT.md',
    'piv_d32.pt', 'piv_d16.pt', 'dcae_real_centered.pt',
    'radial_quantile_indices.npz', 'audit_result.npz',
    'verified_row_to_filename.csv', 'imagenette_audit.py', 'provenance_code_and_logs.tar')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_release(release, receipt_path):
    """Require the parent's successful receipt and independently check manifests."""
    receipt = read_json(receipt_path)
    require(receipt.get('status') in ('verified', 'complete', 'passed'),
            'Release verification receipt does not record success')
    # The immutable manifest/addendum are the direct per-asset trust references.
    manifest = read_json(release / 'manifest.json')
    addendum = read_json(release / 'manifest_addendum.json')
    require(manifest.get('release') == 'piv-imagenette-provenance-v1', 'Wrong provenance release')
    references = dict(manifest['uploaded_assets'])
    references.update(addendum['new_assets'])
    records = {}
    for name in REQUIRED_ASSETS:
        path = release / name
        require(path.is_file(), f'Missing required release asset: {path}')
        record = file_record(path)
        if name in references:
            expected = references[name]
            require(record['sha256'] == expected['sha256'], f'Release asset checksum mismatch: {name}')
            require(record['size_bytes'] == expected['bytes'], f'Release asset length mismatch: {name}')
        records[name] = record
    # Pin every asset, including the manifest metadata, to the successful receipt.
    def collect_asset_records(value):
        found = []
        if isinstance(value, dict):
            if isinstance(value.get('sha256'), str):
                found.append(value)
            for child in value.values():
                found.extend(collect_asset_records(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(collect_asset_records(child))
        return found
    receipt_records = collect_asset_records(receipt)
    receipt_hashes = {row['sha256'] for row in receipt_records}
    for name, record in records.items():
        require(record['sha256'] in receipt_hashes,
                f'Successful receipt does not pin current release asset bytes: {name}')
    sources = {}
    with tarfile.open(release / 'provenance_code_and_logs.tar', 'r:') as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and '..' not in name.parts and not member.issym() and not member.islnk(),
                    f'Unsafe provenance archive member: {member.name}')
            if member.isdir():
                continue
            require(member.isfile(), f'Unsupported provenance archive entry: {member.name}')
            payload = archive.extractfile(member).read()
            sources[member.name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}
    for name in ('code_logs/dit_train_sit.py', 'code_logs/extract_dcae_latents.py', 'code_logs/prepare_piv.py'):
        require(name in sources, f'Missing archived source: {name}')
    return {'receipt': file_record(receipt_path), 'assets': records,
            'archived_members': sources, 'manifest': manifest, 'addendum': addendum}


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    payload = value.numpy().tobytes(order='C')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256_contiguous_bytes': hashlib.sha256(payload).hexdigest(),
            'md5_contiguous_bytes': hashlib.md5(payload).hexdigest()}


def exact_tensor(actual, expected, name, torch):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype,
            f'{name}: tensor shape/dtype mismatch')
    require(torch.equal(actual, expected), f'{name}: tensors are not bitwise equal')


def reference_digest(cfg):
    root = Path(cfg['evaluation']['real_reference_dir'])
    paths = sorted(root.glob('*.png'))
    require(len(paths) == cfg['evaluation']['real_n'] == 3925, 'FID reference must contain exactly 3925 PNGs')
    require(all(p.is_file() and p.suffix.lower() in ('.png', '.json', '.txt') for p in root.iterdir()),
            'FID reference must be a flat directory containing only PNGs and allowed metadata')
    digest = hashlib.sha256()
    for path in paths:
        digest.update((path.name + '\0' + sha256(path) + '\n').encode())
    require(digest.hexdigest() == cfg['evaluation']['expected_reference_sha256'], 'FID-reference bytes changed')
    for name, expected in cfg['evaluation']['decoder_files_sha256'].items():
        require(sha256(Path(cfg['evaluation']['decoder_path']) / name) == expected, f'Decoder asset mismatch: {name}')
    require(sha256(cfg['evaluation']['inception_weights_path']) == cfg['evaluation']['inception_weights_sha256'],
            'Inception weights mismatch')
    return {'path': str(root), 'count': len(paths), 'sha256': digest.hexdigest(),
            'manifest_digest_algorithm': 'SHA256 over sorted filename + NUL + file SHA256 + newline'}


def audit_piv(cfg, release, verification, torch):
    from experiments.tflow.data import load_pinned
    cfg['data']['input'] = {key: verification['assets']['piv_d32.pt'][key] for key in ('path', 'sha256')}
    values = load_pinned(cfg['data']['input'])
    require(isinstance(values, torch.Tensor) and values.dtype == torch.float32 and tuple(values.shape) == (998, 32),
            'Recovered PIV32 must be the original float32 (998,32) tensor')
    require(values.is_contiguous() and bool(torch.isfinite(values).all()), 'Recovered PIV32 must be finite and contiguous')
    d16 = load_pinned(verification['assets']['piv_d16.pt'])
    exact_tensor(d16, values[:, :16], 'Recovered PIV16 view equals PIV32 first 16 columns', torch)
    comparisons = {}
    for dim in (64, 256):
        other_cfg = read_json(ROOT / 'configs/rafm_input_study/prepared' / f'piv_d{dim}.json')
        other = load_pinned(other_cfg['data']['input'])
        equal = torch.equal(values, other[:, :32])
        require(not equal, f'Recovered PIV32 unexpectedly equals a PIV{dim} truncation')
        comparisons[f'not_truncated_piv{dim}'] = True
    return {'original_tensor': tensor_record(values), 'piv16_original_view_verified': True,
            **comparisons, 'preprocessing': 'unchanged full-dataset mean subtraction before CPU torch seed-0 split',
            'native_grid_provenance': verification['manifest']['uploaded_assets']['piv_d32.pt']}, None


def audit_image(cfg, release, verification, torch, np):
    from experiments.tflow.data import load_pinned
    raw = load_pinned(cfg['data']['input'])
    labels = load_pinned(cfg['data']['labels'])
    for filename, spec in (('dcae_latents_scaled.pt', cfg['data']['input']), ('dcae_labels.pt', cfg['data']['labels'])):
        require(spec['sha256'] == verification['manifest']['referenced_data_v1_assets'][filename]['sha256'],
                f'data-v1 identity differs from recovered provenance: {filename}')
    archive = load_pinned(verification['assets']['dcae_real_centered.pt'])
    require(isinstance(archive, dict) and {'data', 'mu', 'tr', 'va', 'te'} <= set(archive), 'Invalid saved centering/split dictionary')
    require(raw.dtype == torch.float32 and tuple(raw.shape) == (13394, 2048), 'Wrong original scaled latent tensor')
    require(labels.dtype == torch.int64 and tuple(labels.shape) == (13394,), 'Wrong original class-label tensor')
    indices = {name: archive[key] for name, key in (('train', 'tr'), ('val', 'va'), ('test', 'te'))}
    permutation = torch.randperm(13394, generator=torch.Generator().manual_seed(0))
    expected_indices = {'train': permutation[:8036], 'val': permutation[8036:10714], 'test': permutation[10714:]}
    for name in indices:
        exact_tensor(indices[name], expected_indices[name], 'Archived ' + name + ' indices versus original CPU torch seed0', torch)
    require(tuple(archive['mu'].shape) == (2048,), 'Saved centering mean must have 2048 coordinates')
    exact_tensor(archive['mu'], raw[indices['train']].mean(0), 'Archived mean versus original training-only centering', torch)
    exact_tensor(archive['data'], raw - archive['mu'], 'Archived centered data versus exact scaled input minus saved mean', torch)
    require(bool(torch.isfinite(archive['data']).all()) and bool(torch.isfinite(archive['mu']).all()), 'Nonfinite archived centered data')
    with np.load(release / 'radial_quantile_indices.npz', allow_pickle=False) as radial:
        require({'train_idx', 'val_idx', 'test_idx', 'labels', 'radii'} <= set(radial.files), 'Incomplete radial split archive')
        for name in indices:
            exact_tensor(torch.from_numpy(radial[name + '_idx']), indices[name], 'NPZ ' + name + ' versus saved indices', torch)
        exact_tensor(torch.from_numpy(radial['labels']), labels, 'NPZ labels versus data-v1 labels', torch)
        saved_radii = torch.from_numpy(radial['radii']).clone()
    actual_radii = archive['data'].norm(dim=1)
    require(saved_radii.shape == actual_radii.shape and bool(torch.isfinite(saved_radii).all()), 'Invalid archived radii')
    require(torch.allclose(saved_radii, actual_radii, atol=1e-5, rtol=1e-6), 'Diagnostic radii disagree with original centered values')
    with np.load(release / 'audit_result.npz', allow_pickle=False) as audit:
        require({'matched', 'dists', 'paths', 'derived_labels'} <= set(audit.files), 'Incomplete re-encoding audit')
        matched, distances = audit['matched'].copy(), audit['dists'].copy()
        paths, derived = audit['paths'].copy(), audit['derived_labels'].copy()
    require(matched.dtype.kind in 'iu' and matched.shape == (13394,) and np.array_equal(matched, np.arange(13394)),
            'Re-encoding audit is not an identity bijection of all 13394 rows')
    require(distances.shape == (13394,) and bool(np.isfinite(distances).all()) and bool((distances >= 0).all()),
            'Invalid re-encoding distances')
    require(float(distances.max()) < 0.00005, 'Re-encoding distances contradict the reported max L2=0.0000')
    require(paths.shape == (13394,) and len(set(paths.tolist())) == 13394, 'Re-encoding filenames are not a complete unique list')
    require(paths.tolist() == sorted(paths.tolist()), 'Re-encoding paths do not follow the archived sorted-glob row order')
    exact_tensor(torch.from_numpy(derived), labels, 'Re-encoded derived labels versus original labels', torch)
    with (release / 'verified_row_to_filename.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 13394, 'CSV must describe exactly 13394 latent rows')
    split_names = [''] * 13394
    for name, index in indices.items():
        for value in index.tolist():
            split_names[value] = name
    class_ids = sorted({PurePosixPath(str(path)).parent.name for path in paths})
    require(len(class_ids) == 10, 'Row map must contain ten classes')
    flags, filename_splits = [], {'train': 0, 'val': 0}
    max_radius_difference = 0.0
    for index, row in enumerate(rows):
        require(int(row['row']) == index and row['filename'] == str(paths[index]), f'CSV row/filename mismatch at {index}')
        filename = PurePosixPath(row['filename'])
        require(len(filename.parts) == 3 and filename.parts[0] in filename_splits and filename.suffix == '.JPEG',
                f'Unexpected ImageNette path at row {index}')
        filename_splits[filename.parts[0]] += 1
        label = int(labels[index])
        require(row['wordnet_id'] == filename.parent.name == class_ids[label] and int(row['class_index']) == label,
                f'CSV class assignment mismatch at row {index}')
        require(row['generator_split'] == split_names[index], f'CSV generator split mismatch at row {index}')
        require(row['in_fid_reference'] in ('0', '1'), f'Invalid CSV FID inclusion flag at row {index}')
        flags.append(int(row['in_fid_reference']))
        difference = abs(float(row['radius']) - float(saved_radii[index]))
        max_radius_difference = max(max_radius_difference, difference)
        require(difference <= 0.00006, f'CSV rounded radius mismatch at row {index}')
    flags = np.asarray(flags, dtype=bool)
    selected = np.sort(np.random.default_rng(1).permutation(13394)[:3925])
    require(np.array_equal(np.flatnonzero(flags), selected), 'CSV FID rows do not match the audit-described subset')
    overlap = {name: int(flags[index.numpy()].sum()) for name, index in indices.items()}
    require(overlap == {'train': 2339, 'val': 775, 'test': 811}, 'FID-reference split-overlap counts differ from addendum')
    reference = reference_digest(cfg)
    report = {'archived_centering_and_runtime_values_bitwise_equal': True,
        'archived_mean_equals_generator_training_mean_bitwise': True,
        'split_indices_archived_npz_and_original_cpu_rng_bitwise_equal': True,
        'split_counts': {name: len(index) for name, index in indices.items()},
        'class_labels_archived_npz_csv_and_data_v1_identical': True,
        'row_mapping': {'n_rows': len(rows), 'n_unique_filenames': len(set(paths.tolist())),
            'reencoded_identity_matches': int((matched == np.arange(13394)).sum()),
            'max_l2': float(distances.max()), 'mean_l2': float(distances.mean()),
            'encoding_was_run_here': False, 'source': verification['assets']['audit_result.npz'],
            'official_filename_split_counts': filename_splits},
        'diagnostic_radius_max_abs_difference': float((saved_radii - actual_radii).abs().max()),
        'csv_rounded_radius_max_abs_difference': max_radius_difference,
        'fid_reference': reference,
        'fid_reference_overlap': {'n_reference': 3925, 'generator_split_counts': overlap,
            'train_fraction': overlap['train'] / 3925,
            'definition': 'sorted(np.random.default_rng(1).permutation(13394)[:3925]) in verified latent-row order',
            'evidence': 'VM audit report/addendum and supplied CSV; CSV flags, seeded subset and archived split intersections independently checked here',
            'png_to_jpeg_identity_independently_reverified_here': False,
            'held_out_reference': False,
            'authorized_use': 'User explicitly authorized the existing reference protocol for this comparison; no reference or manuscript changes'},
        'archived_preprocessing_mean': tensor_record(archive['mu']),
        'archived_centered_data': tensor_record(archive['data'])}
    return report, indices


def materialize(condition, args, verification, torch, np):
    from experiments.tflow.data import load_data
    draft_path = ROOT / 'configs/rafm_input_study/drafts' / (condition + '.json')
    draft = read_json(draft_path)
    cfg = copy.deepcopy(draft)
    directory = args.data_root / condition
    source_paths = ('tools/prepare_recovered_study.py', 'experiments/tflow/data.py')
    sources = {name: sha256(ROOT / name) for name in source_paths}
    identity = {'condition': condition, 'generation': None, 'source_sha256': sources,
        'original_draft_file_sha256': sha256(draft_path), 'release_assets': verification['assets'],
        'release_verification_receipt': verification['receipt'],
        'existing_input': draft['data']['input'], 'existing_labels': draft['data'].get('labels')}
    identity_sha = canonical_sha(identity)
    if directory.exists():
        manifest = read_json(directory / 'manifest.json')
        require(manifest['status'] == 'complete' and manifest['preparation_identity_sha256'] == identity_sha,
                f'Existing recovered cache has another identity: {directory}')
        for asset in manifest['assets']:
            require(sha256(asset['path']) == asset['sha256'], f'Existing cache asset changed: {asset["path"]}')
        prepared = manifest['prepared_config']
        prepared['shared_cache']['manifest_sha256'] = sha256(directory / 'manifest.json')
        return prepared, manifest['verification'], True
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / ('.' + condition + '.recovery.lock')
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    temporary = Path(tempfile.mkdtemp(prefix='.' + condition + '.recovering-', dir=directory.parent))
    began = time.perf_counter()
    try:
        if condition == 'piv_d32':
            audit, indices = audit_piv(cfg, args.release, verification, torch)
        else:
            audit, indices = audit_image(cfg, args.release, verification, torch, np)
        if indices is None:
            # Original PIV loader performs precisely the established full-data
            # recenter and local CPU seed-zero permutation before freezing indices.
            data = load_data(cfg)
            indices = data.indices
        torch.save(indices, temporary / 'split_indices.pt')
        cfg['data']['split'] = {'kind': 'indices', 'seed': 0,
            **{'n_' + name: len(index) for name, index in indices.items()},
            'file': {'path': str(temporary / 'split_indices.pt'), 'sha256': sha256(temporary / 'split_indices.pt')}}
        data = load_data(cfg)
        if condition == 'imagenette_dcae':
            require(tensor_record(data.values) == audit['archived_centered_data'], 'Unmodified loader centered-data fingerprint differs from archive')
            require(tensor_record(data.mean) == audit['archived_preprocessing_mean'], 'Unmodified loader decode mean differs from archive')
        torch.save(data.mean, temporary / 'preprocessing_mean.pt')
        splits = {}
        for name in ('train', 'val', 'test'):
            values = data.split(name).contiguous()
            require(bool(torch.isfinite(values).all()), f'Nonfinite recovered {condition}/{name} values')
            torch.save(values, temporary / (name + '.pt'))
            splits[name] = {'values': tensor_record(values), 'indices': tensor_record(data.indices[name]), 'external_test': False}
            labels = data.split_labels(name)
            if labels is not None:
                torch.save(labels.contiguous(), temporary / (name + '_labels.pt'))
                splits[name]['labels'] = tensor_record(labels)
        cfg['data']['split']['file']['path'] = str(directory / 'split_indices.pt')
        cfg['shared_cache'] = {'directory': str(directory), 'manifest_path': str(directory / 'manifest.json'), 'manifest_sha256': None}
        cfg['protocol_status'] = 'resolved'
        cfg['blocking_issues'] = []
        cfg['provenance']['shared_preparation_identity_sha256'] = identity_sha
        cfg['provenance']['recovered_artifacts'] = {
            'release': 'foubari/msgm-sparse-control/piv-imagenette-provenance-v1',
            'receipt': verification['receipt'], 'assets': verification['assets'],
            'checkout_holding_artifacts': verification['manifest']['checkout_HEAD'],
            'verified_extraction_or_training_commit': None,
            'commit_caveat': verification['manifest']['commit_caveat'],
            'archived_members': verification['archived_members'],
            'numerical_verification': audit,
            'authorization': 'User authorized recovery, checks and remaining A/B/C runs on 2026-09-10; preserve existing image-reference overlap protocol'}
        for key in ('model', 'training', 'evaluation', 'sampler', 'seeds', 'rafm_sampling_source', 'study'):
            require(cfg[key] == draft[key], f'Preparation altered agreed {key} settings')
        assets = [{'path': str(directory / p.name), 'sha256': sha256(p), 'size_bytes': p.stat().st_size}
                  for p in sorted(temporary.iterdir()) if p.is_file()]
        assets.extend(verification['assets'].values())
        assets.append(verification['receipt'])
        assets.append(file_record(cfg['data']['input']['path']))
        if 'labels' in cfg['data']:
            assets.append(file_record(cfg['data']['labels']['path']))
        manifest = {'schema_version': 1, 'study_id': STUDY, 'condition_id': condition,
            'status': 'complete', 'preparation_identity_sha256': identity_sha, 'preparation_identity': identity,
            'input': cfg['data']['input'], 'split_file': cfg['data']['split']['file'],
            'split_rule': cfg['data']['split'],
            'split_rng': 'Archived ImageNette indices verified against local CPU torch.Generator().manual_seed(0), torch.randperm; original PIV local CPU seed0 rule',
            'splits': splits, 'preprocessing_mean': tensor_record(data.mean),
            'comparison_label': cfg['provenance']['comparison_label'], 'historical_synthetic_identity_claim': False,
            'generation_matrices': {}, 'historical_split_hashes_verified': condition == 'imagenette_dcae',
            'verification': audit, 'prepared_config': cfg,
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'host': platform.node(), 'slurm_job_id': os.environ['SLURM_JOB_ID']},
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_worktree_hashes': sources, 'duration_s': time.perf_counter() - began, 'assets': assets}
        write_json(temporary / 'manifest.json', manifest)
        os.rename(temporary, directory)
        cfg['shared_cache']['manifest_sha256'] = sha256(directory / 'manifest.json')
        # Verify final immutable paths with precisely the same loader used by A/B/C.
        final = load_data(cfg)
        for name in ('train', 'val', 'test'):
            require(tensor_record(final.split(name)) == splits[name]['values'], f'Published cache {name} values changed')
        return cfg, audit, False
    except BaseException as error:
        if temporary.exists():
            write_json(temporary / 'failure.json', {'condition_id': condition, 'error_type': type(error).__name__,
                'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--verification-receipt', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT)
    parser.add_argument('--report-root', type=Path, default=REPORT_ROOT)
    parser.add_argument('--condition', action='append', choices=CONDITIONS)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
        parser.error('Recovered tensor preparation requires a Slurm compute allocation; no login-node tensor work')
    for name in ('release', 'verification_receipt', 'data_root', 'config_root', 'report_root'):
        require(getattr(args, name).is_absolute(), name + ' must be an absolute path')
    verification = verify_release(args.release, args.verification_receipt)
    import numpy as np
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    rows, configs = [], []
    for condition in args.condition or CONDITIONS:
        try:
            cfg, audit, reused = materialize(condition, args, verification, torch, np)
            path = args.config_root / 'prepared' / (condition + '.json')
            if path.exists():
                require(read_json(path) == cfg, f'Existing recovered prepared config differs: {path}')
            else:
                write_json(path, cfg)
            configs.append(condition)
            row = {'condition_id': condition, 'status': 'verified', 'reused_existing_cache': reused,
                'config': file_record(path), 'verification': audit,
                'shared_cache_manifest': file_record(cfg['shared_cache']['manifest_path'])}
        except Exception as error:
            row = {'condition_id': condition, 'status': 'failed', 'error_type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc()}
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    report = {'schema_version': 1, 'study_id': STUDY,
        'status': 'complete' if all(row['status'] == 'verified' for row in rows) else 'failed',
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'no_models_run': True,
        'conditions': configs, 'rows': rows, 'release_verification': verification['receipt'],
        'source': file_record(__file__), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'host': platform.node()}
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    write_json(args.report_root / ('preparation_' + timestamp + '.json'), report)
    write_json(args.config_root / 'materialization.json', report)
    raise SystemExit(0 if report['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
