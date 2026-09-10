"""Strict byte verification of the recovered PIV/ImageNette release (stdlib only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'piv-imagenette-provenance-v1'
REQUIRED = {'manifest.json', 'manifest_addendum.json', 'AUDIT_REPORT.md',
            'audit_result.npz', 'imagenette_audit.py', 'verified_row_to_filename.csv',
            'dcae_real_centered.pt', 'radial_quantile_indices.npz',
            'piv_d32.pt', 'piv_d16.pt', 'provenance_code_and_logs.tar'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def api_verify(root, api, expected):
    require(api['tagName'] == expected, 'Wrong release tag')
    require(api['url'] == f'https://github.com/foubari/msgm-sparse-control/releases/tag/{expected}',
            'Wrong release repository/URL')
    assets = {}
    for asset in api['assets']:
        name = asset['name']
        require(Path(name).name == name and name not in ('.', '..') and name not in assets,
                'Unsafe or duplicate asset name')
        require(asset['state'] == 'uploaded', f'Asset is not uploaded: {name}')
        assets[name] = asset
    if expected != RELEASE:
        return assets
    require(REQUIRED <= assets.keys(), f'Missing required release assets: {sorted(REQUIRED-assets.keys())}')
    records = {}
    for name, asset in assets.items():
        p = root / name
        require(p.is_file(), f'Missing downloaded asset: {p}')
        actual = sha256(p)
        require(asset.get('digest') == 'sha256:' + actual, f'SHA256 MISMATCH: {name}')
        require(asset['size'] == p.stat().st_size, f'Size mismatch: {name}')
        records[name] = {'path': str(p.resolve()), 'sha256': actual,
                         'bytes': p.stat().st_size, 'status': 'verified'}
    return records


def verify(root):
    records = api_verify(root, load(root / 'github_release.json'), RELEASE)
    manifest, addendum = (load(root / name) for name in ('manifest.json', 'manifest_addendum.json'))
    require(manifest['release'] == RELEASE, 'Manifest tag mismatch')
    require(addendum['addendum_to'] == f'manifest.json ({RELEASE})', 'Wrong addendum target')
    for section in (manifest['uploaded_assets'], addendum['new_assets']):
        for name, item in section.items():
            require(name in records, f'Manifest references missing asset: {name}')
            require(item['sha256'] == records[name]['sha256'], f'MANIFEST SHA256 MISMATCH: {name}')
            require(item['bytes'] == records[name]['bytes'], f'Manifest size mismatch: {name}')
    referenced = manifest['referenced_data_v1_assets']
    data_api = api_verify(root, load(root / 'github_data_v1.json'), 'data-v1')
    draft = load(ROOT / 'configs/rafm_input_study/drafts/imagenette_dcae.json')
    data_records = {}
    for name, spec in [('dcae_latents_scaled.pt', draft['data']['input']),
                       ('dcae_labels.pt', draft['data']['labels'])]:
        require(name in data_api, f'Missing data-v1 asset: {name}')
        p = Path(spec['path'])
        require(p.is_file(), f'Missing existing data-v1 file: {p}')
        actual = sha256(p)
        require(actual == referenced[name]['sha256'] == spec['sha256'], f'DATA SHA256 MISMATCH: {name}')
        require(data_api[name].get('digest') == 'sha256:' + actual, f'Data API SHA256 mismatch: {name}')
        require(data_api[name]['size'] == p.stat().st_size, f'Data size mismatch: {name}')
        data_records[name] = {'path': str(p), 'sha256': actual, 'bytes': p.stat().st_size, 'status': 'verified'}
    # Read individual regular archive members; never execute downloaded code or
    # permit absolute paths, parent traversal, special files, or archive links.
    members = []
    with tarfile.open(root / 'provenance_code_and_logs.tar') as archive:
        names = set()
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and '..' not in name.parts and member.name not in names,
                    f'Unsafe/duplicate archive path: {member.name}')
            names.add(member.name)
            require(member.isfile() or member.isdir(), f'Archive links/special files refused: {member.name}')
            if member.isfile():
                require(member.size < 10 * 1024 * 1024, 'Unexpectedly large provenance code member')
                content = archive.extractfile(member).read()
                members.append({'name': member.name, 'bytes': len(content),
                                'sha256': hashlib.sha256(content).hexdigest()})
    require(any(x['name'].endswith('dit_train_sit.py') for x in members), 'Missing archived trainer source')
    return {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'release_directory': str(root.resolve()), 'assets': records, 'data_v1_assets': data_records,
            'archive_members': members, 'verification_script_sha256': sha256(__file__),
            'metadata_api_sha256': {name: sha256(root / name) for name in ('github_release.json', 'github_data_v1.json')},
            'limitations': ['Byte identity verified; tensor/split compatibility requires separate compute-node audit.',
                           'Checkout d3006dc8 is not a verified extraction/training commit.',
                           'FID-reference overlap is retained under the user-authorized existing protocol.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        report = verify(args.release_dir)
    except Exception as exc:
        report = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'verified_release_assets': len(report['assets']),
                      'verified_existing_data_assets': len(report['data_v1_assets']), 'report': str(args.output)}))


if __name__ == '__main__':
    main()
