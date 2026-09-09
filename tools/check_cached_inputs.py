"""Read-only cached-input verification for an allocated CPU node. No models."""
import hashlib
import json
from pathlib import Path
import platform

import torch

from experiments.tflow.data import load_data


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'configs/tflow/suite_manifest.json').read_text())
    results={"kind":"input_verification_only", "host":platform.node(), "torch":torch.__version__, "cases":[]}
    for case in manifest['benchmarks']:
        if not case['resolved_protocol']:
            continue
        cfg=json.loads((root/'configs/tflow/prepared'/f"{case['id']}.json").read_text())
        data=load_data(cfg)
        row={"id":case['id'],"shape":list(data.values.shape),"counts":{key:len(data.split(key)) for key in ['train','val','test']},
             "finite":bool(torch.isfinite(data.values).all()),"input_sha256":cfg['data']['input']['sha256']}
        expected=case.get('historical_split_hashes')
        if expected:
            row['historical_tensor_identity']={}
            for split in ['train','test']:
                actual=hashlib.md5(data.split(split).contiguous().numpy().tobytes()).hexdigest()
                row['historical_tensor_identity'][split]={"actual_md5":actual,"expected_md5":expected[f'{split}_md5'],"matches":actual==expected[f'{split}_md5']}
        results['cases'].append(row)
        del data
    # Verify the exact existing reference manifest as bytes. Never rebuild PNGs.
    ref=Path(manifest['inputs']['dcae_reference']['path'])
    committed=root.parent/'msgm-sparse-control'/'real_ref_manifest.sha256'
    desired={name:digest for line in committed.read_text().splitlines() if line and not line.startswith('#') for digest,name in [line.split('  ',1)]}
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ref.glob('*.png')}
    results['fid_reference']={"expected_count":3925,"actual_count":len(actual),"manifest_count":len(desired),
                              "byte_identical":actual==desired and len(desired)==3925,
                              "manifest_sha256":hashlib.sha256(committed.read_bytes()).hexdigest()}
    path=root/'docs/artifact_audit/cached_input_checks.json'
    path.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))
    failures=[row['id'] for row in results['cases'] if not row['finite'] or any(not v['matches'] for v in row.get('historical_tensor_identity',{}).values())]
    if failures or not results['fid_reference']['byte_identical']:
        raise SystemExit(f'Input verification mismatch: {failures}, FID={results["fid_reference"]["byte_identical"]}')


if __name__=='__main__':main()
