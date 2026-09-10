"""Verify saved sample tensors on the allocated compute node after evaluation."""
import argparse
import json
from pathlib import Path
import traceback

import torch

from experiments.tflow.run import compute_device, write_json, hardware
from experiments.tflow.data import sha256


def audit(result_path):
    compute_device()  # Keep tensor/data work off login nodes.
    result_path=Path(result_path)
    result=json.loads(result_path.read_text())
    if result['status']!='complete':
        raise ValueError('Sample audit requires a complete per-seed evaluation')
    cfg=result['config']; artifact=result['sample_artifact']; path=Path(artifact['path'])
    digest=sha256(path)
    if digest!=artifact['sha256']:
        raise ValueError('Saved sample SHA-256 mismatch; aborting')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    samples=saved['samples']; labels=saved.get('labels')
    expected=cfg['evaluation']
    counts=None if labels is None else torch.bincount(labels,minlength=10).tolist()
    finite=torch.isfinite(samples).all(dim=1)
    report={'schema_version':1,'status':'passed','sample_sha256':digest,
        'result_sha256':sha256(result_path),'n_samples':len(samples),
        'dimension':samples.shape[1],'class_counts':counts,
        'nonfinite_rows':int((~finite).sum()),'nan_rows':int(torch.isnan(samples).any(dim=1).sum()),
        'inf_rows':int(torch.isinf(samples).any(dim=1).sum()),
        'zero_radius_rows':int((samples.norm(dim=1)==0).sum()),
        'nfe':saved['nfe'],'model_calls_total':saved['model_calls_total'],
        'n_batches':saved['n_batches'],'configured_sample_seed':expected['sample_seed'],
        'seed_evidence':'Configuration and frozen sampling implementation; cannot infer RNG seed from output tensor',
        'hardware':hardware()}
    errors=[]
    if samples.shape!=(expected['n_samples'],cfg['data']['shape'][1]): errors.append('sample shape mismatch')
    if report['nonfinite_rows']: errors.append('nonfinite generated samples')
    if saved['nfe']!=expected['model_evaluations']: errors.append('network evaluation budget mismatch')
    if saved['model_calls_total']!=saved['nfe']*saved['n_batches']: errors.append('total network calls mismatch')
    if cfg['kind']!='vector' and counts!=[expected['n_samples']//10]*10: errors.append('class balance mismatch')
    if cfg['kind']=='vector' and labels is not None: errors.append('unexpected vector conditioning labels')
    report['errors']=errors
    if errors: report['status']='failed'
    write_json(result_path.with_name('sample_audit.json'),report)
    if errors: raise ValueError('; '.join(errors))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(audit(args.result),indent=2))
    except Exception as error:
        target=Path(args.result).with_name('sample_audit.json')
        if not target.exists():
            write_json(target,{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        raise
