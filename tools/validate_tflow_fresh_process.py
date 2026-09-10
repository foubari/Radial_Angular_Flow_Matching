"""Disposable check of the actual public trainer, resume and sampler entrypoints.

Run only through experiment_entrypoint.py on an allocated GPU. No source
selection, test-data inspection or final benchmark checkpoints are produced.
"""
import argparse
import json
from pathlib import Path
import traceback

import torch

from baselines.tflow_core import TFlowSourceConfig
from experiments.tflow import run
from experiments.tflow.data import load_data,sha256
from experiments.tflow.validation import source_candidates


def check(cfg,root):
    output=Path(root)/cfg['condition_id']
    if output.exists(): raise FileExistsError('Preserve previous fresh-process validation')
    output.mkdir(parents=True)
    initialized_before_metadata=torch.cuda.is_initialized()
    row={'status':'running','purpose':'disposable_public_trainer_validation',
        'condition_id':cfg['condition_id'],'test_data_used':False,'source_selection_performed':False,
        'config_sha256':run.json_hash(cfg),'implementation_sha256':run.implementation_sha256(cfg),
        'entrypoint_sha256':sha256(Path(__file__).with_name('experiment_entrypoint.py')),
        'validator_sha256':sha256(Path(__file__)),
        'hardware':run.hardware(),'gpu_initialized_before_training':initialized_before_metadata}
    try:
        if not row['gpu_initialized_before_training']: raise RuntimeError('GPU not initialized before trainer')
        data=load_data(cfg,include_external_test=False)
        choice=next(x for x in source_candidates(data.split('train')) if x['nu']==5 and x['scale_multiplier']==1)
        source=TFlowSourceConfig(choice['nu'],choice['scale'])
        seed=46021
        first=run.train(cfg,data,output/'resumed',seed,source,budget=4,stage='tuning')
        model,saved=run.load_trained(cfg,data,output/'resumed',seed,source,stage='tuning')
        if saved['step']!=4: raise RuntimeError('Wrong disposable checkpoint budget')
        generated=run.sample(cfg,model,source,40 if cfg['kind']=='audio' else 128,seed=61717)
        if not bool(torch.isfinite(generated['samples']).all()): raise FloatingPointError('Nonfinite public-trainer samples')
        if generated['nfe']!=cfg['evaluation']['model_evaluations']: raise RuntimeError('Network budget mismatch')
        del model, saved, generated
        resumed=run.train(cfg,data,output/'resumed',seed,source,budget=6,stage='tuning')
        direct=run.train(cfg,data,output/'uninterrupted',seed,source,budget=6,stage='tuning')
        a=torch.load(output/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
        b=torch.load(output/'uninterrupted/checkpoint.pt',map_location='cpu',weights_only=False)
        errors={}
        for name in ('model','ema'):
            if a[name] is None:
                if b[name] is not None: raise RuntimeError('EMA checkpoint mismatch')
                continue
            for key in a[name]:
                if not torch.equal(a[name][key],b[name][key]):
                    errors[name+'.'+key]=float((a[name][key]-b[name][key]).abs().max())
        if errors: raise RuntimeError('Resume differs from uninterrupted trajectory: '+str(errors))
        row.update(status='passed',initial_updates=4,resumed_updates=2,uninterrupted_updates=6,
            disposable_optimizer_updates_total=12,checkpoint_load=True,resume_bitwise_identical=True,
            finite_samples=True,nfe=cfg['evaluation']['model_evaluations'],source=vars(source),
            first_training=first,resumed_training=resumed,uninterrupted_training=direct)
        run.write_json(output/'check.json',row)
        return row
    except Exception as error:
        row.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        run.write_json(output/'check.json',row)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='outputs_tflow_full/v2/entrypoint_checks')
    args=parser.parse_args()
    print(json.dumps(check(json.loads(Path(args.config).read_text()),args.output),indent=2))
