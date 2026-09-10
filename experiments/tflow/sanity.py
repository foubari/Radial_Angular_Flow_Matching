"""Approved-plan GPU sanity stage; NOT a paper-result run or tuning selection.

Uses a fixed tiny number of updates at the exact benchmark batch/precision and
validates sampling without inspecting test data. Requires experiment approval.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from baselines.tflow_core import TFlowSourceConfig, tflow_loss
from experiments.tflow.data import load_data
from experiments.tflow.run import (validate_config, compute_device, build_model,
                                   noise_callback, sample, hardware, write_json,
                                   implementation_sha256, json_hash, _finite_state,
                                   sanitize_metrics, dataset_manifest, finite_gradients)
from experiments.tflow.validation import source_candidates, selection_score
from rafm.utils.seeds import set_all_seeds


def check(cfg, root):
    """Run only the explicitly approved tiny sanity stage, with isolated output."""
    validate_config(cfg)
    output=(Path(root)/cfg['condition_id']).resolve()
    if output.exists():
        raise FileExistsError(f'Preserve previous sanity outcomes: {output}')
    output.mkdir(parents=True)
    losses=[]
    identity={'purpose':'sanity_only_not_paper_result','condition_id':cfg['condition_id'],
              'config_sha256':json_hash(cfg),'implementation_sha256':None,
              'test_data_used':False,'checkpoint_written':False}
    try:
        identity['implementation_sha256']=implementation_sha256(cfg)
        write_json(output/'config.json',cfg)
        device=compute_device()
        data=load_data(cfg,include_external_test=False)
        write_json(output/'dataset_manifest.json',dataset_manifest(data))
        set_all_seeds(46021)
        model=build_model(cfg,data.values.shape[1],device).train()
        settings=cfg['training']
        ema_rate=settings.get('ema')
        ema=copy.deepcopy(model).eval() if ema_rate is not None else None
        values=data.split('train').to(device)
        labels=data.split_labels('train')
        if labels is not None:labels=labels.to(device)
        choice=next(row for row in source_candidates(data.split('train')) if row['nu']==5 and row['scale_multiplier']==1)
        source=TFlowSourceConfig(choice['nu'],choice['scale'])
        identity['source']=vars(source)
        optimizer=getattr(torch.optim,settings['optimizer'])(model.parameters(),lr=settings['lr'],betas=tuple(settings['betas']),eps=settings['eps'],weight_decay=0)
        steps=4 if cfg['kind']!='vector' else 16
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        for step in range(1,steps+1):
            if settings['batch_rule']=='step_seeded':
                generator=torch.Generator().manual_seed(46021*1_000_003+step)
                index=torch.randint(len(values),(settings['batch_size'],),generator=generator).to(device)
            else:index=torch.randint(len(values),(settings['batch_size'],),device=device)
            y=None if labels is None else labels[index].clone()
            if cfg['kind']=='audio':
                drop=torch.rand(len(index),generator=torch.Generator().manual_seed(46021*7+step))
                classes=int(cfg['model'].get('num_classes',cfg['model'].get('ncls',10)))
                y[(drop<settings['class_dropout']).to(device)]=classes
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings['precision']=='bfloat16_autocast'):
                loss=tflow_loss(noise_callback(model,cfg,y),values[index],source,reduction='batch_mean')
            loss.backward()
            if not finite_gradients(model):
                raise FloatingPointError('Nonfinite sanity gradient')
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for average,current in zip(ema.parameters(),model.parameters()):
                        average.mul_(ema_rate).add_(current,alpha=1-ema_rate)
            losses.append(float(loss.detach()))
        _finite_state(model.state_dict(),'sanity.model')
        _finite_state(optimizer.state_dict(),'sanity.optimizer')
        if ema is not None:_finite_state(ema.state_dict(),'sanity.ema')
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        n=40 if cfg['kind']!='vector' else 128
        generated=sample(cfg,ema if ema is not None else model,source,n,seed=61717)
        score=selection_score(generated['samples'],data.split('val'))
        score,nonfinite=sanitize_metrics(score)
        if nonfinite:raise FloatingPointError(f'Nonfinite sanity diagnostic: {nonfinite}')
        if not bool(torch.isfinite(generated['samples']).all()):
            raise FloatingPointError('Nonfinite sanity samples')
        report={**identity,'status':'passed','steps':steps,'batch_size':settings['batch_size'],
                'precision':settings['precision'],'batch_rule':settings['batch_rule'],
                'ema':ema_rate,'evaluation_weights':'ema' if ema is not None else 'raw',
                'loss_reduction':'batch_mean','losses':losses,
                'finite_samples':bool(torch.isfinite(generated['samples']).all()),'n_generated':n,
                'nfe':generated['nfe'],'model_calls_total':generated['model_calls_total'],
                'n_batches':generated['n_batches'],'time_grid':generated['time_grid'].tolist(),
                'validation_diagnostic':score,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_memory_scope':'model, EMA, optimizer, tiny training and sampling; no downstream decoder/classifier',
                'measured_training_loop_s':elapsed,'measured_ms_per_step_including_first_step':elapsed*1000/steps,
                'timing_limitation':'Tiny uncompiled sanity loop; not a steady-state/full-training timing claim',
                'hardware':hardware()}
        write_json(output/'sanity.json',report)
        return report
    except Exception as error:
        write_json(output/'sanity.json',{**identity,'status':'failed','error_type':type(error).__name__,'error':str(error),
                   'losses':losses,'hardware':hardware()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='outputs_tflow/sanity')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    print(json.dumps(check(cfg,args.output),indent=2))


if __name__=='__main__':main()
