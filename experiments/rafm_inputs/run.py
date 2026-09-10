"""A/B/C RAFM-Ang study: original objective, coupling and ambient RK4 sampler.

All model computation is restricted to single-GPU Slurm allocations. Data and
protocol come from the same pinned configuration used by the t-Flow comparison.
Sanity is a separate disposable run; final jobs always start from their own seed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import torch

from baselines.rafm_input_parameterization import (
    ARMS, build_input_model, fit_radius_statistics, angular_target,
    ambient_velocity_callback, radius_drift,
)
from experiments.tflow import run as common
from experiments.tflow.data import load_data, load_pinned, sha256
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.utils.seeds import set_all_seeds
from rafm.utils.sphere import uniform_on_sphere

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = Path('/mnt/vast01/users/fouad.oubari/data/rafm_input_study/generated')


def implementation(cfg):
    base = common.implementation_manifest(cfg)
    names = ['experiments/rafm_inputs/run.py', 'baselines/rafm_input_parameterization.py',
             'rafm/flow_matching/loss.py', 'rafm/flow_matching/sampler.py',
             'rafm/paths/spherical_geodesic.py', 'rafm/utils/sphere.py',
             'rafm/sources/radial_empirical.py']
    base['files'].update({name: sha256(ROOT/name) for name in names})
    return base


def validate(cfg, arm):
    common.validate_config(cfg)
    if arm not in ARMS:
        raise ValueError('Arm must be A, B, or C')
    if cfg['evaluation']['model_evaluations'] % 4:
        raise ValueError('Original RK4 requires a multiple of four network calls')
    source = cfg.get('rafm_sampling_source', {'kind':'radial_empirical_ecdf'})
    if source['kind'] != 'radial_empirical_ecdf':
        raise ValueError('Unreviewed RAFM-Ang radial source; do not substitute a law')


def statistics(data):
    return fit_radius_statistics(data.split('train'),
        training_data_sha256=common.tensor_fingerprint(data.split('train'))['sha256'],
        training_indices_sha256=common.tensor_fingerprint(data.indices['train'])['sha256'])


def identity(cfg, arm, seed, stats, stage):
    return {'config':cfg, 'arm':arm, 'seed':seed, 'stage':stage,
            'radius_statistics':stats.to_dict(),
            'implementation_sha256':common.json_hash(implementation(cfg))}


def training_pair(values, kind, path):
    """Retain each baseline's exact source/time RNG order and arithmetic."""
    n, dim = values.shape
    if kind == 'vector':
        times = torch.rand(n, device=values.device)
        initial = values.norm(dim=-1,keepdim=True) * uniform_on_sphere(n,dim,device=values.device)
    else:
        radius = values.norm(dim=1,keepdim=True)
        normal = torch.randn(n,dim,device=values.device)
        initial = radius * normal / normal.norm(dim=1,keepdim=True)
        times = torch.rand(n,device=values.device)
    state = path.sample_path(initial,values,times)
    velocity = path.conditional_vector_field(initial,values,times)
    target = angular_target(state,velocity)
    if not bool(torch.stack([torch.isfinite(z).all() for z in (state,target)]).all().item()):
        raise FloatingPointError('Nonfinite spherical path or angular target')
    return state,times,target


def batch(data_values,data_labels,settings,kind,seed,step):
    if settings['batch_rule']=='step_seeded':
        g=torch.Generator().manual_seed(seed*1_000_003+step)
        index=torch.randint(len(data_values),(settings['batch_size'],),generator=g).to(data_values.device)
    else:
        index=torch.randint(len(data_values),(settings['batch_size'],),device=data_values.device)
    labels=None if data_labels is None else data_labels[index].clone()
    if kind=='audio':
        drop=torch.rand(len(index),generator=torch.Generator().manual_seed(seed*7+step))
        labels[(drop<settings['class_dropout']).to(data_values.device)]=10
    return data_values[index],labels


def train(cfg,data,output,arm,seed,*,stage='final',budget=None):
    validate(cfg,arm)
    device=common.compute_device()
    full=cfg['training']['steps']
    budget=full if budget is None else budget
    if stage not in ('final','sanity') or not 0<budget<=full or (stage=='final' and budget!=full):
        raise ValueError('Final runs require the complete unchanged training budget')
    if stage=='final' and seed not in cfg['seeds']:
        raise ValueError('Final seed does not match original seeds')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    stats=statistics(data); signature=identity(cfg,arm,seed,stats,stage); run_hash=common.json_hash(signature)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=signature:
        raise ValueError('Run/config/source/radius-statistics mismatch; existing outputs preserved')
    set_all_seeds(seed)
    model=build_input_model(cfg['model'],data.values.shape[1],arm,stats).to(device)
    settings=cfg['training']; ema_rate=settings.get('ema')
    ema=copy.deepcopy(model).eval() if ema_rate is not None else None
    if ema is not None:
        for p in ema.parameters(): p.requires_grad_(False)
    optimizer=getattr(torch.optim,settings['optimizer'])(model.parameters(),lr=settings['lr'],
        betas=tuple(settings['betas']),eps=settings['eps'],weight_decay=0)
    values=data.split('train').to(device); labels=data.split_labels('train')
    if labels is not None: labels=labels.to(device)
    # The original vector Trainer resets RNG immediately before its loop.
    if cfg['kind']=='vector': set_all_seeds(seed)
    checkpoint=output/'checkpoint.pt'; start=0; elapsed_before=0.; previous_peak=None; last_loss=None
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['run_sha256']!=run_hash or saved['stage']!=stage:
            raise ValueError('Checkpoint identity mismatch')
        for key in ('model','ema','optimizer'): common._finite_state(saved[key],key)
        model.load_checkpoint_payload({'metadata':saved['model_metadata'],'state_dict':saved['model']})
        if ema is not None: ema.load_checkpoint_payload({'metadata':saved['model_metadata_ema'],'state_dict':saved['ema']})
        optimizer.load_state_dict(saved['optimizer'])
        start=saved['step']; elapsed_before=saved['train_time_s']; previous_peak=saved.get('peak_memory')
        last_loss=saved['last_loss']; common.restore_rng(saved['rng'])
        if start>budget: raise ValueError('Checkpoint exceeds requested budget')
    common.write_json(output/'config.json',signature)
    common.write_json(output/'implementation.json',implementation(cfg))
    common.write_json(output/'dataset.json',common.dataset_manifest(data))
    common.write_json(output/'parameter_counts.json',model.parameter_report())
    common.write_json(output/'radius_statistics.json',stats.to_dict())
    if start==budget:
        stats_path=output/'training_stats.json'
        if stats_path.exists(): return json.loads(stats_path.read_text())
        result={'run_sha256':run_hash,'training_step':budget,'total_train_time_s':elapsed_before,
            'final_loss':last_loss,'parameters':model.parameter_report(),'hardware':common.hardware(),
            'peak_memory':previous_peak,'timing_kind':'measured','timing_scope':'completed checkpoint; no additional updates'}
        common.write_json(stats_path,result)
        return result
    model.train()
    train_model=torch.compile(model) if settings.get('compile',False) else model
    path=SphericalGeodesicPath(); torch.cuda.reset_peak_memory_stats(device)
    common._synchronize(device); begin=time.perf_counter()
    gradient_observed=False; radius_gradient_observed=False
    for step in range(start+1,budget+1):
        x1,y=batch(values,labels,settings,cfg['kind'],seed,step)
        xt,t,target=training_pair(x1,cfg['kind'],path)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings['precision']=='bfloat16_autocast'):
            prediction=train_model(xt,t,y).float()
            loss=((prediction-target)**2).sum(dim=-1).mean()
        if not bool(torch.isfinite(loss).item()): raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        if not common.finite_gradients(model): raise FloatingPointError(f'Nonfinite gradient at step {step}')
        if stage=='sanity':
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            gradient_observed |= bool(torch.stack([g.abs().max() for g in gradients]).max().item()>0)
            extra=[p.grad for name,p in model.named_parameters() if 'radius_embed' in name and p.grad is not None]
            if arm=='B' and cfg['kind']=='vector':
                added_column=model.backbone.net[0].weight.grad[:,data.values.shape[1]]
                extra.append(added_column)
            if extra: radius_gradient_observed |= bool(torch.stack([g.abs().max() for g in extra]).max().item()>0)
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.parameters()): average.mul_(ema_rate).add_(current,alpha=1-ema_rate)
        last_loss=float(loss.detach())
        if step%settings['log_every']==0 or step==budget:
            common._synchronize(device)
            record={'arm':arm,'seed':seed,'step':step,'loss':last_loss,'elapsed_s':elapsed_before+time.perf_counter()-begin}
            with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record),flush=True)
        if step%settings['checkpoint_every']==0 or step==budget:
            common._synchronize(device)
            saved={'run_sha256':run_hash,'model':model.state_dict(),'ema':None if ema is None else ema.state_dict(),
                'optimizer':optimizer.state_dict(),'model_metadata':model.checkpoint_metadata(),
                'model_metadata_ema':None if ema is None else ema.checkpoint_metadata(),'step':step,'train_time_s':elapsed_before+time.perf_counter()-begin,
                'rng':common.rng_state(),'stage':stage,'last_loss':last_loss,
                'peak_memory':common.peak_memory(device,previous_peak)}
            for key in ('model','ema','optimizer'): common._finite_state(saved[key],key)
            tmp=checkpoint.with_suffix('.tmp'); torch.save(saved,tmp); tmp.replace(checkpoint)
    common._synchronize(device)
    result={'run_sha256':run_hash,'training_step':budget,'total_train_time_s':elapsed_before+time.perf_counter()-begin,
        'final_loss':last_loss,'parameters':model.parameter_report(),'hardware':common.hardware(),
        'peak_memory':common.peak_memory(device,previous_peak),'timing_kind':'measured',
        'timing_scope':'training loop including compile, finite checks, logging and checkpoints',
        'gradient_observed':gradient_observed if stage=='sanity' else None,
        'radius_embedding_gradient_observed':radius_gradient_observed if stage=='sanity' else None}
    result['backend']={'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'note':'Shared seed helper for every new arm; historical downstream scripts did not explicitly force deterministic cuDNN/MIOpen'}
    common.write_json(output/'training_stats.json',result)
    return result


class _CountModel:
    def __init__(self,model): self.model,self.calls=model,0
    def __call__(self,x,t): self.calls+=1; return self.model(x,t)


@torch.no_grad()
def sample(cfg,model,data,n,seed=None):
    device=next(model.parameters()).device
    if seed is not None: set_all_seeds(seed)
    model.eval(); dim=data.values.shape[1]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    source=RadialEmpiricalSource(mode='ecdf').fit(data.split('train'))
    labels=None if cfg['kind']=='vector' else torch.arange(10,device=device).repeat_interleave(n//10)
    if labels is not None and n%10: raise ValueError('Conditional samples must be class balanced')
    common._synchronize(device); begin=time.perf_counter()
    if cfg['kind']=='vector':
        initial=source.sample(n,dim,device=device)
    else:
        radii=source.sample(n,dim).norm(dim=1,keepdim=True).to(device)
        normal=torch.randn(n,dim,device=device); initial=radii*normal/normal.norm(dim=1,keepdim=True)
    if not bool(torch.isfinite(initial).all().item()): raise FloatingPointError('Nonfinite source sample')
    outputs=[]; calls=0; nfe=cfg['evaluation']['model_evaluations']; nsteps=nfe//4
    size=cfg['evaluation']['sample_batch_size']
    for start in range(0,n,size):
        x=initial[start:start+size]
        if cfg['kind']=='vector':
            # Execute the original RK4 and original radius/projection policy.
            from rafm.flow_matching.sampler import Sampler
            counted=_CountModel(model)
            sampler=Sampler(counted,source,{'device':device,'angular':True,'path':'spherical_geodesic'})
            x=sampler._rk4(x,nsteps)
            count=counted.calls
        else:
            y=labels[start:start+size]; velocity=ambient_velocity_callback(model,y,radius_floor=0. if cfg['kind']=='audio' else 1e-8)
            count=0; dt=1./nsteps
            def v(xx,tt):
                nonlocal count
                count+=1
                return velocity(xx,torch.full((len(xx),),tt,device=device))
            # Exact original downstream RK4 arithmetic and time construction.
            for i in range(nsteps):
                t0=i*dt
                k1=v(x,t0); k2=v(x+dt/2*k1,t0+dt/2); k3=v(x+dt/2*k2,t0+dt/2); k4=v(x+dt*k3,t0+dt)
                x=x+dt/6*(k1+2*k2+2*k3+k4)
        if count!=nfe: raise RuntimeError(f'Wrong actual network count: {count} != {nfe}')
        if not bool(torch.isfinite(x).all().item()):
            raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')
        outputs.append(x.cpu()); calls+=count
    common._synchronize(device); elapsed=time.perf_counter()-begin
    result=torch.cat(outputs)
    return {'samples':result,'initial_radii':initial.norm(dim=1).cpu(),'labels':None if labels is None else labels.cpu(),
        'sample_time_s':elapsed,'nfe':nfe,'n_batches':len(outputs),'model_calls_total':calls,
        'n_samples':len(result),'sample_seed':seed,
        'class_counts':None if labels is None else torch.bincount(labels,minlength=10).cpu().tolist(),
        'peak_memory':common.peak_memory(device),
        'radius_drift':radius_drift(initial.cpu(),result),'solver':'original_ambient_rk4',
        'state_renormalization':False,'projection_r_min':1e-3,
        'radius_multiplier_floor':0. if cfg['kind']=='audio' else 1e-8,
        'timing_scope':'source draw, original RK4, finite checks and CPU transfer; excludes metrics'}


def load_model(cfg,data,output,arm,seed,stage):
    stats=statistics(data); expected=common.json_hash(identity(cfg,arm,seed,stats,stage))
    saved=torch.load(Path(output)/'checkpoint.pt',map_location='cpu',weights_only=False)
    if saved['run_sha256']!=expected or saved['stage']!=stage: raise ValueError('Checkpoint mismatch')
    if stage=='final' and saved['step']!=cfg['training']['steps']: raise ValueError('Incomplete final checkpoint')
    device=common.compute_device(); model=build_input_model(cfg['model'],data.values.shape[1],arm,stats).to(device).eval()
    state=saved['ema'] if saved['ema'] is not None else saved['model']
    common._finite_state(state,'evaluation_weights')
    # EMA metadata records requires_grad=False; inference uses the same setting.
    if saved['ema'] is not None:
        for parameter in model.parameters(): parameter.requires_grad_(False)
    model.load_checkpoint_payload({'metadata':saved['model_metadata_ema'] if saved['ema'] is not None else saved['model_metadata'],'state_dict':state})
    common.restore_rng(saved['rng'])
    return model,saved


def quality(cfg,data,generated,model,output,sample_path):
    samples=generated['samples']; details={}
    set_all_seeds(cfg['evaluation']['metric_seed'])
    if cfg['kind']=='vector':
        metrics=common.vector_metrics(samples,data.split('test'))
    elif cfg['kind']=='audio':
        from experiments.poc_audio.audio_classifier import Clf
        from experiments.poc_audio.audio_empirical_gain import summarize,flat_metrics
        device=next(model.parameters()).device; classifier=Clf(ncls=10).to(device).eval()
        classifier.load_state_dict(load_pinned(cfg['evaluation']['classifier']),strict=True)
        with torch.no_grad():
            norms=samples.norm(dim=1,keepdim=True)
            if bool((norms<=0).any()): raise FloatingPointError('Nonpositive audio radius')
            logits=classifier((samples/norms).reshape(-1,2,129,63).to(device)).cpu()
        if not bool(torch.isfinite(logits).all()): raise FloatingPointError('Nonfinite classifier logits')
        summary=summarize(samples,logits,generated['labels'],data.external_gains.numpy())
        metrics=flat_metrics(summary,full_precision=True)
        details={'audio_summary':summary,'classifier_batch_size':len(samples),'classifier_sha256':cfg['evaluation']['classifier']['sha256']}
        logits_path=sample_path.parent/'classifier_logits.pt'; torch.save(logits,logits_path)
        details['classifier_logits']={'path':str(logits_path),'sha256':sha256(logits_path)}
    else:
        from baselines.tflow_downstream import evaluate_image
        settings=dict(cfg['evaluation']); settings['device']=str(next(model.parameters()).device)
        settings['generated_image_dir']=str(SAMPLE_ROOT/'decoded'/sample_path.parent.name)
        image=evaluate_image(samples,data.split('test'),train_mean=data.mean,cfg=settings,
            output_dir=output/'image_evaluation',labels=generated['labels'])
        metrics={**image['image'],**image['latent']}; details=image
    # Preserve existing conditional-angular metric, including empty-bin failures.
    if cfg['kind']!='vector':
        from rafm.metrics.angular import angular_metrics
        set_all_seeds(cfg['evaluation']['metric_seed'])
        metrics.update(angular_metrics(samples,data.split('test'),n_bins=4,n_projections=200))
    return metrics,details


def evaluate(cfg,data,output,arm,seed):
    output=Path(output); model,saved=load_model(cfg,data,output,arm,seed,'final')
    checkpoint_hash=sha256(output/'checkpoint.pt')
    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])
    key=common.json_hash({'run':saved['run_sha256'],'checkpoint_sha256':checkpoint_hash})
    sample_path=SAMPLE_ROOT/cfg['condition_id']/arm/f'seed_{seed}'/key/'samples.pt'
    sample_path.parent.mkdir(parents=True,exist_ok=True); torch.save(generated,sample_path)
    metrics,details=quality(cfg,data,generated,model,output,sample_path)
    metrics.update(sample_time_s=generated['sample_time_s'],total_train_time_s=saved['train_time_s'],nfe=generated['nfe'])
    clean,bad=common.sanitize_metrics(metrics)
    result={'schema_version':1,'condition_id':cfg['condition_id'],'method':'rafm_ang_input','arm':arm,'seed':seed,
        'status':'failed' if bad else 'complete','config':cfg,'config_sha256':common.json_hash(cfg),
        'implementation_sha256':common.json_hash(implementation(cfg)),'metrics':clean,'nonfinite_metric_fields':bad,
        'parameters':model.parameter_report(),'radius_drift':generated['radius_drift'],
        'radius_statistics':model.radius_statistics.to_dict(),'hardware':common.hardware(),
        'checkpoint':{'path':str((output/'checkpoint.pt').resolve()),'sha256':checkpoint_hash,'step':saved['step']},
        'sample_artifact':{'path':str(sample_path),'sha256':sha256(sample_path)},
        'sampler':{k:v for k,v in generated.items() if k not in ('samples','labels','initial_radii')},
        'dataset':common.dataset_manifest(data),'evaluation_details':details,
        'peak_memory':{'training':saved['peak_memory'],'sampling':generated['peak_memory'],
            'sampling_and_evaluation':common.peak_memory(next(model.parameters()).device)},
        'training_stats':json.loads((output/'training_stats.json').read_text())}
    common.write_json(output/'result.json',result)
    if bad: raise FloatingPointError(f'Undefined/nonfinite metrics retained: {bad}')
    common.write_json(output/'metrics.json',clean)
    return result


def sanity(cfg,output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Preserve previous sanity results')
    data=load_data(cfg,include_external_test=False)
    rows=[]; seed=46021; budget=16 if cfg['kind']=='vector' else 8
    for arm in ARMS:
        arm_dir=output/arm
        try:
            trained=train(cfg,data,arm_dir,arm,seed,stage='sanity',budget=budget)
            model,saved=load_model(cfg,data,arm_dir,arm,seed,'sanity')
            if not trained['gradient_observed']: raise RuntimeError('No backbone gradient in sanity')
            if arm=='B' and not trained['radius_embedding_gradient_observed']:
                raise RuntimeError('No radius-conditioning gradient in B sanity')
            gen=sample(cfg,model,data,128 if cfg['kind']=='vector' else 40,seed=61717)
            row={'arm':arm,'status':'passed','training':trained,'n_generated':len(gen['samples']),
                 'nfe':gen['nfe'],'model_calls_total':gen['model_calls_total'],'radius_drift':gen['radius_drift'],
                 'finite_samples':True,'checkpoint_strict_load':True,'test_data_used':False}
            common.write_json(arm_dir/'sanity.json',row); rows.append(row)
        except Exception as error:
            common.write_json(arm_dir/'sanity.json',{'arm':arm,'status':'failed','error':str(error),'traceback':traceback.format_exc()})
            raise
    if rows[1]['training']['parameters']!=rows[2]['training']['parameters']:
        # Parameter reports may include arm name; compare counts explicitly below.
        b=rows[1]['training']['parameters']; c=rows[2]['training']['parameters']
        for key in b:
            if 'param' in key and isinstance(b[key],(int,float)) and b[key]!=c[key]:
                raise RuntimeError('B/C parameter count mismatch')
    common.write_json(output/'sanity.json',{'status':'passed','condition_id':cfg['condition_id'],'rows':rows,'no_tuning':True,'disposable_updates_per_arm':budget})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['sanity','train-evaluate']); p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=ARMS); p.add_argument('--seed',type=int)
    p.add_argument('--output',default='outputs_rafm_input_study/v1')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text())
    if args.action=='sanity':
        sanity(cfg,Path(args.output)/'sanity'/cfg['condition_id']); return
    if args.arm is None or args.seed is None: p.error('Final evaluation requires --arm and --seed')
    validate(cfg,args.arm)
    out=Path(args.output)/'final'/cfg['condition_id']/args.arm/f'seed_{args.seed}'
    if (out/'result.json').exists():
        existing=json.loads((out/'result.json').read_text())
        if (existing.get('status')=='complete' and existing.get('config_sha256')==common.json_hash(cfg)
                and existing.get('implementation_sha256')==common.json_hash(implementation(cfg))):
            print('Already complete:',out); return
        raise RuntimeError('Existing failure/mismatch preserved; explicit review required before retry')
    try:
        data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)
    except Exception as error:
        if not (out/'result.json').exists():
            common.write_json(out/'result.json',{'schema_version':1,'status':'failed','condition_id':cfg['condition_id'],
                'method':'rafm_ang_input','arm':args.arm,'seed':args.seed,'config':cfg,'config_sha256':common.json_hash(cfg),
                'failure':{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},
                'hardware':common.hardware()})
        raise


if __name__=='__main__': main()
