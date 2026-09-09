"""CPU-only interface checks on untrained backbones; no fitting or benchmark."""
import json
from pathlib import Path
import platform

import torch

from baselines.tflow_downstream import build_backbone, flat_noise_callback
from rafm.models.mlp import MLP


root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
torch.manual_seed(61723)
result={'purpose':'CPU forward-interface checks with unit-test fixtures; no training or benchmark',
        'host':platform.node(),'torch':str(torch.__version__),'checks':[]}
with torch.no_grad():
    model=MLP(input_dim=64,hidden_dim=128,n_layers=3).eval()
    out=model(torch.randn(2,64),torch.tensor([0.25,0.75]))
    assert out.shape==(2,64) and torch.isfinite(out).all()
    result['checks'].append({'kind':'mlp','parameters':sum(p.numel() for p in model.parameters()),'shape':list(out.shape),'finite':True})
    del model,out
    for condition in ('audiomnist_stft','imagenette_dcae'):
        cfg=json.loads((root/'configs/tflow/prepared'/f'{condition}.json').read_text())
        dim=cfg['data']['shape'][1]
        model=build_backbone(cfg['model'],dim).eval()
        callback=flat_noise_callback(model,cfg['model'],torch.tensor([2]))
        out=callback(torch.randn(1,dim),torch.tensor([0.5]))
        assert out.shape==(1,dim) and torch.isfinite(out).all()
        result['checks'].append({'kind':cfg['model']['kind'],'parameters':sum(p.numel() for p in model.parameters()),
                                 'shape':list(out.shape),'finite':True,'provenance':model.tflow_provenance})
        del model,out,callback
result['status']='passed'
(root/'docs/artifact_audit/backbone_contract_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
