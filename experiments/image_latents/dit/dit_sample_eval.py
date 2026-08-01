"""Sample latents from a trained DiT (EMA) + latent metrics. Saves gen latents for decoding.

Reuses the same source/path/split as training (dit_train.build). Tangent-projected sampling for
spherical methods. Class labels sampled uniformly over the 10 classes. Metrics vs centered test.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(HERE))
from dit_model import DiT
from dit_train import build
from rafm.flow_matching.sampler import _project_tangent
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics

def sliced(A,B,dirs):
    q=np.linspace(0,1,1000); return float(np.mean(np.abs(np.quantile(A@dirs,q,0)-np.quantile(B@dirs,q,0))))
def directional(gen,test,seed=0,n_proj=300):
    g=gen.numpy().astype(np.float64); t=test.numpy().astype(np.float64)
    gu=g/np.clip(np.linalg.norm(g,axis=1,keepdims=True),1e-8,None); tu=t/np.clip(np.linalg.norm(t,axis=1,keepdims=True),1e-8,None)
    rng=np.random.default_rng(seed); d=g.shape[1]; dirs=rng.standard_normal((d,n_proj)); dirs/=np.linalg.norm(dirs,axis=0,keepdims=True)
    rt=np.linalg.norm(t,axis=1); rdraw=rt[rng.integers(0,len(rt),len(gu))]
    return sliced(gu,tu,dirs), sliced(gu*rdraw[:,None],t,dirs)

@torch.no_grad()
def sample(model, source, spherical, D, n, steps, device, seed=123):
    torch.manual_seed(seed)
    x=source.sample(n,D,device=device)
    y=torch.randint(0,10,(n,),device=device)
    dt=1.0/steps
    for i in range(steps):
        tt=torch.full((n,),i*dt,device=device)
        def vel(xx):
            v=model(xx.reshape(n,32,8,8),tt,y).reshape(n,-1)
            return _project_tangent(v,xx) if spherical else v
        # RK4
        k1=vel(x); k2=vel(x+dt/2*k1); k3=vel(x+dt/2*k2); k4=vel(x+dt*k3)
        x=x+dt/6*(k1+2*k2+2*k3+k4)
    return x.cpu()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True); ap.add_argument("--ema",required=True,help="path to ema_{step}.pt")
    ap.add_argument("--out",default="experiments/image_latents/dit")
    ap.add_argument("--latents",default="experiments/image_latents/data/dcae_latents.pt")
    ap.add_argument("--labels",default="experiments/image_latents/data/dcae_labels.pt")
    ap.add_argument("--n",type=int,default=5000); ap.add_argument("--nfe_steps",type=int,default=50)
    ap.add_argument("--hidden",type=int,default=384); ap.add_argument("--depth",type=int,default=12); ap.add_argument("--heads",type=int,default=6)
    ap.add_argument("--split_seed",type=int,default=0); ap.add_argument("--save_latents",action="store_true")
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    lat=torch.load(a.latents,map_location="cpu").float(); lab=torch.load(a.labels,map_location="cpu").long()
    class A: pass
    aa=A(); aa.method=a.method; aa.split_seed=a.split_seed
    st=build(aa,lat,lab); D=lat.shape[1]; test=st["data"][st["te"]]
    model=DiT(in_ch=32,size=8,patch=1,hidden=a.hidden,depth=a.depth,heads=a.heads,num_classes=10).to(dev).eval()
    model.load_state_dict(torch.load(a.ema,map_location=dev)["ema"])
    step=torch.load(a.ema,map_location="cpu")["step"]
    n=min(a.n,len(test))
    gen=sample(model,st["src"],st["spherical"],D,n,a.nfe_steps,dev)
    m={}; m.update(radial_metrics(gen,test)); m.update(distributional_metrics(gen,test,n_projections=300))
    m.update(angular_metrics(gen,test,n_bins=4,n_projections=100)); m.update(stability_metrics(gen))
    m["dir_sliced_w1"],m["cr_sliced_w1"]=directional(gen,test)
    m["step"]=int(step); m["nfe"]=a.nfe_steps*4
    run=Path(a.out)/"eval"/a.method; run.mkdir(parents=True,exist_ok=True)
    (run/f"metrics_{step}.json").write_text(json.dumps(m,indent=2))
    if a.save_latents:
        torch.save(gen,run/f"gen_{step}.pt"); torch.save(st["mu"],run/"mu.pt")
    print(f"[{a.method} step {step}] radial={m['radial_w1']:.3f} sliced={m['sliced_w1']:.3f} "
          f"dir={m['dir_sliced_w1']:.4f} cr={m['cr_sliced_w1']:.4f} nan={m['nan_rate']:.3f}",flush=True)

if __name__=="__main__": main()
