"""SiT-backbone flow-matching training on SCALED DC-AE latents — four-way, resumable.

Backbone: official SiT (third_party/SiT/models.py, class SiT, learn_sigma=False -> velocity).
Transport: rafm library (4 source/path variants). Latents = raw DC-AE * 0.41407 (scaled), then
train-only centering. See DECISIONS.md (D1, D2, D4).

RESUMABLE across VM stops (USER PRIORITY): checkpoint saves model+EMA+optimizer+step+RNG states
(torch/cuda/numpy/python). Data indices are STEP-SEEDED (deterministic function of step) so batch
order is resume-exact regardless of interruption. Re-run the same command to continue. One process,
no background respawn.
"""
import argparse, json, sys, time, copy, random
from pathlib import Path
import numpy as np, torch

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"third_party/SiT"))
from models import SiT
from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.paths.euclidean import EuclideanPath
from rafm.paths.spherical_geodesic import SphericalGeodesicPath

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def split_idx(n,seed=0):
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr,nva=int(n*0.6),int(n*0.2); return p[:ntr],p[ntr:ntr+nva],p[ntr+nva:]

def build(method, latents, labels, split_seed):
    tr,va,te=split_idx(len(latents),split_seed)
    mu=latents[tr].mean(0); data=latents-mu; R0=float(data[tr].norm(dim=1).mean())
    if method=="fixed_spherical":
        r=data.norm(dim=1,keepdim=True).clamp(min=1e-8); train=(R0*data/r)[tr]
    else:
        train=data[tr]
    src=GaussianSource() if method=="gaussian_euclidean" else RadialEmpiricalSource(mode="ecdf").fit(train)
    path=SphericalGeodesicPath() if method in ("fixed_spherical","rafm","angular_rafm") else EuclideanPath()
    return dict(tr=tr,va=va,te=te,mu=mu,data=data,train=train,src=src,path=path,
                spherical=method in ("fixed_spherical","rafm","angular_rafm"),R0=R0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True,choices=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm","angular_rafm"])
    ap.add_argument("--out",default="experiments/image_latents/dit_sit")
    ap.add_argument("--latents",default="experiments/image_latents/data/dcae_latents_scaled.pt")
    ap.add_argument("--labels",default="experiments/image_latents/data/dcae_labels.pt")
    ap.add_argument("--steps",type=int,default=40000); ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--hidden",type=int,default=384); ap.add_argument("--depth",type=int,default=12); ap.add_argument("--heads",type=int,default=6)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--ckpt_every",type=int,default=2000); ap.add_argument("--log_every",type=int,default=500)
    ap.add_argument("--ema",type=float,default=0.9999); ap.add_argument("--class_dropout",type=float,default=0.1)
    ap.add_argument("--seed",type=int,default=8925); ap.add_argument("--split_seed",type=int,default=0)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    set_seed(a.seed)
    lat=torch.load(a.latents,map_location="cpu").float(); lab=torch.load(a.labels,map_location="cpu").long()
    st=build(a.method,lat,lab,a.split_seed); D=lat.shape[1]
    train=st["train"].to(dev); train_y=lab[st["tr"]].to(dev)
    model=SiT(input_size=8,patch_size=1,in_channels=32,hidden_size=a.hidden,depth=a.depth,num_heads=a.heads,
              num_classes=10,class_dropout_prob=a.class_dropout,learn_sigma=False).to(dev)
    ema=copy.deepcopy(model).eval();[p.requires_grad_(False) for p in ema.parameters()]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.0,betas=(0.9,0.95))
    run=Path(a.out)/"runs"/a.method; run.mkdir(parents=True,exist_ok=True); ckpt=run/"ckpt.pt"; start=0; log=[]
    if ckpt.exists():
        c=torch.load(ckpt,map_location="cpu",weights_only=False)   # cpu: RNG state must stay a CPU ByteTensor
        model.load_state_dict(c["model"]); ema.load_state_dict(c["ema"]); opt.load_state_dict(c["opt"])
        for s in opt.state.values():                                # move optimizer state to device
            for k,v in s.items():
                if torch.is_tensor(v): s[k]=v.to(dev)
        start=c["step"]; log=c.get("log",[])
        torch.set_rng_state(c["rng"]["torch"].to(torch.uint8).cpu())
        if c["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in c["rng"]["cuda"]])
        np.random.set_state(c["rng"]["numpy"]); random.setstate(c["rng"]["python"])
        print(f"[resume] {a.method} @ step {start} (RNG restored)",flush=True)
    (run/"meta.json").write_text(json.dumps({"n_params":sum(p.numel() for p in model.parameters()),
        "R0":st["R0"],"backbone":"SiT_official_pinned","latent":"scaled_x0.41407_train_centered","args":vars(a)},indent=2))
    print(f"{a.method}: SiT {sum(p.numel() for p in model.parameters())/1e6:.1f}M | n_train {len(train)} | dev {dev} | steps {start}->{a.steps}",flush=True)
    model.train(); t0=time.time()
    for step in range(start+1,a.steps+1):
        g=torch.Generator().manual_seed(a.seed*1_000_003+step)             # step-deterministic batch
        idx=torch.randint(len(train),(a.batch,),generator=g).to(dev)
        x1=train[idx]; y=train_y[idx]; n=x1.shape[0]
        if a.method=="gaussian_euclidean":
            x0=torch.randn(n,D,device=dev)
        else:
            R=x1.norm(dim=1,keepdim=True); u0=torch.randn(n,D,device=dev); x0=R*u0/u0.norm(dim=1,keepdim=True)
        t=torch.rand(n,device=dev)
        xt=st["path"].sample_path(x0,x1,t); ut=st["path"].conditional_vector_field(x0,x1,t)
        if a.method=="angular_rafm":                                       # scale-free angular target A=ut/||xt||
            ut=ut/xt.norm(dim=1,keepdim=True).clamp(min=1e-8)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            v=model(xt.reshape(n,32,8,8),t,y).reshape(n,-1)
            loss=((v.float()-ut)**2).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for pe,pm in zip(ema.parameters(),model.parameters()): pe.mul_(a.ema).add_(pm,alpha=1-a.ema)
        if step%a.log_every==0:
            log.append({"step":step,"loss":float(loss),"elapsed_s":time.time()-t0})
            print(f"  [{a.method}] {step}/{a.steps} loss {float(loss):.1f} {(time.time()-t0)/(step-start)*1000:.0f}ms/it",flush=True)
        if step%a.ckpt_every==0 or step==a.steps:
            rng={"torch":torch.get_rng_state(),"cuda":(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                 "numpy":np.random.get_state(),"python":random.getstate()}
            torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"opt":opt.state_dict(),"step":step,"log":log,"rng":rng},ckpt)
            torch.save({"ema":ema.state_dict(),"step":step},run/f"ema_{step}.pt")
            (run/"train_log.json").write_text(json.dumps(log))
    print(f"DONE {a.method} {a.steps} in {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
