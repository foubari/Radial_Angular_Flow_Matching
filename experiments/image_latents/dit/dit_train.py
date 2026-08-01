"""DiT flow-matching training on DC-AE latents — four-way (source x path), Shadow-GPU, resumable.

Methods (same DiT arch/budget; only source+path differ):
  gaussian_euclidean : Gaussian source        + Euclidean path
  matched_euclidean  : empirical radial source + Euclidean path
  fixed_spherical    : fixed radius R0 source  + spherical path  (trains on R0-normalized latents)
  rafm               : empirical radial source + spherical path  (tangent-projected sampling)

Latents flattened to 2048-d for source/coupling/path; DiT operates on [B,32,8,8]. Class-conditional
(CFG dropout). bf16 autocast. EMA. Checkpoints every --ckpt_every (RESUMABLE: model+ema+opt+step).
"""
import argparse, json, sys, time, copy
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from dit_model import DiT
from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.paths.euclidean import EuclideanPath
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.flow_matching.sampler import _project_tangent
from rafm.utils.seeds import set_all_seeds

def split_idx(n, seed=0):
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr,nva=int(n*0.6),int(n*0.2); return p[:ntr],p[ntr:ntr+nva],p[ntr+nva:]

def build(args, latents, labels):
    tr,va,te=split_idx(len(latents),args.split_seed)
    mu=latents[tr].mean(0)
    data=latents-mu
    R0=float(data[tr].norm(dim=1).mean())
    if args.method=="fixed_spherical":
        r=data.norm(dim=1,keepdim=True).clamp(min=1e-8); train=(R0*data/r)[tr]
    else:
        train=data[tr]
    src = GaussianSource() if args.method=="gaussian_euclidean" else RadialEmpiricalSource(mode="ecdf").fit(train)
    path = SphericalGeodesicPath() if args.method in ("fixed_spherical","rafm") else EuclideanPath()
    spherical = args.method in ("fixed_spherical","rafm")
    return dict(tr=tr,va=va,te=te,mu=mu,data=data,train=train,labels=labels,src=src,path=path,
                spherical=spherical,R0=R0)

def cfm_batch(model, b, D, args, device):
    x1=b["x1"]; y=b["y"]; n=x1.shape[0]
    if args.method=="gaussian_euclidean":
        x0=torch.randn(n,D,device=device)
    else:
        R=x1.norm(dim=1,keepdim=True); u0=torch.randn(n,D,device=device)
        x0=R*u0/u0.norm(dim=1,keepdim=True)
    t=torch.rand(n,device=device)
    xt=b["path"].sample_path(x0,x1,t); ut=b["path"].conditional_vector_field(x0,x1,t)
    v=model(xt.reshape(n,32,8,8),t,y,drop=True).reshape(n,-1)
    return ((v-ut)**2).sum(-1).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True,choices=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"])
    ap.add_argument("--out",default="experiments/image_latents/dit")
    ap.add_argument("--latents",default="experiments/image_latents/data/dcae_latents.pt")
    ap.add_argument("--labels",default="experiments/image_latents/data/dcae_labels.pt")
    ap.add_argument("--steps",type=int,default=20000); ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--hidden",type=int,default=384); ap.add_argument("--depth",type=int,default=12); ap.add_argument("--heads",type=int,default=6)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--ckpt_every",type=int,default=2000); ap.add_argument("--log_every",type=int,default=200)
    ap.add_argument("--ema",type=float,default=0.999); ap.add_argument("--seed",type=int,default=8925); ap.add_argument("--split_seed",type=int,default=0)
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    set_all_seeds(args.seed)
    latents=torch.load(args.latents,map_location="cpu").float(); labels=torch.load(args.labels,map_location="cpu").long()
    st=build(args,latents,labels); D=latents.shape[1]
    train_gpu=st["train"].to(dev); train_y=labels[st["tr"]].to(dev)
    model=DiT(in_ch=32,size=8,patch=1,hidden=args.hidden,depth=args.depth,heads=args.heads,num_classes=10).to(dev)
    ema=copy.deepcopy(model).eval();[p.requires_grad_(False) for p in ema.parameters()]
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0,betas=(0.9,0.95))
    run=Path(args.out)/"runs"/args.method; run.mkdir(parents=True,exist_ok=True)
    ckpt=run/"ckpt.pt"; start=0; log=[]
    if ckpt.exists():
        c=torch.load(ckpt,map_location=dev); model.load_state_dict(c["model"]); ema.load_state_dict(c["ema"])
        opt.load_state_dict(c["opt"]); start=c["step"]; log=c.get("log",[]); print(f"[resume] {args.method} @ step {start}",flush=True)
    torch.save({"n_params":model.num_params(),"R0":st["R0"],"args":vars(args)}, run/"meta.pt")
    print(f"{args.method}: DiT {model.num_params()/1e6:.1f}M | n_train {len(train_gpu)} | dev {dev}",flush=True)
    model.train(); t0=time.time()
    for step in range(start+1,args.steps+1):
        idx=torch.randint(len(train_gpu),(args.batch,),device=dev)
        b={"x1":train_gpu[idx],"y":train_y[idx],"path":st["path"]}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss=cfm_batch(model,b,D,args,dev)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            d=args.ema
            for pe,pm in zip(ema.parameters(),model.parameters()): pe.mul_(d).add_(pm,alpha=1-d)
        if step%args.log_every==0:
            log.append({"step":step,"loss":float(loss),"elapsed_s":time.time()-t0})
            print(f"  [{args.method}] step {step}/{args.steps} loss {float(loss):.1f} {(time.time()-t0)/(step-start)*1000:.0f}ms/it",flush=True)
        if step%args.ckpt_every==0 or step==args.steps:
            torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"opt":opt.state_dict(),"step":step,"log":log},ckpt)
            torch.save({"ema":ema.state_dict(),"step":step}, run/f"ema_{step}.pt")   # light snapshot for FID-vs-step
            (run/"train_log.json").write_text(json.dumps(log))
    print(f"DONE {args.method} {args.steps} steps in {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
