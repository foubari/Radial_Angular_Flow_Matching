"""PoC-B B2: digit-conditioned flow-matching on STFT AudioMNIST, 4 source/path methods, resumable.

Backbone: compact conv velocity net on (2,129,63). Transport: rafm library (same 4 methods as the image
experiment). NO centering (by construction ||x||=g=energy, the radius we study). Gaussian source is
MOMENT-MATCHED (sigma=E[g]/sqrt(D)) for a fair baseline. Complete-state resumable (model+EMA+opt+step+RNG,
step-seeded batches) — re-run the same command to continue.
"""
import argparse, json, sys, time, copy, random, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; sys.path.insert(0,str(REPO))
from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.paths.euclidean import EuclideanPath
from rafm.paths.spherical_geodesic import SphericalGeodesicPath

FREQ,FRAMES=129,63; C_IN=2; D=C_IN*FREQ*FRAMES

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def split_idx(n,seed=0):
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr=int(n*0.85); return p[:ntr],p[ntr:]

def timestep_embed(t,dim=128):
    half=dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
    a=t[:,None]*f[None]; return torch.cat([a.sin(),a.cos()],dim=1)

class FiLM(nn.Module):
    def __init__(s,cond,ch): super().__init__(); s.l=nn.Linear(cond,2*ch)
    def forward(s,h,c): g,b=s.l(c)[:,:,None,None].chunk(2,1); return h*(1+g)+b
class ResBlock(nn.Module):
    def __init__(s,ch,cond):
        super().__init__(); s.n1=nn.GroupNorm(8,ch); s.c1=nn.Conv2d(ch,ch,3,padding=1)
        s.film=FiLM(cond,ch); s.n2=nn.GroupNorm(8,ch); s.c2=nn.Conv2d(ch,ch,3,padding=1)
    def forward(s,h,c):
        r=h; h=s.c1(torch.nn.functional.silu(s.n1(h))); h=s.film(h,c)
        h=s.c2(torch.nn.functional.silu(s.n2(h))); return h+r
class VelNet(nn.Module):
    def __init__(s,ch=64,K=5,ncls=10,cond=128):
        super().__init__(); s.inp=nn.Conv2d(C_IN,ch,3,padding=1)
        s.temb=nn.Sequential(nn.Linear(128,cond),nn.SiLU(),nn.Linear(cond,cond))
        s.yemb=nn.Embedding(ncls+1,cond); s.blocks=nn.ModuleList([ResBlock(ch,cond) for _ in range(K)])
        s.out=nn.Conv2d(ch,C_IN,3,padding=1); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s,x,t,y):
        c=s.temb(timestep_embed(t))+s.yemb(y); h=s.inp(x)
        for b in s.blocks: h=b(h,c)
        return s.out(h)

# ---- larger multi-scale UNet velocity net (self-attention at coarse levels) ----
class RB(nn.Module):
    def __init__(s,ci,co,cond):
        super().__init__(); s.n1=nn.GroupNorm(min(8,ci),ci); s.c1=nn.Conv2d(ci,co,3,padding=1)
        s.film=FiLM(cond,co); s.n2=nn.GroupNorm(min(8,co),co); s.c2=nn.Conv2d(co,co,3,padding=1)
        s.sk=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
    def forward(s,h,c):
        r=s.sk(h); h=s.c1(torch.nn.functional.silu(s.n1(h))); h=s.film(h,c)
        h=s.c2(torch.nn.functional.silu(s.n2(h))); return h+r
class Attn(nn.Module):
    def __init__(s,ch,heads=4):
        super().__init__(); s.h=heads; s.n=nn.GroupNorm(min(8,ch),ch); s.qkv=nn.Conv2d(ch,ch*3,1); s.pr=nn.Conv2d(ch,ch,1)
    def forward(s,x):
        B,C,H,W=x.shape; q,k,v=s.qkv(s.n(x)).reshape(B,3,s.h,C//s.h,H*W).unbind(1)
        a=torch.softmax((q.transpose(-1,-2)@k)/ (C//s.h)**0.5,dim=-1)
        o=(v@a.transpose(-1,-2)).reshape(B,C,H,W); return x+s.pr(o)
class UNetVel(nn.Module):
    def __init__(s,ch=96,mult=(1,2,4),ncls=10,cond=256,attn_from=2):  # attn only at coarsest level (memory)
        super().__init__(); chs=[ch*m for m in mult]
        s.temb=nn.Sequential(nn.Linear(128,cond),nn.SiLU(),nn.Linear(cond,cond)); s.yemb=nn.Embedding(ncls+1,cond)
        s.inp=nn.Conv2d(C_IN,chs[0],3,padding=1)
        s.enc=nn.ModuleList(); s.encattn=nn.ModuleList(); s.down=nn.ModuleList()
        for i,c in enumerate(chs):
            s.enc.append(nn.ModuleList([RB(c,c,cond),RB(c,c,cond)]))
            s.encattn.append(nn.ModuleList([Attn(c),Attn(c)]) if i>=attn_from else None)
            s.down.append(nn.Conv2d(c,chs[min(i+1,len(chs)-1)],3,2,1) if i<len(chs)-1 else None)
        s.mid=nn.ModuleList([RB(chs[-1],chs[-1],cond),Attn(chs[-1]),RB(chs[-1],chs[-1],cond)])
        s.up=nn.ModuleList(); s.dec=nn.ModuleList(); s.decattn=nn.ModuleList()
        for i in reversed(range(len(chs))):
            c=chs[i]; s.up.append(nn.Conv2d(chs[min(i+1,len(chs)-1)],c,3,padding=1) if i<len(chs)-1 else None)
            s.dec.append(nn.ModuleList([RB(2*c,c,cond),RB(c,c,cond)]))
            s.decattn.append(nn.ModuleList([Attn(c),Attn(c)]) if i>=attn_from else None)
        s.on=nn.GroupNorm(min(8,chs[0]),chs[0]); s.out=nn.Conv2d(chs[0],C_IN,3,padding=1)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s,x,t,y):
        c=s.temb(timestep_embed(t))+s.yemb(y); h=s.inp(x); skips=[]; sizes=[]
        n=len(s.enc)
        for i in range(n):
            for j,rb in enumerate(s.enc[i]):
                h=rb(h,c);
                if s.encattn[i] is not None: h=s.encattn[i][j](h)
            skips.append(h); sizes.append(h.shape[-2:])
            if s.down[i] is not None: h=s.down[i](h)
        h=s.mid[0](h,c); h=s.mid[1](h); h=s.mid[2](h,c)
        for k,i in enumerate(reversed(range(n))):
            if s.up[k] is not None: h=s.up[k](torch.nn.functional.interpolate(h,size=sizes[i],mode="nearest"))
            h=torch.cat([h,skips[i]],dim=1)
            for j,rb in enumerate(s.dec[k]):
                h=rb(h,c)
                if s.decattn[k] is not None: h=s.decattn[k][j](h)
        return s.out(torch.nn.functional.silu(s.on(h)))

def make_model(arch,ch,depth,ncls=10):
    return UNetVel(ch=ch,ncls=ncls) if arch=="unet" else VelNet(ch=ch,K=depth,ncls=ncls)

def build(method,x,labels,split_seed):
    tr,te=split_idx(len(x),split_seed); R0=float(x[tr].norm(dim=1).mean())
    sigma=R0/math.sqrt(D)                                   # moment-matched gaussian scale
    if method=="fixed_spherical":
        r=x.norm(dim=1,keepdim=True).clamp(min=1e-8); train=(R0*x/r)[tr]
    else: train=x[tr]
    src=GaussianSource() if method=="gaussian_euclidean" else RadialEmpiricalSource(mode="ecdf").fit(train)
    path=SphericalGeodesicPath() if method in ("fixed_spherical","rafm") else EuclideanPath()
    return dict(tr=tr,te=te,train=train,src=src,path=path,spherical=method in ("fixed_spherical","rafm"),R0=R0,sigma=sigma)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True,choices=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"])
    ap.add_argument("--out",default="experiments/poc_audio/runs"); ap.add_argument("--data",default="experiments/poc_audio/data/audiomnist_stft_train.pt")
    ap.add_argument("--steps",type=int,default=20000); ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--arch",default="velnet",choices=["velnet","unet"]); ap.add_argument("--ch",type=int,default=64); ap.add_argument("--depth",type=int,default=5)
    ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--ckpt_every",type=int,default=2000); ap.add_argument("--log_every",type=int,default=500)
    ap.add_argument("--ema",type=float,default=0.999); ap.add_argument("--class_dropout",type=float,default=0.1)
    ap.add_argument("--angular",action="store_true")  # predict A=v/||x_t||; reconstruct v=||x||*A at sampling
    ap.add_argument("--seed",type=int,default=8925); ap.add_argument("--split_seed",type=int,default=0)
    ap.add_argument("--ncls",type=int,default=10)   # conditioning classes (digits=10, words=N)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"; set_seed(a.seed)
    blob=torch.load(a.data,map_location="cpu"); x=blob["x"].reshape(len(blob["x"]),-1).float()
    y=(blob["word"] if "word" in blob else blob["digit"]).long()   # generic label
    st=build(a.method,x,y,a.split_seed); train=st["train"].to(dev); train_y=y[st["tr"]].to(dev)
    model=make_model(a.arch,a.ch,a.depth,a.ncls).to(dev); ema=copy.deepcopy(model).eval();[p.requires_grad_(False) for p in ema.parameters()]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.0)
    run=Path(a.out)/a.method; run.mkdir(parents=True,exist_ok=True); ckpt=run/"ckpt.pt"; start=0; log=[]
    if ckpt.exists():
        c=torch.load(ckpt,map_location="cpu",weights_only=False)
        model.load_state_dict(c["model"]); ema.load_state_dict(c["ema"]); opt.load_state_dict(c["opt"])
        for s in opt.state.values():
            for k,v in s.items():
                if torch.is_tensor(v): s[k]=v.to(dev)
        start=c["step"]; log=c.get("log",[]); torch.set_rng_state(c["rng"]["torch"].to(torch.uint8).cpu())
        if c["rng"].get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([t.to(torch.uint8).cpu() for t in c["rng"]["cuda"]])
        np.random.set_state(c["rng"]["numpy"]); random.setstate(c["rng"]["python"]); print(f"[resume] {a.method} @ {start}",flush=True)
    (run/"meta.json").write_text(json.dumps({"n_params":sum(p.numel() for p in model.parameters()),"R0":st["R0"],"sigma":st["sigma"],"D":D,"args":vars(a)},indent=2))
    print(f"{a.method}: VelNet {sum(p.numel() for p in model.parameters())/1e6:.1f}M | n_train {len(train)} | R0 {st['R0']:.2f} | steps {start}->{a.steps}",flush=True)
    model.train(); t0=time.time()
    for step in range(start+1,a.steps+1):
        g=torch.Generator().manual_seed(a.seed*1_000_003+step); idx=torch.randint(len(train),(a.batch,),generator=g).to(dev)
        x1=train[idx]; yy=train_y[idx].clone(); n=x1.shape[0]
        drop=torch.rand(n,generator=torch.Generator().manual_seed(a.seed*7+step))<a.class_dropout
        yy[drop.to(dev)]=a.ncls                             # null class
        if a.method=="gaussian_euclidean": x0=st["sigma"]*torch.randn(n,D,device=dev)
        else:
            R=x1.norm(dim=1,keepdim=True); u0=torch.randn(n,D,device=dev); x0=R*u0/u0.norm(dim=1,keepdim=True)
        t=torch.rand(n,device=dev); xt=st["path"].sample_path(x0,x1,t); ut=st["path"].conditional_vector_field(x0,x1,t)
        tgt=(ut/xt.norm(dim=1,keepdim=True).clamp(min=1e-8)) if a.angular else ut   # angular: scale-free target A=v/||x_t||
        with torch.autocast("cuda",dtype=torch.bfloat16):
            v=model(xt.reshape(n,2,FREQ,FRAMES),t,yy).reshape(n,-1); loss=((v.float()-tgt)**2).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for pe,pm in zip(ema.parameters(),model.parameters()): pe.mul_(a.ema).add_(pm,alpha=1-a.ema)
        if step%a.log_every==0:
            tn=tgt.norm(dim=1)  # target-norm distribution (std-vel: ~||v||; angular: ~||v||/||x_t||)
            log.append({"step":step,"loss":float(loss),"tgt_norm_mean":float(tn.mean()),"tgt_norm_std":float(tn.std())})
            print(f"  [{a.method}{'/ang' if a.angular else ''}] {step}/{a.steps} loss {float(loss):.2f} tgtN {float(tn.mean()):.2f}±{float(tn.std()):.2f} {(time.time()-t0)/(step-start)*1000:.0f}ms/it",flush=True)
        if step%a.ckpt_every==0 or step==a.steps:
            rng={"torch":torch.get_rng_state(),"cuda":(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),"numpy":np.random.get_state(),"python":random.getstate()}
            torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"opt":opt.state_dict(),"step":step,"log":log,"rng":rng},ckpt)
            torch.save({"ema":ema.state_dict(),"step":step},run/f"ema_{step}.pt"); (run/"train_log.json").write_text(json.dumps(log))
    print(f"DONE {a.method} {a.steps} in {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
