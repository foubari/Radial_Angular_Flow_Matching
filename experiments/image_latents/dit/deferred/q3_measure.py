"""Q3 quantify: decode fixed directions at radius quantiles; measure decoded contrast & pixel change.
Semantic change is measured as pixel change vs the q50 decode (same direction). GPU.
"""
import sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[4]; SF=0.41407
def split_idx(n,seed=0):
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr,nva=int(n*0.6),int(n*0.2); return p[:ntr],p[ntr:ntr+nva],p[ntr+nva:]
dev="cuda" if torch.cuda.is_available() else "cpu"
lat=torch.load(REPO/"experiments/image_latents/data/dcae_latents_scaled.pt",map_location="cpu").float()
n=len(lat); tr,va,te=split_idx(n,0); mu=lat[tr].mean(0); z=lat-mu
rtr=z[tr].norm(dim=1).numpy(); qs=[0.05,0.25,0.5,0.75,0.95]; Rq=[float(np.quantile(rtr,q)) for q in qs]
from diffusers import AutoencoderDC
ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
rng=np.random.default_rng(0); pick=te[torch.tensor(rng.choice(len(te),32,replace=False))]
U=(z[pick]/z[pick].norm(dim=1,keepdim=True))
@torch.no_grad()
def dec(zc): return ((ae.decode(((zc+mu)/SF).to(dev).reshape(-1,32,8,8)).sample.clamp(-1,1)+1)/2).cpu()
imgs={q:dec(U*R) for q,R in zip(qs,Rq)}
base=imgs[0.5]
print("radius_q | decoded_contrast(std) | mean|Δpix vs q50|")
for q in qs:
    c=float(imgs[q].std()); d=float((imgs[q]-base).abs().mean())
    print(f"  q{int(q*100):02d} (R={dict(zip(qs,Rq))[q]:.1f})   {c:.4f}                {d:.4f}")
print("\n(semantic change ~ pixel change vs q50; contrast should rise with radius)")
