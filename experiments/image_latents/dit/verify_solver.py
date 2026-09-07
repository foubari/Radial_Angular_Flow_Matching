"""Instrument the ACTUAL image sampler: count real model forward calls + measure norm drift.
CPU-only (CUDA hidden) so it does not disturb the running GPU eval. Counts via a forward hook
(fires once per real forward), NOT by reading the config name.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"
import sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"third_party/SiT")); sys.path.insert(0,str(REPO/"experiments/image_latents/dit"))
from models import SiT
from dit_train_sit import build
from dit_eval_sit import rk4_sample

def load(method, step=40000, seed=8925):
    lat=torch.load(REPO/"experiments/image_latents/data/dcae_latents_scaled.pt",map_location="cpu").float()
    lab=torch.load(REPO/"experiments/image_latents/data/dcae_labels.pt",map_location="cpu").long()
    st=build(method,lat,lab,0); D=lat.shape[1]
    m=SiT(input_size=8,patch_size=1,in_channels=32,hidden_size=384,depth=12,num_heads=6,
          num_classes=10,class_dropout_prob=0.1,learn_sigma=False).eval()
    ema=torch.load(REPO/f"experiments/image_latents/dit_sit/seed_{seed}/runs/{method}/ema_{step}.pt",map_location="cpu")["ema"]
    m.load_state_dict(ema); return m,st,D

def source_x0(method,st,D,n,seed=0):
    torch.manual_seed(seed)
    if method=="gaussian_euclidean": return torch.randn(n,D)
    r=st["src"].sample(n,D).norm(dim=1,keepdim=True); u0=torch.randn(n,D); return r*u0/u0.norm(dim=1,keepdim=True)

def run(method, nfe):
    m,st,D=load(method); n=16; x0=source_x0(method,st,D,n)
    y=torch.arange(10).repeat_interleave(2)[:n]
    calls=[0]; h=m.register_forward_hook(lambda mod,i,o: calls.__setitem__(0,calls[0]+1))
    xf=rk4_sample(m,x0,y,st["spherical"],nfe=nfe); h.remove()
    r0=x0.norm(dim=1); rf=xf.norm(dim=1); drift=(rf-r0)/r0
    return calls[0], float(r0.mean()), float(rf.mean()), float(drift.abs().mean()*100), float(drift.abs().max()*100)

print(f"{'method':18} {'nfe(steps)':11} {'model_calls':12} {'r0_mean':9} {'rf_mean':9} {'|drift|%_mean':13} {'|drift|%_max':12}")
for method in ["fixed_spherical","rafm"]:
    for nfe in [1,25]:
        c,r0,rf,dm,dx=run(method,nfe)
        print(f"{method:18} {nfe:<11} {c:<12} {r0:<9.3f} {rf:<9.3f} {dm:<13.4f} {dx:<12.4f}")
# euclidean control (no projection) at nfe=25 for call-count parity
c,r0,rf,dm,dx=run("gaussian_euclidean",25)
print(f"{'gaussian_euclid':18} {25:<11} {c:<12} {r0:<9.3f} {rf:<9.3f} {dm:<13.4f} {dx:<12.4f}  (euclid, no proj)")
print("\nRK4 => 4 model calls per step; state x is NOT renormalized; velocity is tangent-projected each stage for spherical.")
