"""Screening A1: regenerate per-method GENERATED latents (the eval saved only decoded PNGs).
Sample n=3000 latents from each trained EMA (seed 8925, 40k, nfe25) and SAVE genlat_<method>.pt in
model space (centered, scaled). No decode. GPU.
"""
import sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"third_party/SiT")); sys.path.insert(0,str(REPO/"experiments/image_latents/dit"))
from models import SiT
from dit_train_sit import build
from dit_eval_sit import rk4_sample
RS=REPO/"experiments/image_latents/radial_semantics"; SEED=8925; STEP=40000; N=3000; NFE=25
METHODS=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    lat=torch.load(REPO/"experiments/image_latents/data/dcae_latents_scaled.pt",map_location="cpu").float()
    lab=torch.load(REPO/"experiments/image_latents/data/dcae_labels.pt",map_location="cpu").long()
    D=lat.shape[1]
    for mth in METHODS:
        st=build(mth,lat,lab,0)
        model=SiT(input_size=8,patch_size=1,in_channels=32,hidden_size=384,depth=12,num_heads=6,
                  num_classes=10,class_dropout_prob=0.1,learn_sigma=False).to(dev).eval()
        ema=torch.load(REPO/f"experiments/image_latents/dit_sit/seed_{SEED}/runs/{mth}/ema_{STEP}.pt",map_location="cpu")["ema"]
        model.load_state_dict(ema)
        torch.manual_seed(0); np.random.seed(0)
        y=torch.arange(10).repeat_interleave(int(np.ceil(N/10)))[:N].to(dev)
        if mth=="gaussian_euclidean": x0=torch.randn(N,D,device=dev)
        else:
            r=st["src"].sample(N,D).norm(dim=1,keepdim=True).to(dev); u0=torch.randn(N,D,device=dev); x0=r*u0/u0.norm(dim=1,keepdim=True)
        gen=torch.cat([rk4_sample(model,x0[i:i+512],y[i:i+512],st["spherical"],NFE).cpu() for i in range(0,N,512)])
        torch.save(gen, RS/f"genlat_{mth}.pt")
        print(f"{mth}: genlat {tuple(gen.shape)} norm-mean {gen.norm(dim=1).mean():.2f} -> genlat_{mth}.pt",flush=True)
    # also save the real centered test/train/val latents once for convenience
    tr=build("rafm",lat,lab,0); torch.save({"data":tr["data"],"tr":tr["tr"],"va":tr["va"],"te":tr["te"],"mu":tr["mu"]}, RS/"real_centered.pt")
    print("saved real_centered.pt",flush=True); print("A1 DONE")

if __name__=="__main__": main()
