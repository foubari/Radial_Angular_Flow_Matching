"""DEFERRED (run after training). Radius intervention -> decode.

GPU + frozen DC-AE. Takes held-out TEST latents, rescales each to a set of target radial quantiles
(direction preserved, radius set to q05/q25/q50/q75/q95 of the TRAIN radial law), decodes both the
original and the rescaled latent, and tiles them so you can see what changing ONLY the radius does to
the decoded image. Tests whether the decoder is radius-tolerant (mechanism behind the negative result).

Preprocessing matches training: model space = scaled x0.41407, train-centered (mu). Decode inverts:
raw = (z_model + mu)/0.41407.
"""
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[4]; import sys; sys.path.insert(0,str(REPO))
SF=0.41407
def split_idx(n,seed=0):
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr,nva=int(n*0.6),int(n*0.2); return p[:ntr],p[ntr:ntr+nva],p[ntr+nva:]

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    lat=torch.load(REPO/"experiments/image_latents/data/dcae_latents_scaled.pt",map_location="cpu").float()
    n=len(lat); tr,va,te=split_idx(n,0); mu=lat[tr].mean(0); z=lat-mu
    rtr=z[tr].norm(dim=1).numpy(); targets={q:float(np.quantile(rtr,q)) for q in (0.05,0.25,0.5,0.75,0.95)}
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
    def dec(zc):
        with torch.no_grad(): return ((ae.decode(((zc+mu)/SF).to(dev).reshape(-1,32,8,8)).sample.clamp(-1,1)+1)/2).cpu()
    from torchvision.utils import save_image
    OUT=REPO/"experiments/image_latents/figures/radius_intervention"; OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0); pick=te[torch.tensor(rng.choice(len(te),8,replace=False))]
    for j,i in enumerate(pick):
        zc=z[i:i+1]; u=zc/zc.norm(); rows=[dec(zc)]
        for q,R in targets.items(): rows.append(dec(u*R))
        save_image(torch.cat(rows),OUT/f"intervene_{j}.png",nrow=len(rows))
    (OUT/"targets.txt").write_text(str(targets)); print("DONE radius intervention. targets:",targets)

if __name__=="__main__": main()
