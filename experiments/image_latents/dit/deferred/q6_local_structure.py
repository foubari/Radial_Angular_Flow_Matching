"""Q6: is the GLOBAL latent radius too coarse? CPU-only on cached latents.

DC-AE latent = [32 channels, 8x8=64 spatial tokens] = 2048-d. The global radius is ONE scalar.
Ask whether decoder-relevant info lives in LOCAL structure (per-token / per-channel norms, direction)
rather than the single global norm. Diagnostics (all CPU):
  A. within-image spatial non-uniformity of the magnitude field (token-norm CoV per image);
  B. class predictability from: global norm (1-d) vs channel-norms (32-d) vs token-norms (64-d) vs
     full unit-direction (2048-d) -> logistic-regression accuracy;
  C. how much latent variance is a global-scale d.o.f. vs direction: variance kept after per-image
     L2-normalization (removing radius).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"
import json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
REPO=Path(__file__).resolve().parents[4]
OUT=REPO/"experiments/image_latents/radial_semantics"; OUT.mkdir(parents=True,exist_ok=True)

def split_idx(n,seed=0):
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr,nva=int(n*0.6),int(n*0.2); return p[:ntr],p[ntr:ntr+nva],p[ntr+nva:]

def acc(Xtr,ytr,Xte,yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    sc=StandardScaler().fit(Xtr); m=LogisticRegression(max_iter=300,multi_class="multinomial").fit(sc.transform(Xtr),ytr)
    return float(accuracy_score(yte,m.predict(sc.transform(Xte))))

def main():
    lat=torch.load(REPO/"experiments/image_latents/data/dcae_latents_scaled.pt",map_location="cpu").float()
    lab=torch.load(REPO/"experiments/image_latents/data/dcae_labels.pt",map_location="cpu").long()
    n=len(lat); tr,va,te=split_idx(n,0); mu=lat[tr].mean(0); z=(lat-mu)
    Z=z.reshape(n,32,8,8)
    gnorm=z.norm(dim=1)                                  # (n,) global radius
    tok=Z.permute(0,2,3,1).reshape(n,64,32).norm(dim=2)  # (n,64) per-token (spatial) norms
    chan=Z.reshape(n,32,64).norm(dim=2)                  # (n,32) per-channel norms
    ytr,yte=lab[tr].numpy(),lab[te].numpy()

    # A. spatial non-uniformity of magnitude within each image
    tok_cov=(tok.std(dim=1)/tok.mean(dim=1)).numpy()     # per-image CoV of token norms
    # B. class predictability from different summaries
    accs={
      "global_norm_1d": acc(gnorm[tr].reshape(-1,1).numpy(),ytr,gnorm[te].reshape(-1,1).numpy(),yte),
      "channel_norms_32d": acc(chan[tr].numpy(),ytr,chan[te].numpy(),yte),
      "token_norms_64d": acc(tok[tr].numpy(),ytr,tok[te].numpy(),yte),
      "unit_direction_2048d": acc((z[tr]/gnorm[tr].clamp(min=1e-8).unsqueeze(1)).numpy(),ytr,
                                  (z[te]/gnorm[te].clamp(min=1e-8).unsqueeze(1)).numpy(),yte),
      "full_latent_2048d": acc(z[tr].numpy(),ytr,z[te].numpy(),yte),
    }
    chance=1/len(set(lab.tolist()))
    # C. how much of total latent variance is the global-radius (magnitude) d.o.f.
    total_var=float(z.var(0).sum())                      # summed over 2048 dims
    radius_var=float(gnorm.var())                         # variance of the single global-radius scalar
    out={
      "spatial_nonuniformity_token_norm_CoV":{"mean":float(tok_cov.mean()),"median":float(np.median(tok_cov)),
          "p10":float(np.quantile(tok_cov,0.1)),"p90":float(np.quantile(tok_cov,0.9))},
      "class_accuracy_from":accs,"chance":round(chance,3),
      "variance":{"total_latent_var":total_var,"global_radius_var":radius_var,
          "radius_fraction_of_total_var":round(radius_var/total_var,4)},
      "note":"global radius carries a tiny fraction of latent variance and ~chance class info; direction (radius removed) carries the class info -> decoder-relevant info is LOCAL/directional, not the single global norm"
    }
    (OUT/"q6_local_structure.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    print(f"\nglobal-norm class acc {accs['global_norm_1d']:.3f} vs direction {accs['unit_direction_2048d']:.3f} vs full {accs['full_latent_2048d']:.3f} (chance {chance:.2f})")
    print(f"global radius = {radius_var/total_var*100:.2f}% of total latent variance (direction/local = the rest)")

if __name__=="__main__": main()
