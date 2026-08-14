"""Q2 detail: radius vs DINO perceptual rarity, global + within-class, + scatter figure. CPU.
Rarity = mean distance to k nearest neighbours in DINOv2 feature space (higher = more isolated/rare).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[4]
RS=REPO/"experiments/image_latents/radial_semantics"
F=np.load(RS/"dino_feats.npy"); d=np.load(RS/"radial_quantile_indices.npz")
r=d["radii"]; y=d["labels"]
IMGROOT=REPO/"experiments/image_latents/data/imagenette2-320/train"
names=[p.name for p in sorted(IMGROOT.iterdir())] if IMGROOT.exists() else [str(c) for c in range(10)]

from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=6).fit(F)
dist,_=nn.kneighbors(F); rarity=dist[:,1:].mean(1)          # exclude self

glob=float(np.corrcoef(r,rarity)[0,1])
per={}
for c in sorted(set(y.tolist())):
    m=y==c; per[names[c]]=round(float(np.corrcoef(r[m],rarity[m])[0,1]),3)
out={"corr_radius_rarity_global":round(glob,3),"corr_within_class":per,
     "mean_abs_within_class":round(float(np.mean([abs(v) for v in per.values()])),3),
     "interpretation":"~0 => latent radius is UNRELATED to perceptual (DINO) rarity, globally and within class"}
(RS/"q2_rarity.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

# figure: radius vs rarity scatter (subsample for legibility)
rng=np.random.default_rng(0); s=rng.permutation(len(r))[:4000]
plt.figure(figsize=(5.2,4.2))
plt.scatter(r[s],rarity[s],s=4,alpha=0.35,c=y[s],cmap="tab10")
plt.xlabel("DC-AE latent radius  ||z||"); plt.ylabel("DINOv2 kNN rarity (mean dist to 5-NN)")
plt.title(f"Latent radius vs perceptual rarity  (r={glob:+.3f})")
plt.tight_layout(); fig=REPO/"experiments/image_latents/figures/q2_radius_vs_rarity.png"
plt.savefig(fig,dpi=140); print("saved",fig)
