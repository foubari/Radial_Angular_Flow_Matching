"""Punchline figure: RAFM wins RADIAL-tail calibration (Q4) but NOT perceptual coverage (Q5). CPU."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[4]; RS=REPO/"experiments/image_latents/radial_semantics"
M=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]; short=["gaussian","matched","fixed_sph","rafm"]
tail={m:json.loads((RS/f"tail_{m}_8925_40000.json").read_text()) for m in M}
cov=json.loads((RS/"q5_coverage.json").read_text())
q95=[tail[m]["tail_coverage_gt_q95"] for m in M]; pcov=[cov[m]["tail_coverage"] for m in M]
x=np.arange(4); c=["#888","#5b8","#39c","#e44"]
fig,(a,b)=plt.subplots(1,2,figsize=(10,4))
a.bar(x,q95,color=c); a.axhline(0.05,ls="--",c="k",lw=1); a.text(3.5,0.052,"target 0.05",ha="right",fontsize=8)
a.set_xticks(x); a.set_xticklabels(short,rotation=15); a.set_ylabel("gen mass above real q95")
a.set_title("Q4: RADIAL-tail calibration\n(RAFM ≈ target; Gaussian misses the tail)")
for i,v in enumerate(q95): a.text(i,v+0.001,f"{v:.3f}",ha="center",fontsize=8)
b.bar(x,pcov,color=c); b.set_xticks(x); b.set_xticklabels(short,rotation=15)
b.set_ylabel("DINO coverage of real high-radius tail"); b.set_ylim(0,0.06)
b.set_title("Q5: PERCEPTUAL coverage of real tail\n(all methods tied — RAFM no better)")
for i,v in enumerate(pcov): b.text(i,v+0.0008,f"{v:.3f}",ha="center",fontsize=8)
plt.tight_layout(); out=REPO/"experiments/image_latents/figures/q4q5_tail_vs_perceptual.png"
plt.savefig(out,dpi=140); print("saved",out)
