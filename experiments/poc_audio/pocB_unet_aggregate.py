"""Aggregate UNet PoC-B eval across methods x checkpoints -> tables + trajectory figure. CPU."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; M=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]
short={"gaussian_euclidean":"gaussian","matched_euclidean":"matched","fixed_spherical":"fixed_sph","rafm":"rafm"}
STEPS=[12000,18000,24000]
def load(m,s):
    f=HERE/"eval_unet"/m/f"step{s}"/"eval.json"; return json.loads(f.read_text()) if f.exists() else None
for s in STEPS:
    print(f"\n===== checkpoint {s} =====")
    print(f"{'method':12} {'e_KS':7} {'>q95(.05)':10} {'>q99(.01)':10} {'<q10(.10)':10} {'PIT':6} {'digitAcc':9} {'lo/mid/hi'}")
    for m in M:
        d=load(m,s)
        if not d: print(f"{short[m]:12} (pending)"); continue
        e=d["energy"]; c=d["content"]; ab=c["acc_by_energy"]
        print(f"{short[m]:12} {e['ks']:<7} {e['cov_gt_q95']:<10} {e['cov_gt_q99']:<10} {e['cov_lt_q10']:<10} {e['pit_mean']:<6} {c['digit_acc']:<9} {ab['low_energy']}/{ab['mid_energy']}/{ab['high_energy']}")
# figure: digit-acc trajectory + energy KS @24k
col={"gaussian_euclidean":"#888","matched_euclidean":"#5b8","fixed_spherical":"#39c","rafm":"#e44"}
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.2))
for m in M:
    accs=[load(m,s)["content"]["digit_acc"] if load(m,s) else np.nan for s in STEPS]
    a.plot(STEPS,accs,"o-",color=col[m],label=short[m])
a.axhline(0.955,ls="--",c="k",lw=1); a.text(24000,0.96,"real 0.955",ha="right",fontsize=8)
a.set_xlabel("training step"); a.set_ylabel("digit accuracy"); a.set_title("Content: digit accuracy trajectory"); a.legend(fontsize=8); a.set_ylim(0,1)
ks=[load(m,24000)["energy"]["ks"] if load(m,24000) else np.nan for m in M]
b.bar(range(4),ks,color=[col[m] for m in M]); b.set_xticks(range(4)); b.set_xticklabels([short[m] for m in M],rotation=15)
b.set_ylabel("energy KS vs data (lower=better)"); b.set_title("Energy calibration @24k")
for i,v in enumerate(ks): b.text(i,v+0.002,f"{v:.3f}",ha="center",fontsize=8)
plt.tight_layout(); out=HERE.parent/"image_latents/figures/pocB_unet_trajectory.png"; plt.savefig(out,dpi=140); print("\nsaved",out)
