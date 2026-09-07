"""Aggregate PoC-B eval.json across the 4 methods -> table + figure. CPU."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; M=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]; short=["gaussian","matched","fixed_sph","rafm"]
R={m:json.loads((HERE/"eval"/m/"eval.json").read_text()) for m in M if (HERE/"eval"/m/"eval.json").exists()}
if not R: print("no eval jsons yet"); raise SystemExit
print(f"{'method':20} {'e_W1':7} {'e_KS':6} {'>q95':6} {'>q99':7} {'<q10':6} {'PIT':6} {'digitAcc':8} {'acc lo/mid/hi'}")
for m in M:
    if m not in R: continue
    e=R[m]["energy"]; c=R[m]["content"]; ab=c["acc_by_energy"]
    print(f"{m:20} {e['radial_w1']:<7} {e['ks']:<6} {e['cov_gt_q95']:<6} {e['cov_gt_q99']:<7} {e['cov_lt_q10']:<6} {e['pit_mean']:<6} {c['digit_acc']:<8} {ab['low_energy']}/{ab['mid_energy']}/{ab['high_energy']}")
# figure: energy KS + digit acc
fig,(a,b)=plt.subplots(1,2,figsize=(10,4)); x=np.arange(len(M)); col=["#888","#5b8","#39c","#e44"]
a.bar(x,[R[m]["energy"]["ks"] for m in M],color=col); a.set_xticks(x); a.set_xticklabels(short,rotation=15)
a.set_ylabel("energy KS vs data (lower=better)"); a.set_title("Energy-distribution fidelity")
for i,m in enumerate(M): a.text(i,R[m]["energy"]["ks"]+.005,f"{R[m]['energy']['ks']:.3f}",ha="center",fontsize=8)
b.bar(x,[R[m]["content"]["digit_acc"] for m in M],color=col); b.axhline(0.955,ls="--",c="k",lw=1); b.text(len(M)-1,0.96,"real 0.955",ha="right",fontsize=8)
b.set_xticks(x); b.set_xticklabels(short,rotation=15); b.set_ylabel("digit accuracy (content)"); b.set_ylim(0,1); b.set_title("Content preservation")
for i,m in enumerate(M): b.text(i,R[m]["content"]["digit_acc"]+.02,f"{R[m]['content']['digit_acc']:.2f}",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig(HERE.parent/"image_latents/figures/pocB_energy_vs_content.png",dpi=140)
print("saved figure pocB_energy_vs_content.png")
