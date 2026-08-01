"""Plot DiT pilot trajectories: FID-vs-step and radial-W1/sliced-W1-vs-step across the 4 methods."""
import json, glob, re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
EV=Path("experiments/image_latents/dit/eval")
methods=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]
col={"gaussian_euclidean":"#888","matched_euclidean":"#4a90d9","fixed_spherical":"#e0883a","rafm":"#2ca02c"}
def latent_traj(m,key):
    out={}
    for p in glob.glob(str(EV/m/"metrics_*.json")):
        s=int(re.search(r"metrics_(\d+).json",p).group(1)); out[s]=json.loads(Path(p).read_text())[key]
    return sorted(out.items())
def fid_traj(m):
    f=EV/m/"fid_traj.json"
    if not f.exists(): return []
    return sorted((int(k),v) for k,v in json.loads(f.read_text()).items())
fig,ax=plt.subplots(1,3,figsize=(16,4.2))
for m in methods:
    ft=fid_traj(m)
    if ft: ax[0].plot([s for s,_ in ft],[v for _,v in ft],"o-",color=col[m],label=m)
    rt=latent_traj(m,"radial_w1")
    if rt: ax[1].plot([s for s,_ in rt],[v for _,v in rt],"o-",color=col[m],label=m)
    stt=latent_traj(m,"sliced_w1")
    if stt: ax[2].plot([s for s,_ in stt],[v for _,v in stt],"o-",color=col[m],label=m)
ax[0].axhline(21.3,ls=":",c="k",label="decoder rFID floor"); ax[0].set_title("small-FID vs step (DC-AE, Imagenette-10)")
ax[1].set_yscale("log"); ax[1].set_title("latent radial-W1 vs step")
ax[2].set_title("latent sliced-W1 vs step")
for a in ax: a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("experiments/image_latents/figures/dit_pilot_trajectories.png",dpi=130)
print("saved dit_pilot_trajectories.png")
