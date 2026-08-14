"""Screening A4: token-radius heatmaps, local-radius interventions (causal effect), and
real-vs-generated local-radius distribution plots. GPU (decode) + matplotlib.
"""
import glob
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[4]; RS=REPO/"experiments/image_latents/radial_semantics"; SF=0.41407
FIG=REPO/"experiments/image_latents/figures"; (FIG).mkdir(exist_ok=True)
METHODS=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]
dev="cuda" if torch.cuda.is_available() else "cpu"
R=torch.load(RS/"real_centered.pt",map_location="cpu"); data=R["data"]; mu=R["mu"]; te=R["te"]
from diffusers import AutoencoderDC
ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
@torch.no_grad()
def dec(Z):  # Z (b,32,8,8) centered -> (b,256,256,3)
    raw=((Z.reshape(-1,2048)+mu)/SF).to(dev).reshape(-1,32,8,8)
    return ((ae.decode(raw).sample.clamp(-1,1)+1)/2).cpu().permute(0,2,3,1).numpy()

rng=np.random.default_rng(1); pick=te[torch.tensor(rng.choice(len(te),4,replace=False))]
Zp=data[pick].reshape(-1,32,8,8)
imgs=dec(Zp); tokr=Zp.permute(0,2,3,1).reshape(-1,8,8,32).norm(dim=3).numpy()

# Fig 1: image + token-radius heatmap
fig,ax=plt.subplots(2,4,figsize=(11,5.6))
for j in range(4):
    ax[0,j].imshow(imgs[j]); ax[0,j].axis("off"); ax[0,j].set_title("decoded",fontsize=8)
    im=ax[1,j].imshow(tokr[j],cmap="viridis"); ax[1,j].axis("off"); ax[1,j].set_title("token radius ||z[:,h,w]||",fontsize=8)
    fig.colorbar(im,ax=ax[1,j],fraction=0.046)
plt.tight_layout(); plt.savefig(FIG/"A_token_radius_heatmaps.png",dpi=140); plt.close(); print("saved A_token_radius_heatmaps.png")

# Fig 2: single-token magnitude intervention (fix direction, scale one token norm)
scales=[0.25,0.5,1.0,2.0,4.0]; base=data[pick[0]].reshape(32,8,8).clone()
# choose the max-radius token and a mid token
tk=tokr[0]; hmax,wmax=np.unravel_index(tk.argmax(),tk.shape)
rows=[]
for (h,w,tag) in [(hmax,wmax,"maxtoken"),(4,4,"centertoken")]:
    row=[]
    for s in scales:
        z=base.clone(); z[:,h,w]=z[:,h,w]*s; row.append(z.reshape(1,32,8,8))
    rows.append(torch.cat(row))
allz=torch.cat(rows); out=dec(allz)
fig,ax=plt.subplots(2,len(scales),figsize=(2.0*len(scales),4.2))
for r in range(2):
    for c in range(len(scales)):
        ax[r,c].imshow(out[r*len(scales)+c]); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(f"x{scales[c]}",fontsize=9)
ax[0,0].set_ylabel("max-radius token",fontsize=8); ax[1,0].set_ylabel("center token",fontsize=8)
plt.suptitle("Single-token magnitude intervention (direction fixed)",fontsize=10)
plt.tight_layout(); plt.savefig(FIG/"A_token_intervention.png",dpi=140); plt.close(); print("saved A_token_intervention.png")

# Fig 3: real vs generated token-radius distributions
def tok_radii(Z): return Z.reshape(-1,32,8,8).permute(0,2,3,1).reshape(-1,32).norm(dim=1).numpy()
realr=tok_radii(data[te])
plt.figure(figsize=(6,4))
plt.hist(realr,bins=80,density=True,histtype="step",lw=2,label="real",color="k")
for m in METHODS:
    g=torch.load(RS/f"genlat_{m}.pt",map_location="cpu"); plt.hist(tok_radii(g),bins=80,density=True,histtype="step",lw=1.2,label=m.replace("_euclidean","").replace("_"," "))
plt.xlabel("token radius ||z[:,h,w]||"); plt.ylabel("density"); plt.legend(fontsize=7); plt.title("Token-radius distribution: real vs generated")
plt.tight_layout(); plt.savefig(FIG/"A_token_radius_real_vs_gen.png",dpi=140); plt.close(); print("saved A_token_radius_real_vs_gen.png")
print("A4 DONE")
