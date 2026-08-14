"""DEFERRED (run after training). Uncurated image grids by radial quantile.

CPU-only but reads many original JPEGs -> heavy disk I/O, so deferred out of the training window.
Loads the ORIGINAL dataset images (no decode, no neural model) for a random sample of each radial
quantile group and tiles them, so you can visually inspect what low-radius vs high-radius images
look like. Index i <-> i-th path of sorted(glob(".../imagenette2-320/**/*.JPEG")).
"""
import glob
from pathlib import Path
import numpy as np, torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

REPO=Path(__file__).resolve().parents[4]
NPZ=REPO/"experiments/image_latents/radial_semantics/radial_quantile_indices.npz"
OUT=REPO/"experiments/image_latents/figures/radial_quantile_grids"; OUT.mkdir(parents=True,exist_ok=True)
PATHS=sorted(glob.glob(str(REPO/"experiments/image_latents/data/imagenette2-320/**/*.JPEG"),recursive=True))
TF=transforms.Compose([transforms.Resize(160),transforms.CenterCrop(160),transforms.ToTensor()])

def grid(idxs,name,k=16,seed=0):
    rng=np.random.default_rng(seed); pick=rng.choice(idxs,min(k,len(idxs)),replace=False)
    ims=torch.stack([TF(Image.open(PATHS[i]).convert("RGB")) for i in pick])
    save_image(ims,OUT/f"{name}.png",nrow=4); print(f"  {name}: {len(pick)} imgs -> {name}.png",flush=True)

def main():
    d=np.load(NPZ); print("groups:",[k for k in d.files if k not in("radii","labels","train_idx","val_idx","test_idx")])
    for g in ["bottom05","bottom10","median45_55","top10","top05","top01"]:
        grid(d[g],f"radius_{g}")
    print("DONE radial grids.")

if __name__=="__main__": main()
