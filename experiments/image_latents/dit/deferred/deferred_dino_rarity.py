"""DEFERRED (run after training). DINO/Inception rarity vs radius + nearest neighbours.

GPU + a pretrained feature extractor (timm DINOv2 or torch-fidelity InceptionV3). Extracts one
feature vector per ORIGINAL image, computes kNN distance (rarity: mean distance to k nearest in
feature space), and correlates rarity with the latent RADIUS (does high radius <-> perceptual
rarity?). Also dumps, for the top-radius tail, each image's nearest feature-space neighbour.

Writes features to radial_semantics/dino_feats.npy so deferred_radius_regressor.py can reuse them
CPU-only. Index i <-> i-th sorted JPEG path (same convention as extraction).
"""
import glob
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[4]
PATHS=sorted(glob.glob(str(REPO/"experiments/image_latents/data/imagenette2-320/**/*.JPEG"),recursive=True))
NPZ=REPO/"experiments/image_latents/radial_semantics/radial_quantile_indices.npz"
FEAT=REPO/"experiments/image_latents/radial_semantics/dino_feats.npy"

def extract(dev,bs=32,backbone="dinov2_vitb14"):
    import timm
    from torchvision import transforms; from PIL import Image
    m=timm.create_model("vit_base_patch14_dinov2.lvd142m",pretrained=True,num_classes=0,
                        img_size=224,dynamic_img_size=True).to(dev).eval()  # DINOv2 default is 518; interpolate pos-embed to 224
    tf=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    feats=[]
    with torch.no_grad():
        for i in range(0,len(PATHS),bs):
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in PATHS[i:i+bs]]).to(dev)
            feats.append(m(x).cpu().numpy())
            if i%(bs*20)==0: print(f"  {i}/{len(PATHS)}",flush=True)
    return np.concatenate(feats)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    F=np.load(FEAT) if FEAT.exists() else extract(dev); np.save(FEAT,F)
    d=np.load(NPZ); r=d["radii"]
    # kNN rarity on a bounded subsample (feature space is 768-d; full kNN is O(n^2), subsample)
    from sklearn.neighbors import NearestNeighbors
    rng=np.random.default_rng(0); sub=rng.choice(len(F),min(4000,len(F)),replace=False)
    nn=NearestNeighbors(n_neighbors=6).fit(F[sub]); dist,_=nn.kneighbors(F[sub])
    rarity=dist[:,1:].mean(1)          # mean dist to 5 nearest (exclude self)
    corr=float(np.corrcoef(r[sub],rarity)[0,1])
    print(f"corr(radius, DINO kNN-rarity) = {corr:.3f}  (n_sub={len(sub)})")
    np.savez(REPO/"experiments/image_latents/radial_semantics/dino_rarity.npz",sub=sub,rarity=rarity,radii_sub=r[sub],corr=corr)
    print("DONE dino rarity.")

if __name__=="__main__": main()
