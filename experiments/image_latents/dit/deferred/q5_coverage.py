"""Q5: does RAFM's radial-tail calibration give better PERCEPTUAL coverage of real tail examples?

For each method, extract DINOv2 features of its (already-decoded) generations, then measure how well
those generations cover the REAL high-radius tail in DINO space:
  * mean/median nearest-generated-neighbour distance for real tail examples (lower = better covered);
  * prdc coverage & recall of the real tail set by each method's generations;
  * control: same for the real median-radius set.
If RAFM (best radial tail) does NOT cover real tails better than Gaussian, then radial calibration does
not buy perceptual coverage of rare/extreme images.  GPU (DINO on generations), then CPU.
"""
import os, glob
from pathlib import Path
import numpy as np, torch, json
REPO=Path(__file__).resolve().parents[4]
RS=REPO/"experiments/image_latents/radial_semantics"
METHODS=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]
SEED=8925; STEP=40000

def dino_extract(paths, dev, bs=64):
    import timm
    from torchvision import transforms; from PIL import Image
    m=timm.create_model("vit_base_patch14_dinov2.lvd142m",pretrained=True,num_classes=0,
                        img_size=224,dynamic_img_size=True).to(dev).eval()
    tf=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    out=[]
    with torch.no_grad():
        for i in range(0,len(paths),bs):
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]).to(dev)
            out.append(m(x).cpu().numpy())
    return np.concatenate(out)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    d=np.load(RS/"radial_quantile_indices.npz"); r=d["radii"]; realF=np.load(RS/"dino_feats.npy")
    tail=np.where(r>=np.quantile(r,0.90))[0]; mid=np.where((r>=np.quantile(r,0.45))&(r<=np.quantile(r,0.55)))[0]
    realF_tail=realF[tail]; realF_mid=realF[mid]
    from sklearn.neighbors import NearestNeighbors
    from prdc import compute_prdc
    res={}
    for mth in METHODS:
        cache=RS/f"gen_dino_{mth}.npy"
        if cache.exists(): G=np.load(cache)
        else:
            paths=sorted(glob.glob(str(REPO/f"experiments/image_latents/dit_sit/seed_{SEED}/eval_std/{mth}/gen_{STEP}/*.png")))
            G=dino_extract(paths,dev); np.save(cache,G)
            print(f"[{mth}] gen DINO feats {G.shape}",flush=True)
        nn=NearestNeighbors(n_neighbors=1).fit(G)
        dt,_=nn.kneighbors(realF_tail); dm,_=nn.kneighbors(realF_mid)
        prdc_tail=compute_prdc(realF_tail,G,nearest_k=5)   # real=tail, fake=gen
        res[mth]={"tail_NN_dist_mean":round(float(dt.mean()),3),"tail_NN_dist_median":round(float(np.median(dt)),3),
                  "mid_NN_dist_mean":round(float(dm.mean()),3),
                  "tail_coverage":round(prdc_tail["coverage"],4),"tail_recall":round(prdc_tail["recall"],4),
                  "tail_density":round(prdc_tail["density"],4)}
        print(mth, res[mth],flush=True)
    (RS/"q5_coverage.json").write_text(json.dumps(res,indent=2))
    print("\n=== real-TAIL coverage by method (lower NN dist / higher coverage = better) ===")
    print(f"{'method':20} {'tailNN_mean':11} {'tailNN_med':10} {'tail_cov':9} {'tail_recall':11}")
    for m in METHODS:
        x=res[m]; print(f"{m:20} {x['tail_NN_dist_mean']:<11} {x['tail_NN_dist_median']:<10} {x['tail_coverage']:<9} {x['tail_recall']:<11}")

if __name__=="__main__": main()
