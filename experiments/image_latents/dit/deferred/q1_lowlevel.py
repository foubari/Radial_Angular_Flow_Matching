"""Q1 (quantitative): what low-level property does latent radius track? CPU.
Grids show high-radius = flat white/black background product shots, low-radius = busy natural scenes.
Quantify: correlate radius with brightness, global contrast, and background flatness
(fraction of near-white/near-black pixels; border-ring std) on a fixed subsample.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
import glob, json
from pathlib import Path
import numpy as np
from PIL import Image
REPO=Path(__file__).resolve().parents[4]
NPZ=REPO/"experiments/image_latents/radial_semantics/radial_quantile_indices.npz"
PATHS=sorted(glob.glob(str(REPO/"experiments/image_latents/data/imagenette2-320/**/*.JPEG"),recursive=True))

def feats(img):
    g=np.asarray(img.convert("L").resize((256,256)),dtype=np.float32)/255.0
    rgb=np.asarray(img.convert("RGB").resize((256,256)),dtype=np.float32)/255.0
    border=np.concatenate([g[:16].ravel(),g[-16:].ravel(),g[:,:16].ravel(),g[:,-16:].ravel()])
    return {"brightness":float(g.mean()),"contrast":float(g.std()),
            "white_frac":float((g>0.9).mean()),"black_frac":float((g<0.1).mean()),
            "flat_extreme_frac":float(((g>0.9)|(g<0.1)).mean()),"border_std":float(border.std()),
            "color_sat":float(rgb.max(2).mean()-rgb.min(2).mean())}

def main():
    d=np.load(NPZ); r=d["radii"]
    rng=np.random.default_rng(0); k=3000; sub=np.sort(rng.permutation(len(PATHS))[:k])
    keys=["brightness","contrast","white_frac","black_frac","flat_extreme_frac","border_std","color_sat"]
    acc={kk:[] for kk in keys}; rr=[]
    for i in sub:
        try: f=feats(Image.open(PATHS[i]))
        except Exception: continue
        for kk in keys: acc[kk].append(f[kk])
        rr.append(r[i])
    rr=np.array(rr)
    corr={kk:round(float(np.corrcoef(rr,np.array(acc[kk]))[0,1]),3) for kk in keys}
    # also mean feature in low vs high radius terciles
    lo=rr<=np.quantile(rr,0.10); hi=rr>=np.quantile(rr,0.90)
    contrast={"low_radius_mean":{kk:round(float(np.array(acc[kk])[lo].mean()),3) for kk in keys},
              "high_radius_mean":{kk:round(float(np.array(acc[kk])[hi].mean()),3) for kk in keys}}
    out={"n":len(rr),"pearson_corr_radius_vs":corr,"low_vs_high_radius_decile":contrast}
    (REPO/"experiments/image_latents/radial_semantics/q1_lowlevel.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
