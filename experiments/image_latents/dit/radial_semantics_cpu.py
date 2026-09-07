"""Radial-semantics analysis — CPU ONLY, safe to run alongside GPU training.

HARD CONSTRAINTS (do not disturb the training job):
  * GPU hidden (CUDA_VISIBLE_DEVICES="") — no CUDA context, no VRAM.
  * single process, 1 thread (OMP/MKL/torch), no workers, no multiprocessing.
  * NO neural model loaded (no DC-AE/DINO/Inception), NO decode/encode, NO FID/KID.
  * reads the already-cached latents ONCE; PCA on a bounded subsample only.

Uses the SAME preprocessing as training (scaled latents x0.41407 already on disk, then train-only
centering with the seed-0 60/20/20 split) so radii live in the model space where RAFM's radial
source is fit. Outputs -> experiments/image_latents/radial_semantics/.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""          # hide GPU BEFORE torch import
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import json, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/"experiments/image_latents/radial_semantics"; OUT.mkdir(parents=True,exist_ok=True)
SCALED=REPO/"experiments/image_latents/data/dcae_latents_scaled.pt"
LABELS=REPO/"experiments/image_latents/data/dcae_labels.pt"
IMGROOT=REPO/"experiments/image_latents/data/imagenette2-320/train"

def split_idx(n,seed=0):                        # identical to dit_train_sit.split_idx
    g=torch.Generator().manual_seed(seed); p=torch.randperm(n,generator=g)
    ntr,nva=int(n*0.6),int(n*0.2); return p[:ntr],p[ntr:ntr+nva],p[ntr+nva:]

def main():
    lat=torch.load(SCALED,map_location="cpu").float(); lab=torch.load(LABELS,map_location="cpu").long()
    n,D=lat.shape; tr,va,te=split_idx(n,0)
    mu=lat[tr].mean(0); z=(lat-mu)                       # model space (train-centered)
    r=z.norm(dim=1).numpy()                              # per-sample radius (full dataset)
    rtr=r[tr.numpy()]                                    # TRAIN radii define the law
    y=lab.numpy(); classes=sorted(set(y.tolist()))
    cls_names=[p.name for p in sorted(IMGROOT.iterdir())] if IMGROOT.exists() else [str(c) for c in classes]

    # ---- 1. global radial quantiles (TRAIN law) ----
    qs=[0.0,0.01,0.05,0.10,0.25,0.45,0.50,0.55,0.75,0.90,0.95,0.99,1.0]
    gq={f"q{int(q*100):02d}" if q not in(0.0,1.0) else ("min" if q==0.0 else "max"):float(np.quantile(rtr,q)) for q in qs}
    summary={"n_total":int(n),"n_train":int(len(tr)),"D":int(D),
             "radius_train":{"mean":float(rtr.mean()),"std":float(rtr.std()),"cov":float(rtr.std()/rtr.mean()),
                             "min":float(rtr.min()),"max":float(rtr.max())},
             "train_radial_quantiles":gq}

    # ---- 2. quantile index groups (into FULL dataset), thresholds from TRAIN law ----
    def thr(q): return float(np.quantile(rtr,q))
    groups={
        "bottom05":np.where(r<=thr(0.05))[0], "bottom10":np.where(r<=thr(0.10))[0],
        "median45_55":np.where((r>=thr(0.45))&(r<=thr(0.55)))[0],
        "top10":np.where(r>=thr(0.90))[0], "top05":np.where(r>=thr(0.95))[0],
        "top01":np.where(r>=thr(0.99))[0],
    }
    np.savez(OUT/"radial_quantile_indices.npz", radii=r, labels=y, train_idx=tr.numpy(),
             val_idx=va.numpy(), test_idx=te.numpy(), **{k:v for k,v in groups.items()})
    summary["quantile_group_sizes"]={k:int(len(v)) for k,v in groups.items()}
    summary["quantile_group_thresholds"]={"bottom05":thr(0.05),"bottom10":thr(0.10),
        "median45_55":[thr(0.45),thr(0.55)],"top10":thr(0.90),"top05":thr(0.95),"top01":thr(0.99)}

    # ---- 3. per-class radius distributions + class-conditional quantiles ----
    percls={}
    for c in classes:
        rc=r[y==c]
        percls[cls_names[c]]={"n":int(len(rc)),"mean":float(rc.mean()),"std":float(rc.std()),
            "q05":float(np.quantile(rc,0.05)),"q50":float(np.quantile(rc,0.50)),
            "q95":float(np.quantile(rc,0.95)),"min":float(rc.min()),"max":float(rc.max())}
    summary["per_class_radius"]=percls

    # ---- 4. correlations / class effect on radius ----
    # correlation ratio eta^2 = between-class var / total var (how much radius is explained by class)
    grand=r.mean(); ss_tot=((r-grand)**2).sum()
    ss_bet=sum(len(r[y==c])*(r[y==c].mean()-grand)**2 for c in classes)
    eta2=float(ss_bet/ss_tot)
    # rank the classes by mean radius
    order=sorted(classes,key=lambda c:r[y==c].mean())
    summary["class_effect_on_radius"]={"eta_squared":eta2,
        "interpretation":"fraction of radius variance explained by class label (0=none,1=fully separable)",
        "classes_low_to_high_radius":[cls_names[c] for c in order]}

    # ---- 5. randomized PCA on a BOUNDED subsample (CPU, no full PCA) ----
    from sklearn.decomposition import PCA
    rng=np.random.default_rng(0); k=min(4000,n); sub=rng.choice(n,k,replace=False)
    Zsub=z[sub].numpy()
    pca=PCA(n_components=50,svd_solver="randomized",random_state=0).fit(Zsub)
    scores=pca.transform(Zsub); rsub=r[sub]
    # correlation of radius with each PC score, and with score magnitude
    corr=[float(np.corrcoef(rsub,scores[:,i])[0,1]) for i in range(scores.shape[1])]
    # radius vs overall latent norm in PC space is trivially high; report top-|corr| PCs
    top_pc=int(np.argmax(np.abs(corr)))
    summary["pca_subsample"]={"k":int(k),"n_components":50,
        "explained_var_ratio_top10":[float(x) for x in pca.explained_variance_ratio_[:10]],
        "cum_evr_50":float(pca.explained_variance_ratio_.sum()),
        "max_abs_radius_pc_corr":{"pc":top_pc,"corr":corr[top_pc]},
        "radius_pc_corr_top10":corr[:10]}
    np.savez(OUT/"pca_subsample.npz", sub=sub, scores=scores.astype(np.float32),
             evr=pca.explained_variance_ratio_, components=pca.components_.astype(np.float32), radii_sub=rsub)

    (OUT/"radial_semantics.json").write_text(json.dumps(summary,indent=2))
    # compact human summary
    lines=["# Radial-semantics (CPU, model space = scaled x0.41407, train-centered)\n",
        f"- n={n} (train {len(tr)}), D={D}; radius mean {rtr.mean():.2f} std {rtr.std():.2f} CoV {rtr.std()/rtr.mean():.3f}",
        f"- train radial quantiles: q05={gq['q05']:.2f} q50={gq['q50']:.2f} q95={gq['q95']:.2f} q99={gq['q99']:.2f} (min {gq['min']:.2f} max {gq['max']:.2f})",
        f"- quantile group sizes: "+", ".join(f"{k}={len(v)}" for k,v in groups.items()),
        f"- class effect on radius: eta^2={eta2:.3f} (classes low->high radius: {', '.join(cls_names[c] for c in order)})",
        f"- PCA(sub={k}): top-10 EVR sum {pca.explained_variance_ratio_[:10].sum():.3f}, 50-comp {pca.explained_variance_ratio_.sum():.3f}; max |corr(radius,PC)|={abs(corr[top_pc]):.3f} @PC{top_pc}",
        "\nSaved: radial_semantics.json, radial_quantile_indices.npz, pca_subsample.npz",]
    (OUT/"SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines)); print("\nDONE radial-semantics (CPU-only).")

if __name__=="__main__": main()
