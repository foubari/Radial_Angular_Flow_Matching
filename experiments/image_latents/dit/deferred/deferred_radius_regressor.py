"""DEFERRED (run after training; CPU-only once DINO features exist).

Two diagnostics on the radius semantics:
  (a) regression: predict the latent RADIUS from DINO features (Ridge) -> R^2. High R^2 => radius is
      perceptually meaningful (encodes image content), not a nuisance scale.
  (b) classification: predict the class label from the radius ALONE (1-D logistic) -> accuracy vs the
      10-way chance (0.10). Tells how much class information the scalar radius carries (cf. eta^2=0.16).

Needs radial_semantics/dino_feats.npy (from deferred_dino_rarity.py) for (a); (b) needs only the npz.
"""
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[4]
NPZ=REPO/"experiments/image_latents/radial_semantics/radial_quantile_indices.npz"
FEAT=REPO/"experiments/image_latents/radial_semantics/dino_feats.npy"

def main():
    d=np.load(NPZ); r=d["radii"]; y=d["labels"]; tr=d["train_idx"]; te=d["test_idx"]
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score
    out={}
    if FEAT.exists():
        F=np.load(FEAT)
        reg=Ridge(alpha=10.0).fit(F[tr],r[tr]); pr=reg.predict(F[te])
        out["radius_from_dino_R2"]=float(r2_score(r[te],pr))
    else:
        out["radius_from_dino_R2"]="SKIPPED (run deferred_dino_rarity.py first)"
    clf=LogisticRegression(max_iter=500,multi_class="multinomial").fit(r[tr].reshape(-1,1),y[tr])
    out["class_from_radius_acc"]=float(accuracy_score(y[te],clf.predict(r[te].reshape(-1,1))))
    out["chance_acc"]=round(1/len(set(y.tolist())),3)
    (REPO/"experiments/image_latents/radial_semantics/radius_predictors.json").write_text(str(out))
    print(out); print("DONE radius predictors.")

if __name__=="__main__": main()
