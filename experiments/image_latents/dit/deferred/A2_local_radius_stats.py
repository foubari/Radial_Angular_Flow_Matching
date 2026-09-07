"""Screening A2: local-radius distributions, train/val/test stability, class predictivity,
and real-vs-generated local-radius fidelity per method. CPU. Needs A1 outputs (genlat_*, real_centered).

Local-radius families for z in R^{32x8x8}:
  global(1) | channel(32) | block2x2(4, 4x4 blocks) | block4x4(16, 2x2 blocks) | token(64)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="3"
import json
from pathlib import Path
import numpy as np, torch
from scipy.stats import skew, kurtosis, wasserstein_distance, ks_2samp
REPO=Path(__file__).resolve().parents[4]; RS=REPO/"experiments/image_latents/radial_semantics"
METHODS=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]

def families(Z):  # Z: (N,32,8,8) -> dict of (N,K) local-radius vectors
    N=Z.shape[0]
    glob=Z.reshape(N,-1).norm(dim=1,keepdim=True)
    chan=Z.reshape(N,32,64).norm(dim=2)
    tok=Z.permute(0,2,3,1).reshape(N,64,32).norm(dim=2)
    b44=Z.reshape(N,32,2,4,2,4).permute(0,2,4,1,3,5).reshape(N,4,-1).norm(dim=2)   # 2x2 grid of 4x4
    b22=Z.reshape(N,32,4,2,4,2).permute(0,2,4,1,3,5).reshape(N,16,-1).norm(dim=2)  # 4x4 grid of 2x2
    return {"global":glob,"channel":chan,"block2x2":b44,"block4x4":b22,"token":tok}

def dist_stats(v):  # pooled 1-D stats of a local-radius family
    x=v.reshape(-1).numpy()
    return {"mean":float(x.mean()),"cov":float(x.std()/x.mean()),"skew":float(skew(x)),
            "excess_kurtosis":float(kurtosis(x)),"q99_over_q50":float(np.quantile(x,0.99)/np.quantile(x,0.50)),
            "q999_over_q50":float(np.quantile(x,0.999)/np.quantile(x,0.50))}

def clf_acc(Xtr,ytr,Xte,yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    sc=StandardScaler().fit(Xtr); m=LogisticRegression(max_iter=300).fit(sc.transform(Xtr),ytr)
    return float(accuracy_score(yte,m.predict(sc.transform(Xte))))

def main():
    R=torch.load(RS/"real_centered.pt",map_location="cpu")
    data=R["data"]; tr,va,te=R["tr"],R["va"],R["te"]
    lab=torch.load(REPO/"experiments/image_latents/data/dcae_labels.pt",map_location="cpu").long().numpy()
    Z=data.reshape(-1,32,8,8); fam=families(Z)
    out={"families":list(fam.keys()),"pooled_distribution":{},"stability_KS_train_vs_test":{},
         "stability_KS_train_vs_val":{},"class_accuracy":{},"real_vs_gen":{}}
    # distributions + stability
    for k,v in fam.items():
        out["pooled_distribution"][k]=dist_stats(v)
        a=v[tr].reshape(-1).numpy(); b=v[te].reshape(-1).numpy(); c=v[va].reshape(-1).numpy()
        ss=np.random.default_rng(0)
        A=ss.choice(a,20000); B=ss.choice(b,20000); C=ss.choice(c,20000)
        out["stability_KS_train_vs_test"][k]=round(float(ks_2samp(A,B).statistic),4)
        out["stability_KS_train_vs_val"][k]=round(float(ks_2samp(A,C).statistic),4)
    # class predictivity
    ytr,yte=lab[tr],lab[te]
    feats={"global(1)":fam["global"],"channel(32)":fam["channel"],"block2x2(4)":fam["block2x2"],
           "block4x4(16)":fam["block4x4"],"token(64)":fam["token"],
           "direction(2048)":(data/data.norm(dim=1,keepdim=True).clamp(min=1e-8)),"full(2048)":data}
    for name,F in feats.items():
        out["class_accuracy"][name]=round(clf_acc(F[tr].numpy(),ytr,F[te].numpy(),yte),3)
    out["class_accuracy"]["chance"]=0.1
    # real vs generated local-radius fidelity (W1 + KS per family, pooled)
    rng=np.random.default_rng(0)
    for mth in METHODS:
        g=torch.load(RS/f"genlat_{mth}.pt",map_location="cpu").reshape(-1,32,8,8); gf=families(g)
        d={}
        for k in fam:
            real=fam[k][te].reshape(-1).numpy(); gen=gf[k].reshape(-1).numpy()
            n=min(20000,len(real),len(gen)); a=rng.choice(real,n); b=rng.choice(gen,n)
            d[k]={"w1":round(float(wasserstein_distance(a,b)),4),"ks":round(float(ks_2samp(a,b).statistic),4)}
        # joint dependence: token-radius correlation matrix distance (real vs gen)
        Ct=np.corrcoef(fam["token"][te].numpy().T); Cg=np.corrcoef(gf["token"].numpy().T)
        d["token_corr_frobenius_diff"]=round(float(np.linalg.norm(Ct-Cg)),3)
        out["real_vs_gen"][mth]=d
    (RS/"A2_local_radius.json").write_text(json.dumps(out,indent=2))
    # concise print
    print("=== pooled local-radius distributions (real) ===")
    for k,s in out["pooled_distribution"].items():
        print(f"  {k:10} CoV {s['cov']:.3f}  skew {s['skew']:+.2f}  exkurt {s['excess_kurtosis']:+.2f}  q99/q50 {s['q99_over_q50']:.2f}  KS(tr,te) {out['stability_KS_train_vs_test'][k]}")
    print("\n=== class accuracy from radius family (chance 0.10) ===")
    for k,a in out["class_accuracy"].items(): print(f"  {k:16} {a}")
    print("\n=== real-vs-gen token-radius KS + token-corr Frobenius (does model match LOCAL radii?) ===")
    for m in METHODS:
        d=out["real_vs_gen"][m]; print(f"  {m:20} token KS {d['token']['ks']:.3f}  channel KS {d['channel']['ks']:.3f}  block2x2 KS {d['block2x2']['ks']:.3f}  tokenCorrΔ {d['token_corr_frobenius_diff']}")
    print("\nA2 DONE ->",RS/"A2_local_radius.json")

if __name__=="__main__": main()
