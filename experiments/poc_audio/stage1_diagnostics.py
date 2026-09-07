"""Stage-1 diagnostics (no retraining): is RAFM's content gap directional or an amplitude interaction?

1.1 gain-invariance: classify REAL clips scaled to low/mid/high real-radius quantiles -> acc by gain.
1.2 radius-controlled: regenerate UNet gen @24k for 4 methods; put RAFM & matched directions at a COMMON
    radius (real median), fixed_spherical directions at empirical radii; decode(istft)->re-STFT->classify;
    report acc natural vs radius-replaced, and by original energy quantile.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(REPO))
from audio_flow import make_model, build, FREQ, FRAMES, D
from audio_classifier import Clf
from audio_data import istft, stft
from audio_eval import rk4
dev="cuda" if torch.cuda.is_available() else "cpu"

def classify_dir(clf, X):  # X:(n,D) -> predicted digit (energy-invariant: classify direction, decode round-trip)
    n=X.shape[0]; out=[]
    with torch.no_grad():
        for i in range(0,n,256):
            x=X[i:i+256].reshape(-1,2,FREQ,FRAMES)
            wav=istft(x.cpu()); xr=stft(wav).to(dev)                      # decode -> re-STFT (faithful to protocol)
            xr=xr/xr.reshape(xr.shape[0],-1).norm(dim=1).clamp(min=1e-8).view(-1,1,1,1)
            out.append(clf(xr).argmax(1).cpu())
    return torch.cat(out)

def acc_by_quant(pred,y,en):
    qt=np.quantile(en,[1/3,2/3]); b=np.digitize(en,qt); r={}
    for k,nm in [(0,"lo"),(1,"mid"),(2,"hi")]:
        m=b==k; r[nm]=round(float((pred.numpy()[m]==y.numpy()[m]).mean()),3) if m.sum() else None
    return r

def main():
    clf=Clf().to(dev).eval(); clf.load_state_dict(torch.load(HERE/"digit_classifier.pt",map_location="cpu"))
    tr=torch.load(HERE/"data/audiomnist_stft_train.pt",map_location="cpu"); xtr=tr["x"].reshape(len(tr["x"]),-1).float()
    te=torch.load(HERE/"data/audiomnist_stft_test.pt",map_location="cpu"); gte=te["g"].numpy()
    Rmed=float(np.median(gte)); qs={"q10":float(np.quantile(gte,.1)),"q50":Rmed,"q90":float(np.quantile(gte,.9))}
    out={"real_median_radius":round(Rmed,3),"gain_levels":{k:round(v,3) for k,v in qs.items()}}

    # ---- 1.1 gain-invariance on REAL clips ----
    N=1500; xt=te["x"].reshape(len(te["x"]),-1).float()[:N]; yt=te["digit"].long()[:N]
    s=xt/xt.norm(dim=1,keepdim=True)                                      # unit content
    inv={}
    for k,R in qs.items():
        pred=classify_dir(clf,(R*s).to(dev)); inv[k]=round(float((pred==yt).float().mean()),3)
    out["1.1_gain_invariance_real_acc"]=inv

    # ---- 1.2 radius-controlled direction comparison ----
    M=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]; n=1500
    emp_radii=torch.tensor(np.random.default_rng(0).choice(gte,n),dtype=torch.float32)  # shared empirical radii
    res={}
    for m in M:
        st=build(m,xtr,tr["digit"].long(),0)
        model=make_model("unet",96,0).to(dev).eval()
        model.load_state_dict(torch.load(HERE/f"runs_unet/{m}/ema_24000.pt",map_location="cpu")["ema"])
        torch.manual_seed(0); y=torch.arange(10).repeat_interleave(int(np.ceil(n/10)))[:n].to(dev)
        if m=="gaussian_euclidean": x0=st["sigma"]*torch.randn(n,D,device=dev)
        else:
            r=st["src"].sample(n,D).norm(dim=1,keepdim=True).to(dev); u0=torch.randn(n,D,device=dev); x0=r*u0/u0.norm(dim=1,keepdim=True)
        gen=torch.cat([rk4(model,x0[i:i+200],y[i:i+200],st["spherical"],40) for i in range(0,n,200)])
        en=gen.norm(dim=1).cpu().numpy(); u=gen/gen.norm(dim=1,keepdim=True)
        nat=classify_dir(clf,gen); a_nat=float((nat==y.cpu()).float().mean())
        # radius replacement: common median for all; also empirical-radii variant
        com=classify_dir(clf,(Rmed*u)); a_com=float((com==y.cpu()).float().mean())
        empv=classify_dir(clf,(emp_radii.to(dev).unsqueeze(1)*u)); a_emp=float((empv==y.cpu()).float().mean())
        res[m]={"acc_natural":round(a_nat,3),"acc_common_radius(Rmed)":round(a_com,3),"acc_empirical_radii":round(a_emp,3),
                "acc_by_orig_energy_quantile":acc_by_quant(nat,y.cpu(),en)}
        print(m,res[m],flush=True)
    out["1.2_radius_controlled"]=res
    (HERE/"stage1_diagnostics.json").write_text(json.dumps(out,indent=2))
    print("\n"+json.dumps(out,indent=2))

if __name__=="__main__": main()
