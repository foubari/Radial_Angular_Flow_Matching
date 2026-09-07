"""DEFERRED (run after training). Generated vs data radial-tail calibration / coverage.

For each trained method (EMA @ a chosen step), sample latents (reuses dit_eval_sit's sampler), then
compare the GENERATED radial law to the DATA (test) radial law, focusing on the TAILS:
  * tail coverage: fraction of generated radii beyond the data q95/q99 (should match 0.05/0.01);
  * PIT/quantile calibration: map generated radii through the data-radius eCDF, check uniformity;
  * radial-W1 and KS overall (context).
This quantifies whether RAFM's matched-radial source actually reproduces the heavy/thin tail that a
fixed radius or a Gaussian source cannot. GPU (sampling only; no DC-AE decode needed for the radii).

Run per method after training, e.g.:  python deferred_tail_calibration.py --method rafm --step 40000 --seed 8925
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"third_party/SiT"))
sys.path.insert(0,str(REPO/"experiments/image_latents/dit"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True); ap.add_argument("--step",type=int,default=40000)
    ap.add_argument("--seed",type=int,default=8925); ap.add_argument("--n",type=int,default=4000); ap.add_argument("--nfe",type=int,default=50)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"
    from models import SiT
    from dit_train_sit import build
    from dit_eval_sit import rk4_sample
    lat=torch.load(REPO/"experiments/image_latents/data/dcae_latents_scaled.pt",map_location="cpu").float()
    lab=torch.load(REPO/"experiments/image_latents/data/dcae_labels.pt",map_location="cpu").long()
    st=build(a.method,lat,lab,0); D=lat.shape[1]; test=st["data"][st["te"]]
    model=SiT(input_size=8,patch_size=1,in_channels=32,hidden_size=384,depth=12,num_heads=6,
              num_classes=10,class_dropout_prob=0.1,learn_sigma=False).to(dev).eval()
    ema=torch.load(REPO/f"experiments/image_latents/dit_sit/seed_{a.seed}/runs/{a.method}/ema_{a.step}.pt",map_location="cpu")["ema"]
    model.load_state_dict(ema)
    torch.manual_seed(0); y=torch.arange(10).repeat_interleave(int(np.ceil(a.n/10)))[:a.n].to(dev)
    if a.method=="gaussian_euclidean": x0=torch.randn(a.n,D,device=dev)
    else:
        r=st["src"].sample(a.n,D).norm(dim=1,keepdim=True).to(dev); u0=torch.randn(a.n,D,device=dev); x0=r*u0/u0.norm(dim=1,keepdim=True)
    gen=torch.cat([rk4_sample(model,x0[i:i+256],y[i:i+256],st["spherical"],a.nfe).cpu() for i in range(0,a.n,256)])
    rg=gen.norm(dim=1).numpy(); rd=test.norm(dim=1).numpy()
    q95,q99=np.quantile(rd,0.95),np.quantile(rd,0.99)
    pit=np.searchsorted(np.sort(rd),rg)/len(rd)          # generated radii through data eCDF
    out={"method":a.method,"step":a.step,"seed":a.seed,"n":a.n,
         "tail_coverage_gt_q95":float((rg>q95).mean()),"target_0.05":0.05,
         "tail_coverage_gt_q99":float((rg>q99).mean()),"target_0.01":0.01,
         "pit_mean":float(pit.mean()),"pit_std":float(pit.std()),"pit_target_mean":0.5,
         "gen_radius_mean":float(rg.mean()),"data_radius_mean":float(rd.mean())}
    (REPO/f"experiments/image_latents/radial_semantics/tail_{a.method}_{a.seed}_{a.step}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2)); print("DONE tail calibration.")

if __name__=="__main__": main()
