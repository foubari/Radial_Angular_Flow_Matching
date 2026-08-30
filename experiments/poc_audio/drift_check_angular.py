"""Verify radial drift is negligible for angular RAFM sampling (no state renormalization).
Draw source radius r (empirical), integrate the ODE with v=||x||*A + tangent projection, compare ||gen||
to r per-sample. Reports relative drift. Compares std-rafm vs angular-rafm."""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(REPO))
from audio_flow import make_model, build, D
from audio_eval import rk4
dev="cuda" if torch.cuda.is_available() else "cpu"
def check(run_dir, angular, n=800):
    tr=torch.load(HERE/"data/audiomnist_stft_train.pt",map_location="cpu"); x=tr["x"].reshape(len(tr["x"]),-1).float()
    st=build("rafm",x,tr["digit"].long(),0)
    model=make_model("unet",96,0).to(dev).eval(); model.load_state_dict(torch.load(HERE/f"{run_dir}/rafm/ema_24000.pt",map_location="cpu")["ema"])
    torch.manual_seed(1); y=torch.arange(10).repeat_interleave(int(np.ceil(n/10)))[:n].to(dev)
    r=st["src"].sample(n,D).norm(dim=1,keepdim=True).to(dev); u0=torch.randn(n,D,device=dev); x0=r*u0/u0.norm(dim=1,keepdim=True)
    gen=torch.cat([rk4(model,x0[i:i+200],y[i:i+200],st["spherical"],40,angular) for i in range(0,n,200)])
    r0=x0.norm(dim=1); rf=gen.norm(dim=1); d=((rf-r0)/r0).abs()
    return {"source_radius_mean":round(float(r0.mean()),3),"gen_radius_mean":round(float(rf.mean()),3),
            "rel_drift_mean_%":round(float(d.mean()*100),4),"rel_drift_max_%":round(float(d.max()*100),4)}
out={"std_rafm":check("runs_unet",False),"angular_rafm":check("runs_angular",True)}
(HERE/"drift_check.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
