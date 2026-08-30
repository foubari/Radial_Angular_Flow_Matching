"""Target-norm diagnostic (CPU): standard-velocity target ||u_t|| vs angular target ||u_t||/||x_t||,
and their dependence on the sample radius ||x_t||. Shows why angular regression decouples training
difficulty from radius. Uses the rafm coupling (empirical radial source + spherical path)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="3"
import sys, json
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(REPO))
from audio_flow import build, D
def main():
    tr=torch.load(HERE/"data/audiomnist_stft_train.pt",map_location="cpu"); x=tr["x"].reshape(len(tr["x"]),-1).float()
    st=build("rafm",x,tr["digit"].long(),0); train=st["train"]
    g=torch.Generator().manual_seed(0); N=8000; idx=torch.randint(len(train),(N,),generator=g)
    x1=train[idx]; R=x1.norm(dim=1,keepdim=True); u0=torch.randn(N,D,generator=g); x0=R*u0/u0.norm(dim=1,keepdim=True)
    t=torch.rand(N,generator=g)
    xt=st["path"].sample_path(x0,x1,t); ut=st["path"].conditional_vector_field(x0,x1,t)
    xn=xt.norm(dim=1); un=ut.norm(dim=1); an=un/xn.clamp(min=1e-8)                 # std vs angular target norm
    def stats(v): v=v.numpy(); return {"mean":round(float(v.mean()),4),"std":round(float(v.std()),4),"cov":round(float(v.std()/v.mean()),4)}
    corr_std=float(np.corrcoef(xn.numpy(),un.numpy())[0,1]); corr_ang=float(np.corrcoef(xn.numpy(),an.numpy())[0,1])
    out={"std_velocity_target_norm":stats(un),"angular_target_norm":stats(an),
         "corr(radius, ||u_t||)":round(corr_std,3),"corr(radius, ||u_t||/||x_t||)":round(corr_ang,3),
         "note":"std-vel target norm scales with radius (high corr, high CoV) -> difficulty varies with amplitude; angular target norm ~radius-free (low corr) -> scale-free"}
    (HERE/"target_norm_diag.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
