"""PoC-B B3 evaluation for ONE method. Run after the sweep + classifier (GPU).

Samples digit-conditioned (CFG=1, RK4, tangent-projected for spherical), then evaluates:
  * ENERGY/tails: gen ||.|| vs test g -> radial W1/KS, tail coverage >q90/q95/q99, low <q10, PIT;
  * CONTENT: energy-invariant digit classifier accuracy of generations (direction), and per-energy-quantile;
  * spectrogram + waveform grids from low / mid / high energy generations (uncurated).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(HERE))
from audio_flow import VelNet, make_model, build, FREQ, FRAMES, D
from audio_classifier import Clf
from audio_data import istft, SR
from rafm.flow_matching.sampler import _project_tangent

@torch.no_grad()
def rk4(model,x0,y,spherical,nfe=40,angular=False):
    dev=x0.device; x=x0; n=x0.shape[0]; dt=1.0/nfe
    def v(xx,tt):
        out=model(xx.reshape(n,2,FREQ,FRAMES),torch.full((n,),tt,device=dev),y).reshape(n,-1)
        if angular: out=xx.norm(dim=1,keepdim=True)*out          # reconstruct v=||x||*A
        return _project_tangent(out,xx) if spherical else out
    for i in range(nfe):
        t0=i*dt; k1=v(x,t0); k2=v(x+dt/2*k1,t0+dt/2); k3=v(x+dt/2*k2,t0+dt/2); k4=v(x+dt*k3,t0+dt)
        x=x+dt/6*(k1+2*k2+2*k3+k4)
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--method",required=True); ap.add_argument("--step",type=int,default=24000)
    ap.add_argument("--run_dir",default="runs_unet"); ap.add_argument("--n",type=int,default=2000); ap.add_argument("--nfe",type=int,default=40); ap.add_argument("--sbatch",type=int,default=128)
    ap.add_argument("--dataset",default="audiomnist_stft"); ap.add_argument("--ncls",type=int,default=10); ap.add_argument("--clf",default="digit_classifier.pt")
    ap.add_argument("--whisper",type=int,default=0); ap.add_argument("--nwhisper",type=int,default=800); a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr=torch.load(HERE/f"data/{a.dataset}_train.pt",map_location="cpu"); xtr=tr["x"].reshape(len(tr["x"]),-1).float()
    te=torch.load(HERE/f"data/{a.dataset}_test.pt",map_location="cpu"); gte=te["g"].numpy()
    ytr=(tr["word"] if "word" in tr else tr["digit"]).long()
    st=build(a.method,xtr,ytr,0)
    rd=HERE/a.run_dir/a.method; meta=json.loads((rd/"meta.json").read_text())["args"]  # arch/ch/angular of this run
    angular=bool(meta.get("angular",False))
    model=make_model(meta.get("arch","velnet"),meta.get("ch",64),meta.get("depth",5),meta.get("ncls",10)).to(dev).eval()
    model.load_state_dict(torch.load(rd/f"ema_{a.step}.pt",map_location="cpu")["ema"])
    clf=Clf(ncls=a.ncls).to(dev).eval(); clf.load_state_dict(torch.load(HERE/a.clf,map_location="cpu"))
    torch.manual_seed(0); y=torch.arange(a.ncls).repeat_interleave(int(np.ceil(a.n/a.ncls)))[:a.n].to(dev)
    if a.method=="gaussian_euclidean": x0=st["sigma"]*torch.randn(a.n,D,device=dev)
    else:
        r=st["src"].sample(a.n,D).norm(dim=1,keepdim=True).to(dev); u0=torch.randn(a.n,D,device=dev); x0=r*u0/u0.norm(dim=1,keepdim=True)
    SB=a.sbatch; gen=torch.cat([rk4(model,x0[i:i+SB],y[i:i+SB],st["spherical"],a.nfe,angular) for i in range(0,a.n,SB)])
    en=gen.norm(dim=1).cpu().numpy()                                        # generated energy
    # energy / tails
    q90,q95,q99,q10=[float(np.quantile(gte,q)) for q in (0.90,0.95,0.99,0.10)]
    pit=np.searchsorted(np.sort(gte),en)/len(gte)
    energy={"radial_w1":round(float(wasserstein_distance(en,gte)),4),"ks":round(float(ks_2samp(en,gte).statistic),4),
        "cov_gt_q90":round(float((en>q90).mean()),4),"cov_gt_q95":round(float((en>q95).mean()),4),"cov_gt_q99":round(float((en>q99).mean()),4),
        "cov_lt_q10":round(float((en<q10).mean()),4),"pit_mean":round(float(pit.mean()),4),
        "gen_energy_mean":round(float(en.mean()),3),"data_energy_mean":round(float(gte.mean()),3)}
    # content: energy-invariant digit classifier on generated direction
    with torch.no_grad():
        dirg=(gen/gen.norm(dim=1,keepdim=True)).reshape(-1,2,FREQ,FRAMES)
        pred=clf(dirg.to(dev)).argmax(1).cpu(); conf=clf(dirg.to(dev)).softmax(1).max(1).values.mean().item()
    acc=float((pred==y.cpu()).float().mean())
    # accuracy per energy tercile
    qt=np.quantile(en,[1/3,2/3]); bin=np.digitize(en,qt); accbin={}
    for b,name in [(0,"low_energy"),(1,"mid_energy"),(2,"high_energy")]:
        m=bin==b; accbin[name]=round(float((pred.numpy()[m]==y.cpu().numpy()[m]).mean()),3) if m.sum() else None
    content={"digit_acc":round(acc,3),"mean_confidence":round(conf,3),"acc_by_energy":accbin}
    if a.whisper:  # primary content metric for LibriSpeech: pretrained Whisper word-presence (gain-robust)
        import json as _json, torchaudio
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        words=_json.loads((HERE/"data/libri_word_list.json").read_text())["words"]
        proc=WhisperProcessor.from_pretrained("openai/whisper-tiny.en"); wm=WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(dev).eval()
        up=torchaudio.transforms.Resample(SR,16000); nw=min(a.nwhisper,gen.shape[0]); hit=0
        with torch.no_grad():
            for i in range(0,nw,16):
                X=gen[i:i+16].reshape(-1,2,FREQ,FRAMES); wav16=up(istft(X.cpu()))
                feat=proc(wav16.numpy(),sampling_rate=16000,return_tensors="pt").input_features.to(dev)
                ids=wm.generate(feat,max_new_tokens=32); txt=proc.batch_decode(ids,skip_special_tokens=True)
                for j,t in enumerate(txt):
                    if words[int(y[i+j])] in t.lower().split(): hit+=1
        content["word_presence"]=round(hit/nw,3); del wm; torch.cuda.empty_cache()
    out={"method":a.method,"step":a.step,"n":a.n,"energy":energy,"content":content}
    evname=a.method if a.run_dir=="runs_unet" else f"{a.run_dir}_{a.method}"   # keep runs separate (angular/seeds don't clobber std)
    ev=HERE/"eval_unet"/evname/f"step{a.step}"; ev.mkdir(parents=True,exist_ok=True); (ev/"eval.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)
    # spectrogram + wav grids low/mid/high energy
    order=np.argsort(en); pick={"low":order[:3],"mid":order[len(en)//2-1:len(en)//2+2],"high":order[-3:]}
    fig,ax=plt.subplots(3,3,figsize=(9,7))
    for ri,(tag,ids) in enumerate(pick.items()):
        for ci,idx in enumerate(ids):
            X=gen[idx].reshape(2,FREQ,FRAMES); mag=torch.log(torch.sqrt(X[0]**2+X[1]**2)+1e-4).cpu().numpy()
            ax[ri,ci].imshow(mag,origin="lower",aspect="auto",cmap="magma"); ax[ri,ci].set_title(f"{tag} E={en[idx]:.2f} d={int(y[idx])}",fontsize=8); ax[ri,ci].axis("off")
            wav=istft(X.unsqueeze(0).cpu())[0].numpy()
            import scipy.io.wavfile as wf; wf.write(ev/f"{tag}_{ci}_d{int(y[idx])}.wav",SR,(wav/np.abs(wav).max().clip(1e-6)*0.9*32767).astype(np.int16))
    plt.suptitle(f"{a.method}: generated spectrograms by energy",fontsize=10); plt.tight_layout()
    plt.savefig(ev/"spectrograms.png",dpi=130); print("saved",ev/"spectrograms.png")

if __name__=="__main__": main()
