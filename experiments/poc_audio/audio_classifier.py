"""PoC-B eval prep: train a small ENERGY-INVARIANT digit classifier on AudioMNIST STFT content.
Input = unit-norm STFT direction s = x/||x|| (energy removed), target = digit. Used to measure whether
generations preserve linguistic content regardless of their energy. GPU (run after the sweep).
"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE=Path(__file__).resolve().parent; FREQ,FRAMES=129,63

class Clf(nn.Module):
    """Energy-invariant digit classifier. Input is a 2-ch complex STFT (real,imag); the net first
    forms the LOG-MAGNITUDE (where the digit lives; phase is not aligned across clips)."""
    def __init__(s,ch=48,ncls=10):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,2,1),nn.GroupNorm(8,o),nn.SiLU())
        s.net=nn.Sequential(blk(1,ch),blk(ch,ch*2),blk(ch*2,ch*4),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(ch*4,ncls))
    def forward(s,x):
        mag=torch.sqrt(x[:,0]**2+x[:,1]**2+1e-12); lm=torch.log(mag+1e-4).unsqueeze(1)  # (B,1,F,T)
        lm=(lm-lm.mean(dim=(2,3),keepdim=True))/(lm.std(dim=(2,3),keepdim=True)+1e-5)     # per-sample norm (scale-invariant)
        return s.net(lm)

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/audiomnist_stft_train.pt")
    ap.add_argument("--ncls",type=int,default=10); ap.add_argument("--out",default="digit_classifier.pt")
    ap.add_argument("--steps",type=int,default=4000); A=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    d=torch.load(HERE/A.data,map_location="cpu"); x=d["x"].float(); y=(d["word"] if "word" in d else d["digit"]).long()
    s=(x/x.reshape(len(x),-1).norm(dim=1).clamp(min=1e-8).view(-1,1,1,1))          # unit-norm content
    g=torch.Generator().manual_seed(0); p=torch.randperm(len(s),generator=g); ntr=int(len(s)*0.9)
    tr,va=p[:ntr],p[ntr:]; Str,ytr=s[tr].to(dev),y[tr].to(dev); Sva,yva=s[va].to(dev),y[va].to(dev)
    m=Clf(ncls=A.ncls).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4); t0=time.time()
    for step in range(1,A.steps+1):
        gg=torch.Generator().manual_seed(step); idx=torch.randint(len(tr),(128,),generator=gg).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logit=m(Str[idx]); loss=nn.functional.cross_entropy(logit,ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step%1000==0:
            m.eval()
            with torch.no_grad():
                acc=(m(Sva).argmax(1)==yva).float().mean().item()
            m.train(); print(f"  step {step} loss {loss.item():.3f} val_acc {acc:.3f} {(time.time()-t0):.0f}s",flush=True)
    torch.save(m.state_dict(),HERE/A.out); print(f"val_acc final {acc:.3f} -> {A.out}")

if __name__=="__main__": main()
