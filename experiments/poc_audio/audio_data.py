"""PoC-B B1: build reversible complex-STFT AudioMNIST tensors with a controlled bimodal gain.

x = g * s, where s = unit-norm complex-STFT of a spoken-digit clip (content = phonetics/speaker,
= DIRECTION) and g = energy/loudness (= global RADIUS), drawn from a bimodal heavy-tailed law so a
Gaussian source badly mismatches it (the pro-RAFM regime). Representation is a reversible complex STFT
(torch.stft/istft), stored as 2 channels (real, imag): shape (2, 129, 63). Global norm ||x||_F = g by
construction; ISTFT gives back the waveform at loudness g. Content and energy are decoupled.

Verifies STFT<->ISTFT round-trip and energy=g.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torchaudio
OUT=Path(__file__).resolve().parent/"data"; OUT.mkdir(parents=True,exist_ok=True)
SR_IN=16000; SR=8000; LEN=8000; N_FFT=256; HOP=128; FREQ=N_FFT//2+1; FRAMES=1+LEN//HOP  # 129 x 63
WIN=torch.hann_window(N_FFT)

def fix_len(w,L=LEN):
    t=w.shape[-1]
    if t<L: w=torch.nn.functional.pad(w,(0,L-t))
    else:
        s=(t-L)//2; w=w[...,s:s+L]
    return w

def stft(w):  # (B,LEN)->(B,2,FREQ,FRAMES)
    X=torch.stft(w,N_FFT,HOP,N_FFT,WIN,center=True,return_complex=True)  # (B,FREQ,FRAMES)
    return torch.stack([X.real,X.imag],dim=1)

def istft(x):  # (B,2,FREQ,FRAMES)->(B,LEN)
    X=torch.complex(x[:,0],x[:,1])
    return torch.istft(X,N_FFT,HOP,N_FFT,WIN,center=True,length=LEN)

def sample_g(n,seed):
    g=torch.Generator().manual_seed(seed)
    hi=torch.rand(n,generator=g)<0.4                      # 40% loud mode, 60% quiet
    mu=torch.where(hi,torch.log(torch.tensor(4.0)),torch.log(torch.tensor(1.0)))
    return torch.exp(mu+0.25*torch.randn(n,generator=g)).clamp(min=0.1)

def main():
    from datasets import load_dataset
    ds=load_dataset("gilkeyio/AudioMNIST",split="train")
    n=len(ds); rng=np.random.default_rng(0); perm=rng.permutation(n)
    ntr,nte=12000,3000; idx={"train":perm[:ntr],"test":perm[ntr:ntr+nte]}
    rs=torchaudio.transforms.Resample(SR_IN,SR)
    for split,ids in idx.items():
        specs=torch.empty(len(ids),2,FREQ,FRAMES); digs=torch.empty(len(ids),dtype=torch.long)
        for j,i in enumerate(ids):
            ex=ds[int(i)]; w=torch.tensor(ex["audio"]["array"],dtype=torch.float32)
            w=rs(w); w=fix_len(w); S=stft(w.unsqueeze(0))[0]                # (2,FREQ,FRAMES)
            S=S/S.norm().clamp(min=1e-8)                                    # unit-norm content (direction)
            specs[j]=S; digs[j]=int(ex["digit"])
            if j%2000==0: print(f"  {split} {j}/{len(ids)}",flush=True)
        g=sample_g(len(ids),seed=(1 if split=="train" else 2))
        x=specs*g.view(-1,1,1,1)                                            # x = g * s
        torch.save({"x":x,"digit":digs,"g":g},OUT/f"audiomnist_stft_{split}.pt")
        print(f"[{split}] x {tuple(x.shape)} energy(||x||) mean {x.reshape(len(ids),-1).norm(dim=1).mean():.3f} vs g mean {g.mean():.3f}",flush=True)
    # round-trip check on test
    d=torch.load(OUT/"audiomnist_stft_test.pt"); x=d["x"][:64]; g=d["g"][:64]
    wav=istft(x); re=stft(wav); rt=(re-x).norm()/x.norm()
    energy=x.reshape(64,-1).norm(dim=1)
    print(f"\nROUND-TRIP stft(istft(x)) rel-err {rt:.2e} | energy==g max|dev| {float((energy-g).abs().max()):.2e}")
    (OUT/"stft_config.json").write_text(json.dumps({"SR":SR,"LEN":LEN,"N_FFT":N_FFT,"HOP":HOP,"FREQ":FREQ,"FRAMES":FRAMES,
        "g":"bimodal lognormal modes~1.0/4.0 (40% loud), sigma0.25","D":2*FREQ*FRAMES}))
    print("B1 DONE",OUT)

if __name__=="__main__": main()
