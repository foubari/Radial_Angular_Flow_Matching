"""LibriSpeech word-conditioned data: forced-align dev-clean, pick top-N frequent words, extract 1s clips
centered on each word occurrence, apply the SAME bimodal gain (x=g*s, ||x||=g). Reuses AudioMNIST STFT.

Output: data/libri/libri_word_{train,test}.pt with {x:(N,2,129,63), word:(N,), g:(N,)} + word_list.json.
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, torch, torchaudio
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from audio_data import stft, istft, sample_g, SR, LEN, N_FFT, HOP, FREQ, FRAMES
ROOT=HERE/"data"/"libri"; OUT=HERE/"data"; dev="cuda" if torch.cuda.is_available() else "cpu"
N_WORDS=20; MIN_DUR=0.12; MAX_DUR=0.9; MAX_PER_WORD=1000

def main():
    ds=torchaudio.datasets.LIBRISPEECH(str(ROOT),url="dev-clean",download=True)
    from torchaudio.pipelines import MMS_FA as B
    model=B.get_model().to(dev); tok=B.get_tokenizer(); aln=B.get_aligner()
    rs=torchaudio.transforms.Resample(16000,SR)
    # pass 1: align all utts, keep 8k waveform + word (mid_s,dur) list
    utt_wav=[]; utt_words=[]; freq=Counter()
    print(f"aligning {len(ds)} utts ...",flush=True)
    for i in range(len(ds)):
        wav,sr,txt,*_=ds[i]; words=[w.lower() for w in txt.split() if w.isalpha()]
        if not words: utt_wav.append(None); utt_words.append([]); continue
        try:
            with torch.inference_mode(): em,_=model(wav.to(dev))
            spans=aln(em[0],tok(words)); ratio=wav.shape[1]/em.shape[1]/sr
        except Exception: utt_wav.append(None); utt_words.append([]); continue
        w8=rs(wav)[0]; utt_wav.append(w8)
        wl=[]
        for w,sp in zip(words,spans):
            st=sp[0].start*ratio; en=sp[-1].end*ratio; d=en-st
            if MIN_DUR<=d<=MAX_DUR: wl.append((w,(st+en)/2)); freq[w]+=1
        utt_words.append(wl)
        if i%500==0: print(f"  {i}/{len(ds)}",flush=True)
    top=[w for w,_ in freq.most_common(N_WORDS)]; widx={w:k for k,w in enumerate(top)}
    print("chosen words:",top,flush=True); print("freqs:",[freq[w] for w in top],flush=True)
    # pass 2: extract 1s clips centered on each chosen-word occurrence
    per=defaultdict(int); X=[]; Y=[]
    for w8,wl in zip(utt_wav,utt_words):
        if w8 is None: continue
        for w,mid in wl:
            if w not in widx or per[w]>=MAX_PER_WORD: continue
            c=int(mid*SR); a=c-LEN//2; b=a+LEN
            seg=torch.zeros(LEN)
            lo=max(0,a); hi=min(len(w8),b); seg[lo-a:hi-a]=w8[lo:hi]
            S=stft(seg.unsqueeze(0))[0]; S=S/S.norm().clamp(min=1e-8)
            X.append(S); Y.append(widx[w]); per[w]+=1
    X=torch.stack(X); Y=torch.tensor(Y,dtype=torch.long); n=len(X)
    rng=np.random.default_rng(0); perm=torch.tensor(rng.permutation(n)); ntr=int(n*0.85)
    for split,idx,seed in [("train",perm[:ntr],1),("test",perm[ntr:],2)]:
        xi=X[idx]; yi=Y[idx]; g=sample_g(len(idx),seed=seed); x=xi*g.view(-1,1,1,1)
        torch.save({"x":x,"word":yi,"g":g},OUT/f"libri_word_{split}.pt")
        print(f"[{split}] x {tuple(x.shape)} n {len(idx)} energy~g {float(x.reshape(len(idx),-1).norm(dim=1).mean()):.3f}/{float(g.mean()):.3f}",flush=True)
    (OUT/"libri_word_list.json").write_text(json.dumps({"words":top,"n_words":len(top),"counts":[freq[w] for w in top]}))
    # round-trip
    d=torch.load(OUT/"libri_word_test.pt"); x=d["x"][:32]; g=d["g"][:32]
    rt=(stft(istft(x))-x).norm()/x.norm(); en=x.reshape(32,-1).norm(dim=1)
    print(f"ROUND-TRIP rel-err {rt:.2e} | energy==g max|dev| {float((en-g).abs().max()):.2e}",flush=True)
    print("LIBRI DATA DONE")

if __name__=="__main__": main()
