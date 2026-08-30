"""Feasibility probe: download LibriSpeech dev-clean + test MMS_FA forced alignment on 1 utterance."""
import torch, torchaudio
from pathlib import Path
ROOT=Path(__file__).resolve().parent/"data"/"libri"; ROOT.mkdir(parents=True,exist_ok=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
print("downloading/loading LibriSpeech dev-clean ...",flush=True)
ds=torchaudio.datasets.LIBRISPEECH(str(ROOT),url="dev-clean",download=True)
print("n utterances:",len(ds),flush=True)
wav,sr,txt,spk,ch,utt=ds[0]
print("sr",sr,"wav",tuple(wav.shape),"dur %.2fs"%(wav.shape[1]/sr),"speaker",spk,flush=True)
print("transcript:",txt[:80],flush=True)
from torchaudio.pipelines import MMS_FA as B
model=B.get_model().to(dev); tok=B.get_tokenizer(); aln=B.get_aligner()
words=[w.lower() for w in txt.split() if w.isalpha()]
with torch.inference_mode(): em,_=model(wav.to(dev))
spans=aln(em[0],tok(words))
ratio=wav.shape[1]/em.shape[1]/sr
print("aligned %d words. first 5 (word, start_s, end_s):"%len(spans),flush=True)
for w,sp in list(zip(words,spans))[:5]:
    print("  %-12s %.2f-%.2f"%(w, sp[0].start*ratio, sp[-1].end*ratio),flush=True)
print("FEAS OK")
