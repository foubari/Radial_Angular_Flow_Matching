"""Test Whisper-tiny word-presence on REAL LibriSpeech clips (is it a stronger content probe than the
0.28 conv classifier?), and gain-invariance. Clips are 8kHz STFT -> istft -> upsample 16k -> Whisper."""
import sys, json
from pathlib import Path
import numpy as np, torch, torchaudio
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from audio_data import istft, SR
dev="cuda" if torch.cuda.is_available() else "cpu"
words=json.loads((HERE/"data/libri_word_list.json").read_text())["words"]
d=torch.load(HERE/"data/libri_word_test.pt",map_location="cpu"); x=d["x"]; y=d["word"]; g=d["g"]
from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc=WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model=WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(dev).eval()
up=torchaudio.transforms.Resample(SR,16000)
def transcribe(wav16):  # (B,16000)
    feat=proc(wav16.numpy(),sampling_rate=16000,return_tensors="pt").input_features.to(dev)
    with torch.no_grad(): ids=model.generate(feat,max_new_tokens=32)
    return [t.lower() for t in proc.batch_decode(ids,skip_special_tokens=True)]
def presence(idx, gain=None):
    hit=0
    for i in range(0,len(idx),16):
        b=idx[i:i+16]; X=x[b]
        if gain is not None: X=(X/X.reshape(len(b),-1).norm(dim=1).view(-1,1,1,1))*gain
        wav=istft(X); wav16=up(wav)
        txt=transcribe(wav16)
        for j,t in enumerate(txt):
            if words[y[b[j]]] in t.split(): hit+=1
    return hit/len(idx)
rng=np.random.default_rng(0); idx=rng.choice(len(x),120,replace=False)
print("REAL-clip Whisper word-presence (natural gain):", round(presence(idx),3),flush=True)
print("  @gain 0.5x :", round(presence(idx,0.5),3),flush=True)
print("  @gain 3.0x :", round(presence(idx,3.0),3),flush=True)
print("chance ~ 1/20 =",0.05,"| classifier ceiling 0.28")
