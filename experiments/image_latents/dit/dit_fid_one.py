"""Decode ONE method @ ONE checkpoint -> grid + limited-sample FID vs a CACHED shared reference.

Strictly single-method, foreground, conservative (small decode batches, no DataLoader/workers,
explicit cleanup). Real InceptionV3 features are computed once and cached so every method uses the
IDENTICAL reference set + evaluator. FID uses torchvision InceptionV3 features (relative-only,
labeled small-sample, NOT FID-50k). Run one at a time, e.g.:
  .venv_img/Scripts/python.exe dit_fid_one.py --method rafm --step 20000 --n 2000
"""
import argparse, glob, os, json, time
from pathlib import Path
import numpy as np, torch

REAL_FEATS = "experiments/image_latents/dit/eval/_real_feats.npy"

def inception_features(imgs01, device, bs=32):
    from torchvision.models import inception_v3, Inception_V3_Weights
    net=inception_v3(weights=Inception_V3_Weights.DEFAULT,transform_input=False).to(device).eval(); net.fc=torch.nn.Identity()
    mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    f=[]
    with torch.no_grad():
        for i in range(0,len(imgs01),bs):
            x=torch.nn.functional.interpolate(imgs01[i:i+bs].to(device),size=(299,299),mode="bilinear",align_corners=False)
            f.append(net((x-mean)/std).cpu().numpy())
    del net; torch.cuda.empty_cache()
    return np.concatenate(f)

def fid(f1,f2):
    from scipy import linalg
    m1,m2=f1.mean(0),f2.mean(0); s1,s2=np.cov(f1,rowvar=False),np.cov(f2,rowvar=False)
    cov=linalg.sqrtm(s1@s2); cov=cov.real if np.iscomplexobj(cov) else cov
    return float(((m1-m2)**2).sum()+np.trace(s1+s2-2*cov))

def load_real(root,n,size=256,seed=1):
    from torchvision import transforms; from PIL import Image
    ps=sorted(glob.glob(os.path.join(root,"**","*.JPEG"),recursive=True))
    rng=np.random.default_rng(seed); ps=[ps[i] for i in sorted(rng.permutation(len(ps))[:n])]
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(size),transforms.ToTensor()])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in ps])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True); ap.add_argument("--step",type=int,default=20000)
    ap.add_argument("--n",type=int,default=2000); ap.add_argument("--real_n",type=int,default=2000)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--img_root",default="experiments/image_latents/data/imagenette2-320/val")
    a=ap.parse_args()
    torch.set_num_threads(2)
    dev="cuda" if torch.cuda.is_available() else "cpu"; t0=time.time()
    # cached shared real reference
    if os.path.exists(REAL_FEATS):
        fr=np.load(REAL_FEATS); print(f"[real] loaded cached features {fr.shape}",flush=True)
    else:
        fr=inception_features(load_real(a.img_root,a.real_n),dev); np.save(REAL_FEATS,fr)
        print(f"[real] computed+cached features {fr.shape}",flush=True)
    # decode this method @ this step, small batches
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
    rd=Path("experiments/image_latents/dit/eval")/a.method
    latc=torch.load(rd/f"gen_{a.step}.pt",map_location="cpu")[:a.n]; mu=torch.load(rd/"mu.pt",map_location="cpu")
    ims=[]
    with torch.no_grad():
        for i in range(0,len(latc),a.batch):
            z=(latc[i:i+a.batch]+mu).to(dev).reshape(-1,32,8,8)
            ims.append(((ae.decode(z).sample.clamp(-1,1)+1)/2).cpu()); del z
    ims=torch.cat(ims); del ae; torch.cuda.empty_cache()
    from torchvision.utils import save_image
    gdir=Path("experiments/image_latents/figures/dit_samples"); gdir.mkdir(parents=True,exist_ok=True)
    save_image(ims[:16].clamp(0,1), gdir/f"{a.method}_{a.step}.png", nrow=4)
    fg=inception_features(ims,dev); val=round(fid(fg,fr),2)
    rt=round(time.time()-t0,1)
    (rd/"fid_pilot.json").write_text(json.dumps({"step":a.step,"n":a.n,"real_n":a.real_n,"batch":a.batch,"fid":val,"runtime_s":rt},indent=2))
    print(f"[{a.method} step {a.step}] small-FID {val} | n {a.n} real {a.real_n} batch {a.batch} | {rt}s | img range [{float(ims.min()):.2f},{float(ims.max()):.2f}]",flush=True)
    del ims,fg; torch.cuda.empty_cache()

if __name__=="__main__": main()
