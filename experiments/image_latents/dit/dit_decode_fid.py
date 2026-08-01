"""Decode DiT-generated DC-AE latents (per checkpoint) -> images -> small-sample FID vs real.

venv (diffusers + torchvision). Reuses torchvision-InceptionV3 feature FID (relative-only,
labeled small-sample, NOT FID-50k). Reads experiments/image_latents/dit/eval/<method>/gen_<step>.pt
(+ mu.pt), decodes, computes FID vs Imagenette val, saves fid_<step> + a sample grid for the last step.
"""
import argparse, glob, os, json, re
from pathlib import Path
import numpy as np, torch

def inception_features(imgs01, device, bs=50):
    from torchvision.models import inception_v3, Inception_V3_Weights
    net=inception_v3(weights=Inception_V3_Weights.DEFAULT,transform_input=False).to(device).eval(); net.fc=torch.nn.Identity()
    mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    f=[]
    with torch.no_grad():
        for i in range(0,len(imgs01),bs):
            x=torch.nn.functional.interpolate(imgs01[i:i+bs].to(device),size=(299,299),mode="bilinear",align_corners=False)
            f.append(net((x-mean)/std).cpu().numpy())
    return np.concatenate(f)

def fid(f1,f2):
    from scipy import linalg
    m1,m2=f1.mean(0),f2.mean(0); s1,s2=np.cov(f1,rowvar=False),np.cov(f2,rowvar=False)
    cov=linalg.sqrtm(s1@s2); cov=cov.real if np.iscomplexobj(cov) else cov
    return float(((m1-m2)**2).sum()+np.trace(s1+s2-2*cov))

def load_real(root,n,seed=1):
    from torchvision import transforms; from PIL import Image
    ps=sorted(glob.glob(os.path.join(root,"**","*.JPEG"),recursive=True))
    rng=np.random.default_rng(seed); ps=[ps[i] for i in sorted(rng.permutation(len(ps))[:n])]
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor()])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in ps])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True); ap.add_argument("--evroot",default="experiments/image_latents/dit/eval")
    ap.add_argument("--img_root",default="experiments/image_latents/data/imagenette2-320/val")
    ap.add_argument("--n",type=int,default=2000)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
    fr=inception_features(load_real(a.img_root,a.n),dev)
    mroot=Path(a.evroot)/a.method; mu=torch.load(mroot/"mu.pt",map_location="cpu")
    steps=sorted(int(re.search(r"gen_(\d+).pt",p).group(1)) for p in glob.glob(str(mroot/"gen_*.pt")))
    res={}
    for stp in steps:
        latc=torch.load(mroot/f"gen_{stp}.pt",map_location="cpu")[:a.n]
        with torch.no_grad():
            z=(latc+mu).to(dev).reshape(-1,32,8,8); ims=[]
            for i in range(0,z.shape[0],64): ims.append(((ae.decode(z[i:i+64]).sample.clamp(-1,1)+1)/2).cpu())
        ims=torch.cat(ims); res[stp]=round(fid(inception_features(ims,dev),fr),2)
        print(f"  {a.method} step {stp} small-FID {res[stp]}",flush=True)
        if stp==steps[-1]:
            from torchvision.utils import save_image
            (Path("experiments/image_latents/figures/dit_samples")).mkdir(parents=True,exist_ok=True)
            save_image(ims[:16].clamp(0,1),f"experiments/image_latents/figures/dit_samples/{a.method}.png",nrow=4)
    (mroot/"fid_traj.json").write_text(json.dumps(res,indent=2))
    print(json.dumps(res))

if __name__=="__main__": main()
