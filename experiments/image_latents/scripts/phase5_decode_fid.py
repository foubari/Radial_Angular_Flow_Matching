"""Phase 5 — decode generated DC-AE latents to images + small-sample FID sanity check.

For each method: un-center the generated latents (+train mu), reshape to [32,8,8], DC-AE decode,
save a fixed-order uncurated 16-sample grid, and compute a SMALL-SAMPLE FID vs real Imagenette
images. FID uses torchvision InceptionV3 pool features (NOT the canonical clean-fid Inception),
so absolute values are only comparable ACROSS these methods here — this is a labeled sanity
check, NOT FID-50k. Also reports the decoder rFID floor (decode real latents vs real images).

venv: experiments/image_latents/.venv_img
"""
import argparse, glob, os, json
from pathlib import Path
import numpy as np, torch

def inception_features(imgs01, device, bs=50):
    """imgs01: (N,3,H,W) in [0,1]. Returns (N,2048) InceptionV3 pool features."""
    import torchvision
    from torchvision.models import inception_v3, Inception_V3_Weights
    net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device).eval()
    net.fc = torch.nn.Identity()
    mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    feats=[]
    with torch.no_grad():
        for i in range(0,len(imgs01),bs):
            x=imgs01[i:i+bs].to(device)
            x=torch.nn.functional.interpolate(x,size=(299,299),mode="bilinear",align_corners=False)
            x=(x-mean)/std
            feats.append(net(x).cpu().numpy())
    return np.concatenate(feats)

def fid(f1,f2):
    from scipy import linalg
    mu1,mu2=f1.mean(0),f2.mean(0); s1,s2=np.cov(f1,rowvar=False),np.cov(f2,rowvar=False)
    covmean=linalg.sqrtm(s1@s2)
    if np.iscomplexobj(covmean): covmean=covmean.real
    return float(((mu1-mu2)**2).sum()+np.trace(s1+s2-2*covmean))

def load_real(root,n,size=256,seed=1):
    from torchvision import transforms; from PIL import Image
    paths=sorted(glob.glob(os.path.join(root,"**","*.JPEG"),recursive=True))
    rng=np.random.default_rng(seed); paths=[paths[i] for i in sorted(rng.permutation(len(paths))[:n])]
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(size),transforms.ToTensor()])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])

def save_grid(imgs01, path, nrow=4, n=16):
    from torchvision.utils import save_image
    save_image(imgs01[:n].clamp(0,1), path, nrow=nrow)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fourway", default="experiments/image_latents/fourway/raw")
    ap.add_argument("--seed", default="8925"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--img_root", default="experiments/image_latents/data/imagenette2-320/val")
    ap.add_argument("--out", default="experiments/image_latents")
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
    out=Path(args.out); (out/"figures"/"samples").mkdir(parents=True,exist_ok=True)

    real=load_real(args.img_root,args.n)
    fr=inception_features(real,dev)

    def decode(latc, mu, cap):
        z=(latc[:cap]+mu).to(dev).reshape(-1,32,8,8)
        imgs=[]
        with torch.no_grad():
            for i in range(0,z.shape[0],64):
                d=ae.decode(z[i:i+64]).sample; imgs.append(((d.clamp(-1,1)+1)/2).cpu())
        return torch.cat(imgs)

    methods=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm_empirical"]
    res={}
    for mth in methods:
        rd=Path(args.fourway)/mth/f"seed_{args.seed}"
        latc=torch.load(rd/"gen_latents.pt",map_location="cpu"); mu=torch.load(rd/"mu.pt",map_location="cpu")
        imgs=decode(latc, mu, args.n)
        save_grid(imgs, out/"figures"/"samples"/f"{mth}.png")
        res[mth]={"fid_small": round(fid(inception_features(imgs,dev), fr),2), "n": int(min(args.n,len(latc)))}
        print(f"  {mth:20} small-FID {res[mth]['fid_small']}", flush=True)
    # decoder rFID floor: decode REAL test latents (from full latent file, test split), compare to real images
    L=torch.load("experiments/image_latents/data/dcae_latents.pt",map_location="cpu").float()
    g=torch.Generator().manual_seed(0); perm=torch.randperm(len(L),generator=g); te=perm[int(len(L)*0.8):][:args.n]
    with torch.no_grad():
        zr=L[te].to(dev).reshape(-1,32,8,8); rimgs=[]
        for i in range(0,zr.shape[0],64): rimgs.append(((ae.decode(zr[i:i+64]).sample.clamp(-1,1)+1)/2).cpu())
    res["_decoder_rFID_floor"]={"fid_small":round(fid(inception_features(torch.cat(rimgs),dev),fr),2)}
    print(f"  decoder rFID floor {res['_decoder_rFID_floor']['fid_small']}", flush=True)
    (out/"fourway"/"fid_small.json").write_text(json.dumps(res,indent=2))
    md=["# Phase 5 — small-sample FID sanity check (DC-AE decoded), NOT FID-50k\n",
        f"torchvision-InceptionV3 features, N≈{args.n}/method, real ref = Imagenette val. Relative across methods only.\n",
        "| method | small-FID (lower=better) |","|---|---|"]
    for mth in methods: md.append(f"| {mth} | {res[mth]['fid_small']} |")
    md.append(f"| _decoder rFID floor_ | {res['_decoder_rFID_floor']['fid_small']} |")
    (out/"fourway"/"fid_small.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("\n".join(md)); print("\nwrote fid_small.{json,md} + figures/samples/*.png")

if __name__=="__main__": main()
