"""Screening A3: does token-radius correlate with LOCAL image properties?
Align the 8x8 token grid to 32x32 image patches; per token correlate token-radius with per-patch
edge/gradient, contrast, brightness, and DC-AE reconstruction error. GPU (decode) + CPU. Subsample.
"""
import glob, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
REPO=Path(__file__).resolve().parents[4]; RS=REPO/"experiments/image_latents/radial_semantics"; SF=0.41407
PATHS=sorted(glob.glob(str(REPO/"experiments/image_latents/data/imagenette2-320/**/*.JPEG"),recursive=True))

def patch_props(img256):  # (256,256,3) float -> per-token (8,8) props
    g=img256.mean(2); P=32; out={k:np.zeros((8,8)) for k in ["edge","contrast","brightness"]}
    gx=np.abs(np.diff(g,axis=1,prepend=g[:,:1])); gy=np.abs(np.diff(g,axis=0,prepend=g[:1,:])); grad=gx+gy
    for h in range(8):
        for w in range(8):
            p=g[h*P:(h+1)*P, w*P:(w+1)*P]
            out["edge"][h,w]=grad[h*P:(h+1)*P, w*P:(w+1)*P].mean()
            out["contrast"][h,w]=p.std(); out["brightness"][h,w]=p.mean()
    return out

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    R=torch.load(RS/"real_centered.pt",map_location="cpu"); data=R["data"]; mu=R["mu"]; te=R["te"]
    rng=np.random.default_rng(0); idx=np.sort(rng.choice(te.numpy(),400,replace=False))
    from diffusers import AutoencoderDC
    from torchvision import transforms
    ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256)])
    acc={k:[] for k in ["token_radius","edge","contrast","brightness","recon_err"]}
    with torch.no_grad():
        for j0 in range(0,len(idx),16):
            ids=idx[j0:j0+16]; zc=data[ids]                                   # centered
            raw=((zc+mu)/SF).to(dev).reshape(-1,32,8,8)
            rec=((ae.decode(raw).sample.clamp(-1,1)+1)/2).cpu().numpy()        # (b,3,256,256)
            Zt=zc.reshape(-1,32,8,8).permute(0,2,3,1).reshape(-1,8,8,32).norm(dim=3).numpy()  # token radius (b,8,8)
            for bi,i in enumerate(ids):
                orig=np.asarray(tf(Image.open(PATHS[i]).convert("RGB")),dtype=np.float32)/255.0
                pp=patch_props(orig)
                r=rec[bi].transpose(1,2,0)
                # per-token recon error (32x32 patch L2)
                re=np.zeros((8,8))
                for h in range(8):
                    for w in range(8):
                        re[h,w]=np.sqrt(((r[h*32:(h+1)*32,w*32:(w+1)*32]-orig[h*32:(h+1)*32,w*32:(w+1)*32])**2).mean())
                acc["token_radius"].append(Zt[bi].ravel()); acc["edge"].append(pp["edge"].ravel())
                acc["contrast"].append(pp["contrast"].ravel()); acc["brightness"].append(pp["brightness"].ravel())
                acc["recon_err"].append(re.ravel())
    for k in acc: acc[k]=np.concatenate(acc[k])
    tr=acc["token_radius"]
    corr={k:round(float(np.corrcoef(tr,acc[k])[0,1]),3) for k in ["edge","contrast","brightness","recon_err"]}
    out={"n_images":len(idx),"n_tokens":len(tr),"corr_tokenRadius_vs":corr,
         "note":"per-token correlation of token-radius with local image properties (pooled over tokens & images)"}
    (RS/"A3_perceptual.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2)); print("A3 DONE")

if __name__=="__main__": main()
