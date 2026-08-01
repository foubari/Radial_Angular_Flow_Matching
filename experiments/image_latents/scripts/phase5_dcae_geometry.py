"""Phase 5 — DC-AE latent geometry (non-degenerate radial control).

Same protocol as phase2_latent_geometry but for the DC-AE (deep compression autoencoder)
latent, which — unlike the near-fixed-radius RAE/DINOv2-B latent — is expected to have a
non-degenerate radial law. If confirmed, this is where matched-radial (RAFM) can be tested
against fixed-radius spherical flow.

venv: experiments/image_latents/.venv_img
"""
import argparse, json, glob, os
from pathlib import Path
import numpy as np, torch

def load_images(root, n, size=256):
    from torchvision import transforms
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(root, "**", "*.JPEG"), recursive=True))
    rng = np.random.default_rng(0); idx = sorted(rng.permutation(len(paths))[:n])
    paths = [paths[i] for i in idx]
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    imgs, labels = [], []
    for p in paths:
        try: imgs.append(tf(Image.open(p).convert("RGB"))); labels.append(Path(p).parent.name)
        except Exception: pass
    return torch.stack(imgs), labels

def stats(x):
    from scipy.stats import skew, kurtosis
    x=np.asarray(x,float)
    return dict(mean=float(x.mean()),std=float(x.std()),cov=float(x.std()/(abs(x.mean())+1e-12)),
               skew=float(skew(x)),kurt=float(kurtosis(x)),q01=float(np.quantile(x,.01)),
               q50=float(np.quantile(x,.5)),q99=float(np.quantile(x,.99)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    ap.add_argument("--img_root", default="experiments/image_latents/data/imagenette2-320/val")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--outdir", default="experiments/image_latents")
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    out=Path(args.outdir)
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
    imgs,labels=load_images(args.img_root,args.n)
    print(f"loaded {len(imgs)} images on {dev}",flush=True)
    gn,tn_all,labs=[],[],[]
    with torch.no_grad():
        for i in range(0,len(imgs),args.batch):
            x=(imgs[i:i+args.batch].to(dev)*2-1)
            z=ae.encode(x).latent          # (b,32,8,8)
            b=z.shape[0]
            gn.append(z.reshape(b,-1).norm(dim=1).cpu().numpy())
            tn_all.append(z.norm(dim=1).reshape(b,-1).cpu().numpy())
            labs+=labels[i:i+b]
            print(f"  {i+b}/{len(imgs)}",end="\r",flush=True)
    gn=np.concatenate(gn); tn=np.concatenate(tn_all,0).reshape(-1)
    labs=np.array(labs)
    res={"model":args.model,"n_images":int(len(gn)),"latent_shape":[32,8,8],
         "global_norm":stats(gn),"per_token_norm":stats(tn)}
    cc={c:{"mean":float(gn[labs==c].mean()),"std":float(gn[labs==c].std())} for c in sorted(set(labs.tolist()))}
    res["class_conditional_global_norm"]=cc
    res["between_class_radius_std"]=float(np.std([cc[c]["mean"] for c in cc]))
    res["within_class_radius_std_mean"]=float(np.mean([cc[c]["std"] for c in cc]))
    # decoder radius sensitivity
    scales=[0.50,0.75,0.90,1.00,1.10,1.25,1.50]
    with torch.no_grad():
        x=(imgs[:8].to(dev)*2-1); z=ae.encode(x).latent; rec0=ae.decode(z).sample
        sens={}
        for s in scales:
            rec=ae.decode(z*s).sample
            mse=torch.mean((rec-rec0)**2).item(); psnr=10*np.log10(4.0/(mse+1e-12))  # range [-1,1] -> peak 2 -> 2^2
            sens[s]={"psnr_vs_unscaled_dB":float(psnr)}
    res["decoder_radius_sensitivity"]=sens
    (out/"diagnostics"/"dcae_geometry.json").write_text(json.dumps(res,indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].hist(gn,bins=60,color="seagreen"); ax[0].set_title(f"DC-AE global norm (CoV={res['global_norm']['cov']:.3f})"); ax[0].set_xlabel("||z|| (2048-d)")
    ax[1].hist(tn,bins=60,color="purple"); ax[1].set_title(f"DC-AE per-token norm (CoV={res['per_token_norm']['cov']:.3f})")
    xs=list(sens); ax[2].plot(xs,[sens[s]['psnr_vs_unscaled_dB'] for s in xs],"o-"); ax[2].axvline(1,ls=":",c="k"); ax[2].set_title("DC-AE decoder PSNR vs radius scale"); ax[2].set_xlabel("radius scale")
    plt.tight_layout(); plt.savefig(out/"figures"/"phase5_dcae_geometry.png",dpi=130)
    g=res["global_norm"]; t=res["per_token_norm"]
    md=[f"# Phase 5 — DC-AE latent geometry (Imagenette val, N={res['n_images']}, latent [32,8,8]=2048-d)\n",
        "| quantity | mean | std | CoV | skew | kurt | q01 | q99 |","|---|---|---|---|---|---|---|---|",
        f"| global norm | {g['mean']:.2f} | {g['std']:.2f} | {g['cov']:.4f} | {g['skew']:.2f} | {g['kurt']:.2f} | {g['q01']:.1f} | {g['q99']:.1f} |",
        f"| per-token norm | {t['mean']:.2f} | {t['std']:.2f} | {t['cov']:.4f} | {t['skew']:.2f} | {t['kurt']:.2f} | {t['q01']:.1f} | {t['q99']:.1f} |",
        f"\n- between-class radius std {res['between_class_radius_std']:.3f} vs within-class {res['within_class_radius_std_mean']:.3f}",
        "\n## Decoder radius sensitivity (PSNR vs unscaled, dB)",
        "| scale | "+" | ".join(str(s) for s in scales)+" |","|"+"---|"*(len(scales)+1),
        "| PSNR | "+" | ".join(f"{sens[s]['psnr_vs_unscaled_dB']:.1f}" for s in scales)+" |"]
    (out/"tables"/"dcae_geometry.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("\n".join(md)); print("\nwrote dcae_geometry.{json,md,png}")

if __name__=="__main__": main()
