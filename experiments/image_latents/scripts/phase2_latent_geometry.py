"""Phase 2 — RAE/DINOv2-B latent geometry diagnostics (the decisive gate).

Measures the norm distribution of the ACTUAL Stage-2 latent (the RAE-normalized
[768,16,16] tensor the generative model consumes), on real ImageNet-domain images
(Imagenette val). Reports global and per-token norm statistics, class-conditional
radius, and decoder sensitivity to radius scaling.

Decision rule (per prompt): if the Stage-2 latent has negligible radial variability,
it is a fixed-radius / angular-only control and RAFM should reduce to spherical flow.
If a non-degenerate decodable radial law exists, the full RAFM comparison is warranted.

Run: experiments/image_latents/.venv_img/Scripts/python.exe this_script.py --rae_dir third_party/RAE --n 2000
"""
import argparse, json, sys, os, glob
from pathlib import Path
import numpy as np
import torch

def build_model(rae_dir, device):
    rae_dir = Path(rae_dir).resolve()
    sys.path.insert(0, str(rae_dir / "src"))
    from stage1.rae import RAE
    m = RAE(encoder_cls='Dinov2withNorm',
            encoder_config_path='facebook/dinov2-with-registers-base',
            encoder_input_size=224,
            encoder_params={'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
            decoder_config_path=str(rae_dir / 'configs/decoder/ViTXL'),
            pretrained_decoder_path=str(rae_dir / 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt'),
            noise_tau=0., reshape_to_2d=True,
            normalization_stat_path=str(rae_dir / 'models/stats/dinov2/wReg_base/imagenet1k/stat.pt'))
    return m.to(device).eval()

def load_images(root, n, size=256):
    from torchvision import transforms
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(root, "**", "*.JPEG"), recursive=True))
    # class = parent dir name; deterministic subsample spread across classes
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(paths))[:n]
    paths = [paths[i] for i in sorted(idx)]
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    imgs, labels = [], []
    for p in paths:
        try:
            imgs.append(tf(Image.open(p).convert("RGB")))
            labels.append(Path(p).parent.name)
        except Exception:
            pass
    return torch.stack(imgs), labels

def stats(x):
    from scipy.stats import skew, kurtosis
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), std=float(x.std()), cov=float(x.std()/ (abs(x.mean())+1e-12)),
                skew=float(skew(x)), kurt=float(kurtosis(x)),
                q01=float(np.quantile(x,.01)), q50=float(np.quantile(x,.5)), q99=float(np.quantile(x,.99)),
                min=float(x.min()), max=float(x.max()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rae_dir", default="third_party/RAE")
    ap.add_argument("--img_root", default="experiments/image_latents/data/imagenette2-320/val")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--outdir", default="experiments/image_latents")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.outdir)
    (out/"diagnostics").mkdir(parents=True, exist_ok=True); (out/"tables").mkdir(parents=True, exist_ok=True)
    (out/"figures").mkdir(parents=True, exist_ok=True)

    m = build_model(args.rae_dir, dev)
    imgs, labels = load_images(args.img_root, args.n)
    print(f"loaded {len(imgs)} images on {dev}", flush=True)

    global_norms, token_norms_all, per_img_token_mean, per_img_token_std, keep_labels = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(imgs), args.batch):
            xb = imgs[i:i+args.batch].to(dev)
            z = m.encode(xb)                       # (b,768,16,16)
            b = z.shape[0]
            gnorm = z.reshape(b, -1).norm(dim=1)   # global flattened norm
            tnorm = z.norm(dim=1)                  # (b,16,16) per-token norms
            global_norms.append(gnorm.cpu().numpy())
            token_norms_all.append(tnorm.reshape(b, -1).cpu().numpy())
            per_img_token_mean.append(tnorm.reshape(b,-1).mean(1).cpu().numpy())
            per_img_token_std.append(tnorm.reshape(b,-1).std(1).cpu().numpy())
            keep_labels += labels[i:i+b]
            print(f"  {i+b}/{len(imgs)}", end="\r", flush=True)
    gn = np.concatenate(global_norms)
    tn = np.concatenate(token_norms_all, axis=0)   # (N, 256)
    tn_flat = tn.reshape(-1)

    res = {"n_images": int(len(gn)), "latent_shape": [768,16,16], "device": dev,
           "global_norm": stats(gn),
           "per_token_norm": stats(tn_flat),
           "per_image_token_norm_cov_mean": float((np.concatenate(per_img_token_std)/ (np.concatenate(per_img_token_mean)+1e-12)).mean())}
    # class-conditional global norm
    labs = np.array(keep_labels)
    cc = {}
    for c in sorted(set(keep_labels)):
        v = gn[labs==c]; cc[c] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    res["class_conditional_global_norm"] = cc
    res["between_class_radius_std"] = float(np.std([cc[c]["mean"] for c in cc]))
    res["within_class_radius_std_mean"] = float(np.mean([cc[c]["std"] for c in cc]))

    (out/"diagnostics"/"latent_geometry.json").write_text(json.dumps(res, indent=2))

    # ---- decoder sensitivity to radius scaling ----
    scales = [0.50,0.75,0.90,1.00,1.10,1.25,1.50]
    with torch.no_grad():
        xb = imgs[:8].to(dev)
        z = m.encode(xb)
        rec0 = m.decode(z)
        sens = {}
        for s in scales:
            recs = m.decode(z * s)
            mse = torch.mean((recs-rec0)**2).item()
            psnr = 10*np.log10(1.0/(mse+1e-12))
            sens[s] = {"mse_vs_unscaled": mse, "psnr_vs_unscaled_dB": float(psnr)}
    res["decoder_radius_sensitivity"] = sens
    (out/"diagnostics"/"latent_geometry.json").write_text(json.dumps(res, indent=2))

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].hist(gn,bins=60,color="steelblue"); ax[0].set_title(f"Global latent norm (CoV={res['global_norm']['cov']:.3f})"); ax[0].set_xlabel("||z|| (flattened 196608-d)")
    ax[1].hist(tn_flat,bins=60,color="darkorange"); ax[1].set_title(f"Per-token norm (CoV={res['per_token_norm']['cov']:.3f})"); ax[1].set_xlabel("||token|| (768-d)")
    xs=list(sens); ax[2].plot(xs,[sens[s]['psnr_vs_unscaled_dB'] for s in xs],"o-"); ax[2].axvline(1.0,ls=":",c="k"); ax[2].set_title("Decoder PSNR vs radius scale"); ax[2].set_xlabel("radius scale"); ax[2].set_ylabel("PSNR vs unscaled (dB)")
    plt.tight_layout(); plt.savefig(out/"figures"/"phase2_latent_geometry.png",dpi=130)

    # ---- table ----
    g=res["global_norm"]; t=res["per_token_norm"]
    md=[f"# Phase 2 — RAE/DINOv2-B Stage-2 latent geometry (Imagenette val, N={res['n_images']})\n",
        "| quantity | mean | std | CoV | skew | kurt | q01 | q99 |",
        "|---|---|---|---|---|---|---|---|",
        f"| global norm ||z|| (196608-d) | {g['mean']:.3f} | {g['std']:.3f} | {g['cov']:.4f} | {g['skew']:.2f} | {g['kurt']:.2f} | {g['q01']:.2f} | {g['q99']:.2f} |",
        f"| per-token norm (768-d) | {t['mean']:.3f} | {t['std']:.3f} | {t['cov']:.4f} | {t['skew']:.2f} | {t['kurt']:.2f} | {t['q01']:.2f} | {t['q99']:.2f} |",
        f"\n- between-class radius std (global norm): {res['between_class_radius_std']:.3f} ; within-class mean std: {res['within_class_radius_std_mean']:.3f}",
        f"- mean per-image token-norm CoV: {res['per_image_token_norm_cov_mean']:.4f}",
        "\n## Decoder radius-scaling sensitivity (PSNR vs unscaled reconstruction)",
        "| scale | "+" | ".join(f"{s}" for s in scales)+" |","|"+"---|"*(len(scales)+1),
        "| PSNR dB | "+" | ".join(f"{sens[s]['psnr_vs_unscaled_dB']:.1f}" for s in scales)+" |"]
    (out/"tables"/"latent_geometry.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print("\n".join(md))
    print("\nWrote diagnostics/latent_geometry.json, tables/latent_geometry.md, figures/phase2_latent_geometry.png")


if __name__ == "__main__":
    main()
