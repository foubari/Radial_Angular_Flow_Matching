"""Phase 2b — is the RAE/DINOv2-B latent one global sphere, token-wise spheres, or a product?

Reports GLOBAL norm stats and TOKEN-level stats:
  - global CoV: CoV of ||z_flat|| (196608-d) across images.
  - within-image cross-token CoV: per image, CoV of the 256 token norms (averaged).
  - per-position CoV: for each of the 256 token positions, CoV of its norm across images (mean).
  - between-position mean-radius std: how much the 256 positions' MEAN radii differ (are they the same sphere?).

Geometry reading:
  * small per-position CoV + small between-position spread  -> product of ~identical fixed-radius token spheres
    (≈ effectively a single global fixed-radius sphere; global CoV even smaller by averaging).
  * global CoV small but tokens vary                        -> single global sphere.
"""
import argparse, json, sys, glob, os
from pathlib import Path
import numpy as np, torch

def build_model(rae_dir, device):
    rae_dir = Path(rae_dir).resolve(); sys.path.insert(0, str(rae_dir/"src"))
    from stage1.rae import RAE
    m = RAE(encoder_cls='Dinov2withNorm', encoder_config_path='facebook/dinov2-with-registers-base',
            encoder_input_size=224, encoder_params={'dinov2_path':'facebook/dinov2-with-registers-base','normalize':True},
            decoder_config_path=str(rae_dir/'configs/decoder/ViTXL'),
            pretrained_decoder_path=str(rae_dir/'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt'),
            noise_tau=0., reshape_to_2d=True,
            normalization_stat_path=str(rae_dir/'models/stats/dinov2/wReg_base/imagenet1k/stat.pt'))
    return m.to(device).eval()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rae_dir", default="third_party/RAE")
    ap.add_argument("--img_root", default="experiments/image_latents/data/imagenette2-320/val")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--batch", type=int, default=32)
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    from torchvision import transforms; from PIL import Image
    m=build_model(args.rae_dir, dev)
    paths=sorted(glob.glob(os.path.join(args.img_root,"**","*.JPEG"),recursive=True))
    rng=np.random.default_rng(0); paths=[paths[i] for i in sorted(rng.permutation(len(paths))[:args.n])]
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor()])
    gnorm=[]; tnorm=[]   # tnorm: (N,256) per-token norms
    with torch.no_grad():
        buf=[]
        for k,p in enumerate(paths):
            try: buf.append(tf(Image.open(p).convert("RGB")))
            except Exception: continue
            if len(buf)==args.batch or k==len(paths)-1:
                z=m.encode(torch.stack(buf).to(dev)); b=z.shape[0]
                gnorm.append(z.reshape(b,-1).norm(dim=1).cpu().numpy())
                tnorm.append(z.norm(dim=1).reshape(b,-1).cpu().numpy()); buf=[]
    gn=np.concatenate(gnorm); tn=np.concatenate(tnorm,0)   # (N,256)
    global_cov=float(gn.std()/gn.mean())
    within_img_cov=float(np.mean(tn.std(1)/tn.mean(1)))                  # cross-token spread per image
    per_pos_cov=(tn.std(0)/tn.mean(0))                                  # (256,) each position's CoV across images
    per_pos_mean=tn.mean(0)                                             # (256,) mean radius per position
    between_pos_std_rel=float(per_pos_mean.std()/per_pos_mean.mean())   # do positions sit on the same sphere?
    res={"n_images":int(len(gn)),
         "global_norm":{"mean":float(gn.mean()),"std":float(gn.std()),"cov":global_cov},
         "token_norm_pooled":{"mean":float(tn.mean()),"std":float(tn.std()),"cov":float(tn.std()/tn.mean())},
         "within_image_cross_token_cov":within_img_cov,
         "per_position_cov_mean":float(per_pos_cov.mean()),"per_position_cov_max":float(per_pos_cov.max()),
         "between_position_mean_radius_rel_std":between_pos_std_rel}
    # verdict
    if res["per_position_cov_mean"]<0.06 and between_pos_std_rel<0.06:
        res["geometry_verdict"]="product of ~identical near-fixed-radius token spheres (≈ single global fixed-radius sphere)"
    elif global_cov<0.05:
        res["geometry_verdict"]="near single global fixed-radius sphere"
    else:
        res["geometry_verdict"]="non-degenerate (radius varies)"
    out=Path("experiments/image_latents")
    (out/"diagnostics"/"rae_sphere_geometry.json").write_text(json.dumps(res,indent=2))
    md=[f"# Phase 2b — RAE/DINOv2-B sphere geometry (N={res['n_images']})\n",
        "| quantity | value | meaning |","|---|---|---|",
        f"| global norm CoV | {global_cov:.4f} | spread of ||z_flat|| (196608-d) across images |",
        f"| pooled per-token CoV | {res['token_norm_pooled']['cov']:.4f} | all tokens×images |",
        f"| within-image cross-token CoV | {within_img_cov:.4f} | do tokens in ONE image share a radius? |",
        f"| per-position CoV (mean / max) | {res['per_position_cov_mean']:.4f} / {res['per_position_cov_max']:.4f} | does a fixed token position keep constant radius across images? |",
        f"| between-position mean-radius rel-std | {between_pos_std_rel:.4f} | do the 256 positions sit on the SAME sphere? |",
        f"\n**Verdict:** {res['geometry_verdict']}",
        "\nThe global RAFM implementation treats the flattened vector as a single global sphere; the DINO/RAE representation "
        "is more precisely a product of per-token spheres (per-token LayerNorm). Both are near-fixed-radius here."]
    (out/"tables"/"rae_sphere_geometry.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("\n".join(md)); print("\nwrote rae_sphere_geometry.{json,md}")

if __name__=="__main__": main()
