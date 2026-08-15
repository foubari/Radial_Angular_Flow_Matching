#!/usr/bin/env python3
"""Uncurated DC-AE qualitative grid for ALL main methods (no retraining, surviving EMAs).

Rows = methods (Gaussian FM, Matched-Eucl., Fixed-spherical, RAFM std, Angular RAFM).
Cols = ImageNette classes 0..9, ONE generated sample each. Identical sample_seed, identical
class labels, identical sampler (RK4, 25 steps = 100 evals), identical split_seed for all methods.
Uncurated: the first sample per class is shown; no selection.

EMA sources (seed 8925): baselines under dit_sit/seed_8925/runs/<m>/ema_40000.pt ;
Angular under dit_sit_s8925/runs/angular_rafm/ema_40000.pt.
Output: rebuttal_experiments/paper_assets/figs/fig_dcae_grid.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "third_party/SiT"))
from models import SiT
from dit_train_sit import build
from dit_eval_sit import rk4_sample, SF

SEED = 8925
NFE = 25
METHODS = [("gaussian_euclidean", "Gaussian FM", f"dit_sit/seed_{SEED}"),
           ("matched_euclidean", "Matched-Eucl.", f"dit_sit/seed_{SEED}"),
           ("fixed_spherical", "Fixed-spherical", f"dit_sit/seed_{SEED}"),
           ("rafm", "RAFM (std)", f"dit_sit/seed_{SEED}"),
           ("angular_rafm", "Angular RAFM (ours)", f"dit_sit_s{SEED}")]


@torch.no_grad()
def gen_method(method, run_root, lat, lab, ae, device):
    st = build(method, lat, lab, 0); D = lat.shape[1]; mu = st["mu"]
    model = SiT(input_size=8, patch_size=1, in_channels=32, hidden_size=384, depth=12,
                num_heads=6, num_classes=10, class_dropout_prob=0.1, learn_sigma=False).to(device).eval()
    ema = torch.load(REPO / "experiments/image_latents" / run_root / "runs" / method / "ema_40000.pt",
                     map_location="cpu")["ema"]
    model.load_state_dict(ema)
    torch.manual_seed(SEED); np.random.seed(SEED)
    y = torch.arange(10).to(device)          # one image per class, identical across methods
    if method == "gaussian_euclidean":
        x0 = torch.randn(10, D, device=device)
    else:
        r = st["src"].sample(10, D).norm(dim=1, keepdim=True).to(device)
        u0 = torch.randn(10, D, device=device); x0 = r * u0 / u0.norm(dim=1, keepdim=True)
    genlat = rk4_sample(model, x0, y, st["spherical"], NFE, method == "angular_rafm")
    z = ((genlat.to(device) + mu.to(device)) / SF).reshape(-1, 32, 8, 8)
    img = ((ae.decode(z).sample.clamp(-1, 1) + 1) / 2).cpu()   # (10,3,256,256)
    del model, z
    torch.cuda.empty_cache()
    return img


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lat = torch.load(REPO / "experiments/image_latents/data/dcae_latents_scaled.pt", map_location="cpu").float()
    lab = torch.load(REPO / "experiments/image_latents/data/dcae_labels.pt", map_location="cpu").long()
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
                                       torch_dtype=torch.float32).to(device).eval()
    imgs = {}
    for method, lbl, root in METHODS:
        imgs[method] = gen_method(method, root, lat, lab, ae, device)
        print(f"  decoded {method}", flush=True)

    nrow, ncol = len(METHODS), 10
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 0.95, nrow * 0.95))
    for r, (method, lbl, _) in enumerate(METHODS):
        for c in range(ncol):
            ax = axes[r][c]
            ax.imshow(imgs[method][c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"class {c}", fontsize=6)
            if c == 0:
                ax.set_ylabel(lbl, fontsize=7.5, rotation=90, va="center",
                              fontweight="bold" if method == "angular_rafm" else "normal")
    fig.suptitle(f"DC-AE ImageNette — uncurated class-conditional samples (seed {SEED}, RK4 25 steps=100 evals, CFG=1)",
                 fontsize=9, y=1.005)
    fig.tight_layout()
    out = REPO / "rebuttal_experiments/paper_assets/figs"
    fig.savefig(out / "fig_dcae_grid.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_dcae_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/'fig_dcae_grid.pdf'} + .png")


if __name__ == "__main__":
    main()
