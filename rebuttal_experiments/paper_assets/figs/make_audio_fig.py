#!/usr/bin/env python3
"""Figure B — AudioMNIST mechanism (3 panels), 3-seed uncertainty.

Message: standard RAFM preserves the radial law but full-velocity regression couples target
scale to radius; scale-free angular regression removes this dependence and improves
directional/content learning while preserving radial calibration.

Panels: (1) digit accuracy (content, ↑), (2) energy KS (calibration, ↓),
        (3) target norm vs radius — full-velocity scales, angular is flat (the mechanism).
Reads master_results.json (audio, 3-seed) + recomputes the audio target-norm curve (geometry only).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from theory_diagnostics import sample_quantities, binned  # reuse geometry

MASTER = json.load(open(REPO / "rebuttal_experiments/master_results.json"))["audio"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18,
})
C_ANG, C_RAFM, C_FULL = "#E76F51", "#0072B2", "#0072B2"
ORDER = [("gaussian", "Gaussian\nFM", "#8C8C8C"), ("matched", "Matched\nEucl.", "#B07AA1"),
         ("fixed_spher", "Fixed\nspher.", "#E9C46A"), ("std_rafm", "RAFM\n(std)", C_RAFM),
         ("angular_rafm", "Angular\n(ours)", C_ANG)]


def bars(ax, metric, ylabel, higher_better):
    labs, vals, errs, cols = [], [], [], []
    for k, lab, col in ORDER:
        cell = MASTER.get(k, {}).get(metric)
        if not cell:
            continue
        labs.append(lab); vals.append(cell["mean"]); errs.append(cell.get("std", 0)); cols.append(col)
    x = np.arange(len(labs))
    b = ax.bar(x, vals, 0.68, yerr=errs, color=cols, ecolor="#333", capsize=2.5,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel(ylabel)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.3, color="#333")
    arrow = "↑ better" if higher_better else "↓ better"
    ax.text(0.02, 0.96, arrow, transform=ax.transAxes, fontsize=7, va="top", color="#555")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    bars(axes[0], "digit_acc", "Digit accuracy", True)
    axes[0].set_title("Content (digit)")
    bars(axes[1], "energy_KS", "Energy KS", False)
    axes[1].set_title("Energy calibration")

    # panel 3: mechanism — target norm vs radius (AudioMNIST)
    d = torch.load(REPO / "experiments/poc_audio/data/audiomnist_stft_train.pt", map_location="cpu")
    x1 = d["x"].reshape(d["x"].shape[0], -1).float()
    r, un, ang = sample_quantities(x1, device, n=20000)
    ax = axes[2]
    binned(ax, r, un, C_FULL, r"full-vel $\|\dot X_t\|$ (std)")
    ax2 = ax.twinx()
    binned(ax2, r, ang, C_ANG, r"angular $\|\dot X_t\|/\|X_t\|$")
    ax.set_xscale("log")
    ax.set_xlabel(r"radius $\|X_t\|$ (energy)")
    ax.set_ylabel(r"full-vel norm", color=C_FULL)
    ax2.set_ylabel(r"angular norm", color=C_ANG)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(axis="y", labelcolor=C_FULL); ax2.tick_params(axis="y", labelcolor=C_ANG)
    ax.set_title("Target norm vs radius")
    ax.text(0.5, 0.06, f"corr: full {np.corrcoef(un, r)[0,1]:.2f}  |  ang {np.corrcoef(ang, r)[0,1]:.2f}",
            transform=ax.transAxes, ha="center", fontsize=6.5, color="#555")

    fig.suptitle("AudioMNIST: scale-free angular target improves content at equal energy calibration",
                 y=1.02, fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "fig_audiomnist_mechanism.pdf"); fig.savefig(HERE / "fig_audiomnist_mechanism.png", dpi=300)
    plt.close(fig)
    print("wrote fig_audiomnist_mechanism.{pdf,png}")
    print(f"  digit_acc: std_rafm={MASTER['std_rafm']['digit_acc']['mean']:.3f} "
          f"angular={MASTER['angular_rafm']['digit_acc']['mean']:.3f} | "
          f"energy_KS both ~{MASTER['angular_rafm']['energy_KS']['mean']:.3f}")


if __name__ == "__main__":
    main()
