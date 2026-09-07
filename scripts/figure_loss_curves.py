"""Figure: RAFM training loss curves across datasets.

Plots loss vs step for RAFM on each dataset, averaged over seeds
with std bands.

Usage:
    python scripts/figure_loss_curves.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASETS = [
    ("student_t_d16_df3.0_cor", "Student-$t$, $d=16$"),
    ("student_t_d32_df3.0_cor", "Student-$t$, $d=32$"),
    ("gaussian_aniso_d16_cor", "Gaussian, $d=16$"),
    ("piv_d64", "PIV, $d=64$"),
    ("piv_d256", "PIV, $d=256$"),
]


def load_logs(dataset: str, method: str = "rafm_empirical"):
    """Load train_log.csv for all seeds, return {step: [loss values]}."""
    base = Path(f"outputs/exp1_main_benchmark/{dataset}/{method}")
    all_seeds = defaultdict(list)
    for seed_dir in sorted(base.iterdir()):
        log_file = seed_dir / "train_log.csv"
        if not log_file.exists():
            continue
        with open(log_file) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                loss = float(row["loss"])
                all_seeds[step].append(loss)
    return all_seeds


def make_figure(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
    })

    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))

    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        logs = load_logs(ds_key)
        if not logs:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        steps = sorted(logs.keys())
        means = np.array([np.mean(logs[s]) for s in steps])
        stds = np.array([np.std(logs[s]) for s in steps])

        ax.plot(steps, means, color='#1f5faa', linewidth=1.5)
        ax.fill_between(steps, means - stds, means + stds,
                         color='#1f5faa', alpha=0.15)

        ax.set_xlabel("Step", fontsize=13)
        if ax == axes[0]:
            ax.set_ylabel("Loss", fontsize=13)
        ax.set_title(ds_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.15)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    fig.suptitle("RAFM training loss", fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_loss_curves.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"Loss curves saved to {out_dir}/figure_loss_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
