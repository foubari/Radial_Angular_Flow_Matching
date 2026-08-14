"""Figure: Radial W1 bar chart across datasets and methods.

Grouped bar chart comparing radial_w1 (mean +/- std) for all methods
on each dataset. More visual than the results table.

Usage:
    python scripts/figure_radial_w1_bar.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHODS_ORDER = [
    "gaussian_fm",
    "source_only_empirical",
    "rafm_empirical",
    "rafm_empirical_no_proj",
    "msgm",
]

METHOD_LABELS = {
    "gaussian_fm": "Gaussian FM",
    "source_only_empirical": "Source-only",
    "rafm_empirical": "RAFM",
    "rafm_empirical_no_proj": "RAFM (no proj.)",
    "msgm": "MSGM",
}

METHOD_COLORS = {
    "gaussian_fm": "#228B22",
    "source_only_empirical": "#999999",
    "rafm_empirical": "#1f5faa",
    "rafm_empirical_no_proj": "#7fb3e0",
    "msgm": "#e67e22",
}

DATASETS_ORDER = [
    "student_t_d16_df3.0_cor",
    "student_t_d32_df3.0_cor",
    "gaussian_aniso_d16_cor",
    "piv_d64",
    "piv_d256",
]

DATASET_LABELS = {
    "student_t_d16_df3.0_cor": "Student-$t$\n$d=16$",
    "student_t_d32_df3.0_cor": "Student-$t$\n$d=32$",
    "gaussian_aniso_d16_cor": "Gaussian\n$d=16$",
    "piv_d64": "PIV\n$d=64$",
    "piv_d256": "PIV\n$d=256$",
}


def load_results(path: str = "outputs/results_summary.csv"):
    data = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            method = row["method"]
            if ds in DATASETS_ORDER and method in METHODS_ORDER:
                data[ds][method] = {
                    "mean": float(row["radial_w1_mean"]),
                    "std": float(row["radial_w1_std"]),
                }
    return data


def make_figure(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
    })

    data = load_results()

    # Filter to datasets that have at least 2 methods
    datasets = [ds for ds in DATASETS_ORDER if ds in data and len(data[ds]) >= 2]
    methods = [m for m in METHODS_ORDER if any(m in data[ds] for ds in datasets)]

    n_ds = len(datasets)
    n_m = len(methods)
    bar_width = 0.8 / n_m
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, method in enumerate(methods):
        means = []
        stds = []
        for ds in datasets:
            if method in data[ds]:
                means.append(data[ds][method]["mean"])
                stds.append(data[ds][method]["std"])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_m / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method],
                      capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets], fontsize=12)
    ax.set_ylabel(r"Radial $W_1$ $\downarrow$", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, framealpha=0.8, ncol=2, loc='upper left')
    ax.grid(True, axis='y', alpha=0.2)
    ax.set_title("Radial fidelity across datasets",
                 fontsize=15, fontweight='bold')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_radial_w1_bar.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"Radial W1 bar chart saved to {out_dir}/figure_radial_w1_bar.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
