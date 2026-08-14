"""Figure: Runtime comparison bar chart.

Compares training time across FM methods and MSGM.
Shows the massive speedup of RAFM over MSGM.

Usage:
    python scripts/figure_runtime.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHODS_ORDER = [
    "gaussian_fm",
    "rafm_empirical",
    "msgm",
]

METHOD_LABELS = {
    "gaussian_fm": "Gaussian FM",
    "rafm_empirical": "RAFM",
    "msgm": "MSGM",
}

METHOD_COLORS = {
    "gaussian_fm": "#228B22",
    "rafm_empirical": "#1f5faa",
    "msgm": "#e67e22",
}

DATASETS_ORDER = [
    "student_t_d16_df3.0_cor",
    "student_t_d32_df3.0_cor",
    "gaussian_aniso_d16_cor",
    "piv_d64",
]

DATASET_LABELS = {
    "student_t_d16_df3.0_cor": "Student-$t$\n$d=16$",
    "student_t_d32_df3.0_cor": "Student-$t$\n$d=32$",
    "gaussian_aniso_d16_cor": "Gaussian\n$d=16$",
    "piv_d64": "PIV\n$d=64$",
}


def load_results(path: str = "outputs/results_summary.csv"):
    data = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            method = row["method"]
            if ds in DATASETS_ORDER and method in METHODS_ORDER:
                train_mean = row.get("total_train_time_s_mean", "")
                sample_mean = row.get("sample_time_s_mean", "")
                if train_mean:
                    data[ds][method] = {
                        "train_mean": float(train_mean),
                        "train_std": float(row.get("total_train_time_s_std", "0")),
                        "sample_mean": float(sample_mean) if sample_mean else 0,
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
    datasets = [ds for ds in DATASETS_ORDER if ds in data]
    methods = [m for m in METHODS_ORDER if any(m in data[ds] for ds in datasets)]

    n_ds = len(datasets)
    n_m = len(methods)
    bar_width = 0.8 / n_m
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        means = []
        stds = []
        for ds in datasets:
            if method in data[ds]:
                means.append(data[ds][method]["train_mean"])
                stds.append(data[ds][method]["train_std"])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_m / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
               label=METHOD_LABELS[method],
               color=METHOD_COLORS[method],
               capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets], fontsize=12)
    ax.set_ylabel("Training time (s)", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11, framealpha=0.8)
    ax.grid(True, axis='y', alpha=0.2)
    ax.set_title("Training time comparison",
                 fontsize=15, fontweight='bold')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_runtime.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"Runtime figure saved to {out_dir}/figure_runtime.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
