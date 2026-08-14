"""Figure: Sample efficiency (Exp 2).

Plots radial_w1 and sliced_w1 vs n_train for different methods.
Shows empirical RAFM converging toward oracle as n increases.

Usage:
    python scripts/figure_sample_efficiency.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_exp2_results(path: str = "outputs/exp2_sample_efficiency/student_t_d16_df3.0_cor/all_results.json"):
    with open(path) as f:
        data = json.load(f)

    # Parse keys like "rafm_empirical_ecdf_n500_seed8925"
    records = defaultdict(lambda: defaultdict(list))
    for key, metrics in data.items():
        m = re.match(r"(.+)_n(\d+)_seed\d+", key)
        if not m:
            continue
        method = m.group(1)
        n = int(m.group(2))
        records[method][n].append(metrics)
    return records


METHOD_DISPLAY = {
    "gaussian_fm": ("Gaussian FM", "#228B22", "-", "s"),
    "rafm_oracle": ("RAFM (oracle)", "#1f5faa", "--", "^"),
    "rafm_empirical_ecdf": ("RAFM (empirical)", "#1f5faa", "-", "o"),
    "source_only_oracle": ("Source-only (oracle)", "#999999", "--", "v"),
    "source_only_empirical_ecdf": ("Source-only (empirical)", "#999999", "-", "D"),
}


def make_figure(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
    })

    records = load_exp2_results()

    metrics_to_plot = [
        ("radial_w1", r"Radial $W_1$"),
        ("sliced_w1", r"Sliced $W_1$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        for method, (label, color, ls, marker) in METHOD_DISPLAY.items():
            if method not in records:
                continue
            ns = sorted(records[method].keys())
            means = []
            stds = []
            for n in ns:
                vals = [r[metric_key] for r in records[method][n]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means = np.array(means)
            stds = np.array(stds)

            ax.errorbar(ns, means, yerr=stds, label=label, color=color,
                        linestyle=ls, marker=marker, markersize=5,
                        linewidth=1.5, capsize=3)

        ax.set_xscale('log')
        ax.set_xlabel(r"$n_{\mathrm{train}}$", fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Sample efficiency — Student-$t$, $d=16$",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_sample_efficiency.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"Sample efficiency figure saved to {out_dir}/figure_sample_efficiency.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
