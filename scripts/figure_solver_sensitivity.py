"""Figure: Solver sensitivity (Exp 4).

Plots sliced_w1 vs NFE for Euler, Heun, RK4 solvers.
Shows RAFM is robust to solver choice.

Usage:
    python scripts/figure_solver_sensitivity.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_exp4_results(path: str = "outputs/exp4_solver_sensitivity/solver_sensitivity_results.json"):
    with open(path) as f:
        data = json.load(f)

    # Parse keys like "rafm_empirical_euler_nfe4"
    records = defaultdict(dict)
    for key, metrics in data.items():
        m = re.match(r"rafm_empirical_(\w+)_nfe(\d+)", key)
        if not m:
            continue
        solver = m.group(1)
        nfe = int(m.group(2))
        records[solver][nfe] = metrics
    return records


SOLVER_STYLE = {
    "euler": ("Euler", "#228B22", "o"),
    "heun": ("Heun", "#e67e22", "s"),
    "rk4": ("RK4", "#1f5faa", "^"),
}


def make_figure(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
    })

    records = load_exp4_results()

    metrics_to_plot = [
        ("sliced_w1", r"Sliced $W_1$"),
        ("radial_w1", r"Radial $W_1$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        for solver, (label, color, marker) in SOLVER_STYLE.items():
            if solver not in records:
                continue
            nfes = sorted(records[solver].keys())
            vals = [records[solver][n][metric_key] for n in nfes]

            ax.plot(nfes, vals, label=label, color=color, marker=marker,
                    markersize=6, linewidth=1.5)

        ax.set_xscale('log', base=2)
        ax.set_xlabel("NFE", fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=11, framealpha=0.8)

    fig.suptitle("Solver sensitivity — RAFM on Student-$t$, $d=16$",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_solver_sensitivity.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"Solver sensitivity figure saved to {out_dir}/figure_solver_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
