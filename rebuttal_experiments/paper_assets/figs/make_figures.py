#!/usr/bin/env python3
"""Paper-ready figures for the Angular RAFM paper (Workflow-2, data-driven, matplotlib).

Reads ONLY the aggregated result files (no retraining):
  rebuttal_experiments/master_suite.json     — dim/df/aniso sweeps + singletons (tabular)
  rebuttal_experiments/master_results.json    — cross-domain (dc-ae, audio) 3-seed aggregates
  rebuttal_experiments/raw_results/E4_nfe_solver/*.csv  — NFE/solver drift sweep

Outputs (PDF vector + PNG 300dpi) into this directory:
  fig_synthetic_scaling.{pdf,png}   (A)
  fig_dcae_tradeoff.{pdf,png}       (C)
  fig_nfe_drift.{pdf,png}           (D)
  fig_realdata.{pdf,png}            (E)
Figure B (AudioMNIST mechanism) is built by make_audio_fig.py (needs the theory panel).
"""
import json, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SUITE = json.load(open(REPO / "rebuttal_experiments/master_suite.json"))
MASTER = json.load(open(REPO / "rebuttal_experiments/master_results.json"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "-",
    "lines.linewidth": 1.7, "lines.markersize": 4.5,
})
# Okabe-Ito colorblind-safe; Angular = coral (our method), std-RAFM = blue
C_ANG, C_RAFM, C_GAUSS, C_SRC, C_ORACLE = "#E76F51", "#0072B2", "#8C8C8C", "#56B4E9", "#009E73"
METHOD_STYLE = {   # key -> (label, color, marker, lw, z)
    "angular_rafm": ("Angular RAFM (ours)", C_ANG, "o", 2.2, 5),
    "rafm_empirical": ("RAFM (std)", C_RAFM, "s", 1.7, 4),
    "gaussian_fm": ("Gaussian FM", C_GAUSS, "^", 1.3, 2),
    "source_only_empirical": ("Source-only", C_SRC, "v", 1.3, 2),
    "rafm_oracle": ("RAFM (oracle)", C_ORACLE, "D", 1.2, 3),
}


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf + .png")


def series(sweep, method, metric):
    """Return (xs, means, stds) over a sweep dict {xval: {method:{metric:{mean,std}}}}, sorted by x."""
    xs, ms, ss = [], [], []
    for xk in sorted(sweep, key=lambda z: float(z)):
        cell = sweep[xk].get(method, {})
        a = cell.get(metric) if cell else None
        if a and a.get("mean") is not None and not math.isnan(a["mean"]):
            xs.append(float(xk)); ms.append(a["mean"]); ss.append(a.get("std", 0.0))
    return np.array(xs), np.array(ms), np.array(ss)


# ---------------------------------------------------------------- Figure A
def fig_synthetic_scaling():
    sweeps = [("dim", SUITE["dim"], "Dimension $d$", True),
              ("df", SUITE["df"], r"Student-$t$ dof (tail: low=heavy)", True),
              ("aniso", SUITE["aniso"], r"Anisotropy $\kappa$", True)]
    metrics = [("radial_w1", "Radial $W_1$ $\\downarrow$"),
               ("sliced_w1", "Sliced $W_1$ $\\downarrow$"),
               ("angular_sw_mean", "Angular SW $\\downarrow$")]
    methods = ["gaussian_fm", "source_only_empirical", "rafm_empirical", "rafm_oracle", "angular_rafm"]
    fig, axes = plt.subplots(3, 3, figsize=(6.75, 6.2))
    for r, (sname, sweep, xlab, logx) in enumerate(sweeps):
        for c, (metric, ylab) in enumerate(metrics):
            ax = axes[r][c]
            for m in methods:
                xs, ms, ss = series(sweep, m, metric)
                if len(xs) == 0:
                    continue
                lab, col, mk, lw, z = METHOD_STYLE[m]
                ax.plot(xs, ms, color=col, marker=mk, lw=lw, zorder=z,
                        label=lab if (r == 0 and c == 0) else None)
                ax.fill_between(xs, ms - ss, ms + ss, color=col, alpha=0.10, zorder=z - 1)
            if logx:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlab)          # each row is a different sweep axis -> label all rows
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(["Radial", "Sliced (overall)", "Directional"][c])
    # row labels on the right
    for r, name in enumerate(["Dimension", "Tail (dof)", "Anisotropy"]):
        axes[r][2].annotate(name, xy=(1.02, 0.5), xycoords="axes fraction",
                            rotation=270, va="center", ha="left", fontsize=8.5, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03),
               columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_synthetic_scaling")


# ---------------------------------------------------------------- Figure C
def fig_dcae_tradeoff():
    dc = MASTER["dcae"]
    order = [("gaussian_euclidean", "Gaussian FM", C_GAUSS, "^"),
             ("matched_euclidean", "Matched-Eucl.", "#B07AA1", "P"),
             ("fixed_spherical", "Fixed-spherical", "#E9C46A", "X"),
             ("rafm", "RAFM (std)", C_RAFM, "s"),
             ("angular_rafm", "Angular RAFM (ours)", C_ANG, "o")]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for key, lab, col, mk in order:
        row = dc.get(key)
        if not row or not row.get("fid"):
            continue
        x, xe = row["radial_w1"]["mean"], row["radial_w1"].get("std", 0)
        y, ye = row["fid"]["mean"], row["fid"].get("std", 0)
        big = 90 if key == "angular_rafm" else 45
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, ms=math.sqrt(big), color=col,
                    ecolor=col, elinewidth=1, capsize=2, zorder=(5 if key == "angular_rafm" else 3),
                    mec="black" if key == "angular_rafm" else "none", mew=0.7, label=lab)
    ax.set_xlabel("Radial $W_1$ (latent) $\\downarrow$")
    ax.set_ylabel("FID $\\downarrow$")
    ax.set_title("DC-AE ImageNette: FID vs radial calibration")
    ax.legend(loc="upper center", fontsize=6.8, ncol=1, handletextpad=0.3)
    ax.annotate("best FID +\nbest radial", xy=(dc["angular_rafm"]["radial_w1"]["mean"],
                dc["angular_rafm"]["fid"]["mean"]), xytext=(1.1, 150),
                fontsize=6.5, color=C_ANG, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_ANG, lw=0.8))
    fig.tight_layout()
    save(fig, "fig_dcae_tradeoff")


# ---------------------------------------------------------------- Figure D
def fig_nfe_drift():
    E4 = REPO / "rebuttal_experiments/raw_results/E4_nfe_solver"
    pairs = [("Student-$t$ d16", "rafm_empirical_d16.csv", "angular_rafm_d16.csv"),
             ("PIV d64", "piv_rafm.csv", "piv_angular.csv"),
             ("Weather", "weather_rafm.csv", "weather_angular.csv"),
             ("Toy-2D (d=2)", "rafm_empirical_toy2d.csv", "angular_rafm_toy2d.csv")]

    def load(fn):
        p = E4 / fn
        if not p.exists():
            return None
        d = {}
        for row in csv.DictReader(open(p)):
            if row["solver"] == "rk4" and row["project_tangent"] == "True":
                try:
                    d[int(row["actual_nfe"])] = float(row["drift_final_mean"])
                except ValueError:
                    d[int(row["actual_nfe"])] = float("nan")
        return d
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.95), sharey=False)
    for ax, (ttl, sf, af) in zip(axes, pairs):
        for lab, fn, col, mk in [("RAFM (std)", sf, C_RAFM, "s"), ("Angular", af, C_ANG, "o")]:
            d = load(fn)
            if not d:
                continue
            xs = sorted(k for k in d if d[k] == d[k])  # drop nan
            ys = [d[k] for k in xs]
            ax.plot(xs, ys, color=col, marker=mk, label=lab)
            nanx = sorted(k for k in d if d[k] != d[k])
            if nanx:  # mark divergence
                ax.axvspan(min(nanx) - 0.5, max(d) + 5, color=C_ANG, alpha=0.06)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("NFE (RK4)")
    axes[0].set_ylabel("Radius drift $\\downarrow$")
    axes[3].text(0.5, 0.5, "Angular diverges\n(NaN) at d=2", transform=axes[3].transAxes,
                 fontsize=6.5, color=C_ANG, ha="center", va="center")
    axes[0].legend(loc="upper right", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_nfe_drift")


# ---------------------------------------------------------------- Figure E
def fig_realdata():
    single = SUITE["singletons"]
    dsets = [("piv_d64", "PIV"), ("weather_au_wind", "Weather"), ("finance_ff49", "Finance")]
    metrics = [("radial_w1", "Radial $W_1$"), ("sliced_w1", "Sliced $W_1$"), ("angular_sw_mean", "Angular SW")]
    methods = ["gaussian_fm", "source_only_empirical", "rafm_empirical", "angular_rafm"]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2))
    for ax, (metric, mlab) in zip(axes, metrics):
        x = np.arange(len(dsets)); n = len(methods); w = 0.8 / n
        for i, m in enumerate(methods):
            lab, col, mk, lw, z = METHOD_STYLE[m]
            vals = [single[d].get(m, {}).get(metric, {}) for d, _ in dsets]
            ys = [(v["mean"] if v and v.get("mean") is not None else np.nan) for v in vals]
            es = [(v.get("std", 0) if v else 0) for v in vals]
            ax.bar(x + (i - n / 2 + 0.5) * w, ys, w * 0.9, yerr=es, color=col,
                   ecolor="#444", capsize=1.5, label=lab, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels([d[1] for d in dsets])
        ax.set_title(mlab); ax.set_yscale("log")
    axes[0].set_ylabel("value $\\downarrow$ (log)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), columnspacing=1.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_realdata")


if __name__ == "__main__":
    print("Building paper figures...")
    fig_synthetic_scaling()
    fig_dcae_tradeoff()
    fig_nfe_drift()
    fig_realdata()
    print("done.")
