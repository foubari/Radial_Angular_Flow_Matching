"""Figure 1 (v3 — heavy-tailed): same banana/crescent target as v2 but with
a heavier radial tail, to make the radial mismatch of an isotropic Gaussian
source even more visually obvious.

Differences vs v2:
  - Student-t df: 5  ->  3   (heavier tails, more outliers).
  - Radial perturbation amplitude: 0.22  ->  0.38.
  - Outward skew bias: 0.45 * |u|  ->  0.65 * |u|.
  - Floor on r removed (not needed once skew is strong enough).

Output filenames are suffixed with _v3 / _heavytail so v2 outputs are kept.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse all helpers from v2; we only override the target sampler and output names.
from scripts.figure1_method_v2 import (
    fm_interpolate, rafm_interpolate,
    KDE_CMAP, KDE_BG, compute_kde,
)


# ─── Heavier-tailed banana target ─────────────────────────────────────────

def sample_target(n: int, seed: int = 42) -> np.ndarray:
    """Banana/crescent target with a markedly heavier radial tail than v2."""
    rng = np.random.default_rng(seed)

    s = rng.beta(2.0, 5.0, size=n)

    theta_min, theta_max = np.deg2rad(-45.0), np.deg2rad(195.0)
    theta = theta_min + s * (theta_max - theta_min)

    r0, dr = 2.0, 1.05
    r_mean = r0 + dr * np.power(s, 0.6)

    # Heavier-tailed perturbation.
    df = 3.0                         # was 5.0
    u = rng.standard_t(df, size=n)
    skew_noise = 0.38 * (u + 0.65 * np.abs(u))   # was 0.22 * (u + 0.45 * |u|)
    r = r_mean + skew_noise

    # Light floor only to avoid pathological negatives from the t tail.
    r = np.maximum(r, 0.6)

    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


# ─── Local copies of save / composite functions, with v3 output names ────

KDE_CACHE_DIR = Path("figures/paper_figures/.kde_cache_v3")


def _compute_kde(data, xlim, ylim, grid_n=200, bw_method=0.15, cache_key=None):
    if cache_key is not None:
        KDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = KDE_CACHE_DIR / f"{cache_key}.npz"
        if p.exists():
            d = np.load(p)
            return d["Xg"], d["Yg"], d["Z"]
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data.T, bw_method=bw_method)
    xg = np.linspace(xlim[0], xlim[1], grid_n)
    yg = np.linspace(ylim[0], ylim[1], grid_n)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(grid_n, grid_n)
    if cache_key is not None:
        np.savez_compressed(p, Xg=Xg, Yg=Yg, Z=Z)
    return Xg, Yg, Z


def save_panel(data, R_target, norm_color, cmap, R_median, out_path, xlim, ylim, title=""):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(data[:, 0], data[:, 1], c=R_target, cmap=cmap,
               norm=norm_color, s=0.8, alpha=0.6, rasterized=True, edgecolors='none')
    ax.add_patch(plt.Circle((0, 0), R_median, fill=False, color='grey',
                            linestyle='--', linewidth=0.5, alpha=0.3))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=13)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def save_kde_panel(data, R_median, out_path, xlim, ylim, title="", cache_key=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor(KDE_BG)
    Xg, Yg, Z = _compute_kde(data, xlim, ylim, cache_key=cache_key)
    zmax = float(Z.max())
    fill_levels = np.linspace(zmax * 0.04, zmax, 12)
    line_levels = np.linspace(zmax * 0.08, zmax, 8)
    ax.contourf(Xg, Yg, Z, levels=fill_levels, cmap=KDE_CMAP, extend='max')
    ax.contour(Xg, Yg, Z, levels=line_levels, colors='k', linewidths=0.3, alpha=0.4)
    ax.add_patch(plt.Circle((0, 0), R_median, fill=False, color='white',
                            linestyle='--', linewidth=0.7, alpha=0.5))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=13)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def save_histogram(R_fm, R_rafm, R_target, R_median, r_max, out_path, show_legend=False):
    fig, ax = plt.subplots(figsize=(4, 2))
    bins = np.linspace(0, r_max, 50)
    ax.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
            label='target' if show_legend else None)
    ax.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
            histtype='step', linewidth=1.8, label='FM' if show_legend else None)
    ax.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
            histtype='step', linewidth=1.8, label='RAFM' if show_legend else None)
    ax.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, r_max); ax.set_yticks([])
    ax.set_xlabel(r"$\|x_t\|$", fontsize=11)
    ax.set_ylabel(r"$p(\|x_t\|)$", fontsize=11)
    if show_legend:
        ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def make_figure(x1, times, out_dir, xlim, ylim, r_max):
    n_t = len(times)
    R_target = np.linalg.norm(x1, axis=1)
    norm_color = mcolors.Normalize(vmin=R_target.min() - 0.2, vmax=R_target.max() + 0.2)
    cmap = plt.cm.plasma
    R_median = np.median(R_target)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm'})

    panel_dir = Path(out_dir); panel_dir.mkdir(parents=True, exist_ok=True)
    seed_vis = 123
    s, alpha = 0.6, 0.6
    row_labels = ["Gaussian FM", "RAFM (ours)"]

    fig = plt.figure(figsize=(3.0 * n_t, 8.8))
    gs = GridSpec(3, n_t, figure=fig, hspace=0.06, wspace=0.10,
                  height_ratios=[1, 1, 0.45])
    first_hist_ax = None

    for col, t in enumerate(times):
        t_label = f"{t:.2f}"
        title = (r"$t=0$ (source)" if t < 1e-8
                 else r"$t=1$ (target)" if t > 1 - 1e-8
                 else f"$t={t_label}$")

        x_fm = fm_interpolate(x1, t, seed=seed_vis)
        x_rafm = rafm_interpolate(x1, t, seed=seed_vis)
        R_fm = np.linalg.norm(x_fm, axis=1)
        R_rafm = np.linalg.norm(x_rafm, axis=1)

        ax_fm = fig.add_subplot(gs[0, col])
        ax_fm.scatter(x_fm[:, 0], x_fm[:, 1], c=R_target, cmap=cmap,
                      norm=norm_color, s=s, alpha=alpha, rasterized=True, edgecolors='none')
        ax_fm.add_patch(plt.Circle((0, 0), R_median, fill=False, color='grey',
                                   linestyle='--', linewidth=0.5, alpha=0.3))
        ax_fm.set_xlim(xlim); ax_fm.set_ylim(ylim)
        ax_fm.set_aspect('equal'); ax_fm.set_xticks([]); ax_fm.set_yticks([])
        ax_fm.set_title(title, fontsize=20, fontweight='bold', pad=4)
        if col == 0:
            ax_fm.set_ylabel(row_labels[0], fontsize=20, fontweight='bold')
        save_panel(x_fm, R_target, norm_color, cmap, R_median,
                   str(panel_dir / f"v3_fm_t{t_label}.png"), xlim, ylim, title)

        ax_rafm = fig.add_subplot(gs[1, col])
        ax_rafm.scatter(x_rafm[:, 0], x_rafm[:, 1], c=R_target, cmap=cmap,
                        norm=norm_color, s=s, alpha=alpha, rasterized=True, edgecolors='none')
        ax_rafm.add_patch(plt.Circle((0, 0), R_median, fill=False, color='grey',
                                     linestyle='--', linewidth=0.5, alpha=0.3))
        ax_rafm.set_xlim(xlim); ax_rafm.set_ylim(ylim)
        ax_rafm.set_aspect('equal'); ax_rafm.set_xticks([]); ax_rafm.set_yticks([])
        if col == 0:
            ax_rafm.set_ylabel(row_labels[1], fontsize=20, fontweight='bold')
        save_panel(x_rafm, R_target, norm_color, cmap, R_median,
                   str(panel_dir / f"v3_rafm_t{t_label}.png"), xlim, ylim, title)

        ax_hist = fig.add_subplot(gs[2, col])
        bins = np.linspace(0, r_max, 50)
        ax_hist.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
                     label='target' if col == 0 else None)
        ax_hist.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
                     histtype='step', linewidth=1.8, label='FM' if col == 0 else None)
        ax_hist.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
                     histtype='step', linewidth=1.8, label='RAFM' if col == 0 else None)
        ax_hist.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_hist.set_xlim(0, r_max); ax_hist.set_yticks([])
        ax_hist.set_xlabel(r"$\|x_t\|$", fontsize=26)
        if col == 0:
            ax_hist.set_ylabel(r"$p(\|x_t\|)$", fontsize=26)
            first_hist_ax = ax_hist
        save_histogram(R_fm, R_rafm, R_target, R_median, r_max,
                       str(panel_dir / f"v3_hist_t{t_label}.png"),
                       show_legend=(col == 0))

    handles, labels = first_hist_ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=18, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.06))

    fig.savefig(str(panel_dir / "figure1_v3_composite.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(str(panel_dir / "figure1_v3_composite.png"), bbox_inches='tight', dpi=300)
    print(f"Composite (scatter): {panel_dir / 'figure1_v3_composite.pdf'}")
    plt.close(fig)


def make_kde_figure(x1, times, out_dir, xlim, ylim, r_max):
    n_t = len(times)
    R_target = np.linalg.norm(x1, axis=1)
    R_median = np.median(R_target)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm'})

    panel_dir = Path(out_dir)
    seed_vis = 123
    row_labels = ["Gaussian FM", "RAFM (ours)"]

    fig = plt.figure(figsize=(3.0 * n_t, 7.5))
    gs = GridSpec(3, n_t, figure=fig, hspace=0.06, wspace=0.10,
                  height_ratios=[1, 1, 0.45])
    first_hist_ax = None

    for col, t in enumerate(times):
        t_label = f"{t:.2f}"
        title = (r"$t=0$ (source)" if t < 1e-8
                 else r"$t=1$ (target)" if t > 1 - 1e-8
                 else f"$t={t_label}$")

        x_fm = fm_interpolate(x1, t, seed=seed_vis)
        x_rafm = rafm_interpolate(x1, t, seed=seed_vis)
        R_fm = np.linalg.norm(x_fm, axis=1)
        R_rafm = np.linalg.norm(x_rafm, axis=1)

        Xg, Yg, Z_fm = _compute_kde(x_fm, xlim, ylim, cache_key=f"v3_fm_t{t_label}")
        zmax_fm = float(Z_fm.max())
        fl_fm = np.linspace(zmax_fm * 0.04, zmax_fm, 12)
        ll_fm = np.linspace(zmax_fm * 0.08, zmax_fm, 8)
        ax_fm = fig.add_subplot(gs[0, col])
        ax_fm.set_facecolor(KDE_BG)
        ax_fm.contourf(Xg, Yg, Z_fm, levels=fl_fm, cmap=KDE_CMAP, extend='max')
        ax_fm.contour(Xg, Yg, Z_fm, levels=ll_fm, colors='k', linewidths=0.5, alpha=0.6)
        ax_fm.add_patch(plt.Circle((0, 0), R_median, fill=False, color='white',
                                   linestyle='--', linewidth=0.7, alpha=0.5))
        ax_fm.set_xlim(xlim); ax_fm.set_ylim(ylim)
        ax_fm.set_aspect('equal'); ax_fm.set_xticks([]); ax_fm.set_yticks([])
        ax_fm.set_title(title, fontsize=20, fontweight='bold', pad=4)
        if col == 0:
            ax_fm.set_ylabel(row_labels[0], fontsize=20, fontweight='bold')
        save_kde_panel(x_fm, R_median, str(panel_dir / f"v3_kde_fm_t{t_label}.png"),
                       xlim, ylim, title, cache_key=f"v3_fm_t{t_label}")

        _, _, Z_rafm = _compute_kde(x_rafm, xlim, ylim, cache_key=f"v3_rafm_t{t_label}")
        zmax_rafm = float(Z_rafm.max())
        fl_rafm = np.linspace(zmax_rafm * 0.04, zmax_rafm, 12)
        ll_rafm = np.linspace(zmax_rafm * 0.08, zmax_rafm, 8)
        ax_rafm = fig.add_subplot(gs[1, col])
        ax_rafm.set_facecolor(KDE_BG)
        ax_rafm.contourf(Xg, Yg, Z_rafm, levels=fl_rafm, cmap=KDE_CMAP, extend='max')
        ax_rafm.contour(Xg, Yg, Z_rafm, levels=ll_rafm, colors='k', linewidths=0.5, alpha=0.6)
        ax_rafm.add_patch(plt.Circle((0, 0), R_median, fill=False, color='white',
                                     linestyle='--', linewidth=0.7, alpha=0.5))
        ax_rafm.set_xlim(xlim); ax_rafm.set_ylim(ylim)
        ax_rafm.set_aspect('equal'); ax_rafm.set_xticks([]); ax_rafm.set_yticks([])
        if col == 0:
            ax_rafm.set_ylabel(row_labels[1], fontsize=20, fontweight='bold')
        save_kde_panel(x_rafm, R_median, str(panel_dir / f"v3_kde_rafm_t{t_label}.png"),
                       xlim, ylim, title, cache_key=f"v3_rafm_t{t_label}")

        ax_hist = fig.add_subplot(gs[2, col])
        bins = np.linspace(0, r_max, 50)
        ax_hist.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
                     label='target' if col == 0 else None)
        ax_hist.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
                     histtype='step', linewidth=1.8, label='FM' if col == 0 else None)
        ax_hist.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
                     histtype='step', linewidth=1.8, label='RAFM' if col == 0 else None)
        ax_hist.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_hist.set_xlim(0, r_max); ax_hist.set_yticks([])
        ax_hist.set_xlabel(r"$\|x_t\|$", fontsize=26)
        if col == 0:
            ax_hist.set_ylabel(r"$p(\|x_t\|)$", fontsize=26)
            first_hist_ax = ax_hist

    handles, labels = first_hist_ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=18, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.12))

    fig.savefig(str(panel_dir / "figure1_v3_kde_composite.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(str(panel_dir / "figure1_v3_kde_composite.png"), bbox_inches='tight', dpi=300)
    print(f"Composite (KDE): {panel_dir / 'figure1_v3_kde_composite.pdf'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--out_dir", type=str, default="figures/paper_figures")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache and KDE_CACHE_DIR.exists():
        import shutil
        shutil.rmtree(KDE_CACHE_DIR)
        print("KDE cache (v3) cleared.")

    x1 = sample_target(args.n)
    R = np.linalg.norm(x1, axis=1)
    span = float(np.quantile(np.abs(x1), 0.995)) + 0.6
    xlim = (-span, span); ylim = (-span, span)
    r_max = float(np.quantile(R, 0.995) * 1.05)

    times = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]
    kde_times = [0.0, 0.50, 0.75, 0.90, 1.0]
    make_figure(x1, times, args.out_dir, xlim, ylim, r_max)
    make_kde_figure(x1, kde_times, args.out_dir, xlim, ylim, r_max)


if __name__ == "__main__":
    main()
