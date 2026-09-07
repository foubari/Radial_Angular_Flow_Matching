"""Figure 1 (v2): FM vs RAFM on a thick asymmetric banana/crescent target.

Target = one curved banana/kidney-shaped cloud with broad angular support, a
non-Gaussian radial law, clear anisotropy and visible asymmetry (one end
denser than the other, radius grows along the arc). Little mass near the
origin so that the radial mismatch of an isotropic Gaussian source is
visually obvious, while leaving enough angular structure for RAFM to
re-organize smoothly.

Layout:
  Row 1: Gaussian FM   — source = N(0, I)
  Row 2: RAFM (ours)   — source = same radial law as target, uniform in angle
  Row 3: ||x_t|| histograms (target grey, FM green, RAFM blue)

Usage:
    python scripts/figure1_method_v2.py
"""
import argparse
import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rafm.utils.sphere import slerp, uniform_on_sphere


# ─── Target: anisotropic, asymmetric, slightly skewed mixture ─────────────

def sample_target(n: int, seed: int = 42) -> np.ndarray:
    """Thick asymmetric banana / crescent target on a broad angular arc.

    - s ~ Beta(2, 5)  →  one end of the arc denser than the other.
    - theta sweeps an arc of ~230° around the origin (broad angular support).
    - Mean radius grows along the arc: r_mean(s) = r0 + dr * s^0.6,
      so the dense end sits closer in and the sparse end further out.
    - Radial thickness is a skewed perturbation (Student-t, slightly biased
      outward) → non-Gaussian radial law with a mild outward tail.
    """
    rng = np.random.default_rng(seed)

    s = rng.beta(2.0, 5.0, size=n)

    theta_min, theta_max = np.deg2rad(-45.0), np.deg2rad(195.0)
    theta = theta_min + s * (theta_max - theta_min)

    r0, dr = 2.0, 1.05
    r_mean = r0 + dr * np.power(s, 0.6)

    # Skewed radial perturbation: heavy-tailed Student-t, biased outward.
    df = 5.0
    u = rng.standard_t(df, size=n)
    skew_noise = 0.22 * (u + 0.45 * np.abs(u))
    r = r_mean + skew_noise

    # Floor radius to keep mass away from the origin.
    r = np.maximum(r, 1.25)

    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


# ─── Path samplers ─────────────────────────────────────────────────────────

def fm_interpolate(x1: np.ndarray, t: float, seed: int = 0) -> np.ndarray:
    """Standard FM: x_t = (1-t)*x0 + t*x1, x0 ~ N(0,I)."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(x1.shape)
    return (1 - t) * x0 + t * x1


def rafm_interpolate(x1: np.ndarray, t: float, seed: int = 0) -> np.ndarray:
    """RAFM: x0 = R*u0 (R from target radial law, u0 uniform), x_t = slerp(x0, x1, t)."""
    n = x1.shape[0]
    x1_t = torch.from_numpy(x1).float()
    R = torch.norm(x1_t, dim=-1, keepdim=True)

    torch.manual_seed(seed)
    u0 = uniform_on_sphere(n, 2)
    x0_t = R * u0

    if t < 1e-8:
        return x0_t.numpy()
    if t > 1 - 1e-8:
        return x1
    x_t = slerp(x0_t, x1_t, t)
    return x_t.numpy()


# ─── Desaturated colormap for KDE ─────────────────────────────────────────

def desaturate_cmap(cmap_name: str, factor: float = 0.7, n: int = 256):
    base = plt.cm.get_cmap(cmap_name, n)
    colors = base(np.linspace(0, 1, n))
    for i, (r, g, b, a) in enumerate(colors):
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s * factor)
        colors[i] = (r2, g2, b2, a)
    return ListedColormap(colors)

KDE_CMAP = desaturate_cmap('magma_r', factor=0.7)
KDE_BG = KDE_CMAP(0)  # lightest (cream) color, used as panel background


# ─── KDE helpers ──────────────────────────────────────────────────────────

KDE_CACHE_DIR = Path("figures/paper_figures/.kde_cache_v2")


def compute_kde(data: np.ndarray, xlim, ylim, grid_n=200, bw_method=0.15,
                cache_key: str | None = None):
    if cache_key is not None:
        KDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = KDE_CACHE_DIR / f"{cache_key}.npz"
        if cache_path.exists():
            d = np.load(cache_path)
            return d["Xg"], d["Yg"], d["Z"]
    kde = gaussian_kde(data.T, bw_method=bw_method)
    xg = np.linspace(xlim[0], xlim[1], grid_n)
    yg = np.linspace(ylim[0], ylim[1], grid_n)
    Xg, Yg = np.meshgrid(xg, yg)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde(positions).reshape(grid_n, grid_n)
    if cache_key is not None:
        np.savez_compressed(cache_path, Xg=Xg, Yg=Yg, Z=Z)
    return Xg, Yg, Z


def save_panel(data, R_target, norm_color, cmap, R_median, out_path,
               xlim, ylim, title=""):
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


def save_kde_panel(data, R_median, out_path, xlim, ylim, title="",
                   cache_key: str | None = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor(KDE_BG)
    Xg, Yg, Z = compute_kde(data, xlim, ylim, cache_key=cache_key)
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


def save_histogram(R_fm, R_rafm, R_target, R_median, r_max, out_path,
                   show_legend=False):
    fig, ax = plt.subplots(figsize=(4, 2))
    bins = np.linspace(0, r_max, 50)
    ax.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
            label='target' if show_legend else None)
    ax.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
            histtype='step', linewidth=1.8,
            label='FM' if show_legend else None)
    ax.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
            histtype='step', linewidth=1.8,
            label='RAFM' if show_legend else None)
    ax.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, r_max); ax.set_yticks([])
    ax.set_xlabel(r"$\|x_t\|$", fontsize=11)
    ax.set_ylabel(r"$p(\|x_t\|)$", fontsize=11)
    if show_legend:
        ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


# ─── Composite figures ────────────────────────────────────────────────────

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
                   str(panel_dir / f"v2_fm_t{t_label}.png"), xlim, ylim, title)

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
                   str(panel_dir / f"v2_rafm_t{t_label}.png"), xlim, ylim, title)

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
                       str(panel_dir / f"v2_hist_t{t_label}.png"),
                       show_legend=(col == 0))

    # Shared legend below the histogram row, centered.
    handles, labels = first_hist_ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=18, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.06))

    composite_pdf = str(panel_dir / "figure1_v2_composite.pdf")
    composite_png = str(panel_dir / "figure1_v2_composite.png")
    fig.savefig(composite_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(composite_png, bbox_inches='tight', dpi=300)
    print(f"Composite (scatter): {composite_pdf}")
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

        Xg, Yg, Z_fm = compute_kde(x_fm, xlim, ylim, cache_key=f"v2_fm_t{t_label}")
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
        save_kde_panel(x_fm, R_median, str(panel_dir / f"v2_kde_fm_t{t_label}.png"),
                       xlim, ylim, title, cache_key=f"v2_fm_t{t_label}")

        _, _, Z_rafm = compute_kde(x_rafm, xlim, ylim, cache_key=f"v2_rafm_t{t_label}")
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
        save_kde_panel(x_rafm, R_median, str(panel_dir / f"v2_kde_rafm_t{t_label}.png"),
                       xlim, ylim, title, cache_key=f"v2_rafm_t{t_label}")

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

    composite_pdf = str(panel_dir / "figure1_v2_kde_composite.pdf")
    composite_png = str(panel_dir / "figure1_v2_kde_composite.png")
    fig.savefig(composite_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(composite_png, bbox_inches='tight', dpi=300)
    print(f"Composite (KDE): {composite_pdf}")
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
        print("KDE cache cleared.")

    x1 = sample_target(args.n)
    R = np.linalg.norm(x1, axis=1)
    # Use a robust quantile so the rare tail does not shrink the visible blob.
    span = float(np.quantile(np.abs(x1), 0.995)) + 0.6
    xlim = (-span, span)
    ylim = (-span, span)
    r_max = float(np.quantile(R, 0.995) * 1.05)

    times = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]
    kde_times = [0.0, 0.50, 0.75, 0.90, 1.0]
    make_figure(x1, times, args.out_dir, xlim, ylim, r_max)
    make_kde_figure(x1, kde_times, args.out_dir, xlim, ylim, r_max)


if __name__ == "__main__":
    main()
