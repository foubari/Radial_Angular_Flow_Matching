"""Figure 1: FM vs RAFM — analytical path visualization.

Shows intermediate distributions p_t for both methods on a 2D toy target
(3 elliptical Gaussian blobs on a ring). No model training — pure path sampling.

Layout:
  Row 1: FM  — t=0, 0.25, 0.5, 0.75, 0.9, 1.0
  Row 2: RAFM — same time steps
  Row 3: ||x_t|| histograms at each t (FM blue, RAFM red, target grey)

Also saves individual panels to figures/paper_figures/ for flexible LaTeX use.

Usage:
    python scripts/figure1_method.py
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
import colorsys
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rafm.utils.sphere import slerp, uniform_on_sphere


# ─── Target distribution: 3 elliptical Gaussian blobs on a ring ───────────

def sample_target(n: int, seed: int = 42) -> np.ndarray:
    """3 elliptical blobs at radius ~2.5, evenly spaced angularly."""
    rng = np.random.default_rng(seed)
    n_per = n // 3
    centers = []
    for k in range(3):
        angle = 2 * np.pi * k / 3 + np.pi / 6
        centers.append(2.5 * np.array([np.cos(angle), np.sin(angle)]))

    samples = []
    for k, c in enumerate(centers):
        theta_rot = 2 * np.pi * k / 3
        R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
                       [np.sin(theta_rot),  np.cos(theta_rot)]])
        cov_local = np.array([[0.12, 0.0], [0.0, 0.03]])
        cov = R @ cov_local @ R.T
        pts = rng.multivariate_normal(c, cov, size=n_per)
        samples.append(pts)

    return np.concatenate(samples, axis=0)


# ─── Path samplers ─────────────────────────────────────────────────────────

def fm_interpolate(x1: np.ndarray, t: float, seed: int = 0) -> np.ndarray:
    """Standard FM: x_t = (1-t)*x0 + t*x1, x0 ~ N(0,I)."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(x1.shape)
    return (1 - t) * x0 + t * x1


def rafm_interpolate(x1: np.ndarray, t: float, seed: int = 0) -> np.ndarray:
    """RAFM: x0 = R*u0, x_t = slerp(x0, x1, t), preserving ||x_t|| = R."""
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


# ─── Individual panel saver ────────────────────────────────────────────────

def save_panel(data: np.ndarray, R_target: np.ndarray, norm_color, cmap,
               R_median: float, out_path: str, title: str = ""):
    """Save a single scatter panel as an individual figure."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(data[:, 0], data[:, 1], c=R_target, cmap=cmap,
               norm=norm_color, s=0.8, alpha=0.6, rasterized=True, edgecolors='none')
    circle = plt.Circle((0, 0), R_median, fill=False, color='grey',
                         linestyle='--', linewidth=0.5, alpha=0.3)
    ax.add_patch(circle)
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=13)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def save_histogram(R_fm, R_rafm, R_target, R_median, out_path, show_legend=False):
    """Save a single histogram panel."""
    fig, ax = plt.subplots(figsize=(4, 2))
    bins = np.linspace(0, 5, 40)
    ax.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
            label='target' if show_legend else None)
    ax.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
            histtype='step', linewidth=1.8,
            label='FM' if show_legend else None)
    ax.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
            histtype='step', linewidth=1.8,
            label='RAFM' if show_legend else None)
    ax.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_yticks([])
    ax.set_xlabel(r"$\|x_t\|$", fontsize=11)
    ax.set_ylabel(r"$p(\|x_t\|)$", fontsize=11)
    if show_legend:
        ax.legend(fontsize=9, loc='upper left', framealpha=0.7)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


# ─── Desaturated colormap ─────────────────────────────────────────────────

def desaturate_cmap(cmap_name: str, factor: float = 0.7, n: int = 256):
    """Return a copy of *cmap_name* with saturation scaled by *factor*."""
    base = plt.cm.get_cmap(cmap_name, n)
    colors = base(np.linspace(0, 1, n))
    for i, (r, g, b, a) in enumerate(colors):
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s * factor)
        colors[i] = (r2, g2, b2, a)
    return ListedColormap(colors)

KDE_CMAP = desaturate_cmap('magma_r', factor=0.7)


# ─── KDE panels ───────────────────────────────────────────────────────────

KDE_CACHE_DIR = Path("figures/paper_figures/.kde_cache")


def compute_kde(data: np.ndarray, xlim, ylim, grid_n=200, bw_method=0.15,
                cache_key: str | None = None):
    """Compute 2D KDE on a grid, with optional disk cache."""
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


def save_kde_panel(data: np.ndarray, R_median: float, out_path: str,
                   title: str = "", xlim=(-4.2, 4.2), ylim=(-4.2, 4.2),
                   cache_key: str | None = None):
    """Save a single KDE contour panel."""
    fig, ax = plt.subplots(figsize=(4, 4))
    Xg, Yg, Z = compute_kde(data, xlim, ylim, cache_key=cache_key)
    ax.contourf(Xg, Yg, Z, levels=12, cmap=KDE_CMAP)
    ax.contour(Xg, Yg, Z, levels=8, colors='k', linewidths=0.3, alpha=0.4)
    circle = plt.Circle((0, 0), R_median, fill=False, color='white',
                         linestyle='--', linewidth=0.7, alpha=0.5)
    ax.add_patch(circle)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=13)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def make_kde_figure(x1: np.ndarray, times: list[float], out_dir: str):
    """Create KDE composite figure + individual panels."""
    n_t = len(times)
    R_target = np.linalg.norm(x1, axis=1)
    R_median = np.median(R_target)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'mathtext.fontset': 'cm',
    })

    panel_dir = Path(out_dir)
    seed_vis = 123
    xlim = (-4.2, 4.2)
    ylim = (-4.2, 4.2)

    # ── Composite KDE figure ──
    fig = plt.figure(figsize=(3.0 * n_t, 7.5))
    # Use equal height ratios so hspace gives identical gaps between all rows
    gs = GridSpec(3, n_t, figure=fig, hspace=0.06, wspace=0.10,
                  height_ratios=[1, 1, 0.45])
    row_labels = ["Gaussian FM", "RAFM (ours)"]

    for col, t in enumerate(times):
        t_label = f"{t:.2f}"
        # Plain-text titles for readability at paper size
        if t < 1e-8:
            title_txt = "t = 0 (source)"
        elif t > 1 - 1e-8:
            title_txt = "t = 1 (target)"
        else:
            title_txt = f"t = {t_label}"

        x_fm = fm_interpolate(x1, t, seed=seed_vis)
        R_fm = np.linalg.norm(x_fm, axis=1)
        x_rafm = rafm_interpolate(x1, t, seed=seed_vis)
        R_rafm = np.linalg.norm(x_rafm, axis=1)

        # FM KDE
        Xg, Yg, Z_fm = compute_kde(x_fm, xlim, ylim, cache_key=f"fm_t{t_label}")
        ax_fm = fig.add_subplot(gs[0, col])
        ax_fm.contourf(Xg, Yg, Z_fm, levels=12, cmap=KDE_CMAP)
        ax_fm.contour(Xg, Yg, Z_fm, levels=8, colors='k', linewidths=0.5, alpha=0.6)
        circle = plt.Circle((0, 0), R_median, fill=False, color='white',
                             linestyle='--', linewidth=0.7, alpha=0.5)
        ax_fm.add_patch(circle)
        ax_fm.set_xlim(xlim); ax_fm.set_ylim(ylim)
        ax_fm.set_aspect('equal')
        ax_fm.set_xticks([]); ax_fm.set_yticks([])
        ax_fm.set_title(title_txt, fontsize=20, fontweight='bold', pad=4)
        if col == 0:
            ax_fm.set_ylabel(row_labels[0], fontsize=20, fontweight='bold')

        save_kde_panel(x_fm, R_median,
                       str(panel_dir / f"kde_fm_t{t_label}.png"), title_txt,
                       cache_key=f"fm_t{t_label}")

        # RAFM KDE
        _, _, Z_rafm = compute_kde(x_rafm, xlim, ylim, cache_key=f"rafm_t{t_label}")
        ax_rafm = fig.add_subplot(gs[1, col])
        ax_rafm.contourf(Xg, Yg, Z_rafm, levels=12, cmap=KDE_CMAP)
        ax_rafm.contour(Xg, Yg, Z_rafm, levels=8, colors='k', linewidths=0.5, alpha=0.6)
        circle2 = plt.Circle((0, 0), R_median, fill=False, color='white',
                              linestyle='--', linewidth=0.7, alpha=0.5)
        ax_rafm.add_patch(circle2)
        ax_rafm.set_xlim(xlim); ax_rafm.set_ylim(ylim)
        ax_rafm.set_aspect('equal')
        ax_rafm.set_xticks([]); ax_rafm.set_yticks([])
        if col == 0:
            ax_rafm.set_ylabel(row_labels[1], fontsize=20, fontweight='bold')

        save_kde_panel(x_rafm, R_median,
                       str(panel_dir / f"kde_rafm_t{t_label}.png"), title_txt,
                       cache_key=f"rafm_t{t_label}")

        # Histogram
        ax_hist = fig.add_subplot(gs[2, col])
        bins = np.linspace(0, 5, 40)
        ax_hist.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
                     label='target' if col == 0 else None)
        ax_hist.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
                     histtype='step', linewidth=1.8,
                     label='FM' if col == 0 else None)
        ax_hist.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
                     histtype='step', linewidth=1.8,
                     label='RAFM' if col == 0 else None)
        ax_hist.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_hist.set_xlim(0, 5)
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r"$\|x_t\|$", fontsize=26)
        if col == 0:
            ax_hist.set_ylabel(r"$p(\|x_t\|)$", fontsize=26)
            ax_hist.legend(fontsize=12, loc='upper left', framealpha=0.7)

    composite_pdf = str(panel_dir / "figure1_kde_composite.pdf")
    composite_png = str(panel_dir / "figure1_kde_composite.png")
    fig.savefig(composite_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(composite_png, bbox_inches='tight', dpi=300)
    print(f"KDE composite: {composite_pdf}")
    plt.close(fig)


# ─── Main composite figure ────────────────────────────────────────────────

def make_figure(x1: np.ndarray, times: list[float], out_dir: str):
    n_t = len(times)
    R_target = np.linalg.norm(x1, axis=1)

    r_min, r_max = R_target.min(), R_target.max()
    norm_color = mcolors.Normalize(vmin=r_min - 0.2, vmax=r_max + 0.2)
    cmap = plt.cm.plasma

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'mathtext.fontset': 'cm',
    })

    R_median = np.median(R_target)
    panel_dir = Path(out_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)

    seed_vis = 123

    # ── Composite figure ──
    fig = plt.figure(figsize=(3.0 * n_t, 8.8))
    gs = GridSpec(3, n_t, figure=fig, hspace=0.12, wspace=0.10,
                  height_ratios=[1, 1, 0.45])

    xlim = (-4.2, 4.2)
    ylim = (-4.2, 4.2)
    s = 0.6
    alpha = 0.6
    row_labels = ["Gaussian FM", "RAFM (ours)"]

    for col, t in enumerate(times):
        t_label = f"{t:.2f}"

        # ── FM ──
        x_fm = fm_interpolate(x1, t, seed=seed_vis)
        R_fm = np.linalg.norm(x_fm, axis=1)

        ax_fm = fig.add_subplot(gs[0, col])
        ax_fm.scatter(x_fm[:, 0], x_fm[:, 1], c=R_target, cmap=cmap,
                      norm=norm_color, s=s, alpha=alpha, rasterized=True, edgecolors='none')
        circle = plt.Circle((0, 0), R_median, fill=False, color='grey',
                             linestyle='--', linewidth=0.5, alpha=0.3)
        ax_fm.add_patch(circle)
        ax_fm.set_xlim(xlim); ax_fm.set_ylim(ylim)
        ax_fm.set_aspect('equal')
        ax_fm.set_xticks([]); ax_fm.set_yticks([])
        if t < 1e-8:
            ax_fm.set_title(r"$t=0$ (source)", fontsize=14, fontweight='bold')
        elif t > 1 - 1e-8:
            ax_fm.set_title(r"$t=1$ (target)", fontsize=14, fontweight='bold')
        else:
            ax_fm.set_title(f"$t={t_label}$", fontsize=14, fontweight='bold')
        if col == 0:
            ax_fm.set_ylabel(row_labels[0], fontsize=14, fontweight='bold')

        # Save individual FM panel
        title_ind = r"$t=0$ (source)" if t < 1e-8 else (r"$t=1$ (target)" if t > 1-1e-8 else f"$t={t_label}$")
        save_panel(x_fm, R_target, norm_color, cmap, R_median,
                   str(panel_dir / f"fm_t{t_label}.png"), title_ind)

        # ── RAFM ──
        x_rafm = rafm_interpolate(x1, t, seed=seed_vis)
        R_rafm = np.linalg.norm(x_rafm, axis=1)

        ax_rafm = fig.add_subplot(gs[1, col])
        ax_rafm.scatter(x_rafm[:, 0], x_rafm[:, 1], c=R_target, cmap=cmap,
                        norm=norm_color, s=s, alpha=alpha, rasterized=True, edgecolors='none')
        circle2 = plt.Circle((0, 0), R_median, fill=False, color='grey',
                              linestyle='--', linewidth=0.5, alpha=0.3)
        ax_rafm.add_patch(circle2)
        ax_rafm.set_xlim(xlim); ax_rafm.set_ylim(ylim)
        ax_rafm.set_aspect('equal')
        ax_rafm.set_xticks([]); ax_rafm.set_yticks([])
        if col == 0:
            ax_rafm.set_ylabel(row_labels[1], fontsize=14, fontweight='bold')

        # Save individual RAFM panel
        save_panel(x_rafm, R_target, norm_color, cmap, R_median,
                   str(panel_dir / f"rafm_t{t_label}.png"), title_ind)

        # ── Histogram ──
        ax_hist = fig.add_subplot(gs[2, col])
        bins = np.linspace(0, 5, 40)
        ax_hist.hist(R_target, bins=bins, density=True, alpha=0.25, color='grey',
                     label='target' if col == 0 else None)
        ax_hist.hist(R_fm, bins=bins, density=True, alpha=0.55, color='#228B22',
                     histtype='step', linewidth=1.8,
                     label='FM' if col == 0 else None)
        ax_hist.hist(R_rafm, bins=bins, density=True, alpha=0.55, color='#1f5faa',
                     histtype='step', linewidth=1.8,
                     label='RAFM' if col == 0 else None)
        ax_hist.axvline(R_median, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_hist.set_xlim(0, 5)
        ax_hist.set_yticks([])
        ax_hist.set_xlabel(r"$\|x_t\|$", fontsize=13)
        if col == 0:
            ax_hist.set_ylabel(r"$p(\|x_t\|)$", fontsize=13)
            ax_hist.legend(fontsize=9, loc='upper left', framealpha=0.7)

        # Save individual histogram
        save_histogram(R_fm, R_rafm, R_target, R_median,
                       str(panel_dir / f"hist_t{t_label}.png"),
                       show_legend=(col == 0))

    # Save composite
    composite_pdf = str(panel_dir / "figure1_composite.pdf")
    composite_png = str(panel_dir / "figure1_composite.png")
    fig.savefig(composite_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(composite_png, bbox_inches='tight', dpi=300)
    print(f"Composite: {composite_pdf}")
    plt.close(fig)

    print(f"Individual panels saved to {panel_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--out_dir", type=str, default="figures/paper_figures")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached KDE grids and recompute")
    args = parser.parse_args()

    if args.clear_cache and KDE_CACHE_DIR.exists():
        import shutil
        shutil.rmtree(KDE_CACHE_DIR)
        print("KDE cache cleared.")

    x1 = sample_target(args.n)
    times = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]
    kde_times = [0.0, 0.50, 0.75, 0.90, 1.0]
    make_figure(x1, times, args.out_dir)
    make_kde_figure(x1, kde_times, args.out_dir)


if __name__ == "__main__":
    main()
