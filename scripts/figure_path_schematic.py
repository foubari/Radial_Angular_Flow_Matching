"""Path geometry schematic: RAFM (geodesic arc) vs FM (straight line).

Clean scientific figure for a NeurIPS-style paper.
Single panel, pure white background, minimalist vector style.

Usage:
    python scripts/figure_path_schematic.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


def make_path_schematic(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Colors ──
    blue = "#1a5276"       # RAFM
    green = "#228B22"      # FM
    charcoal = "#2c2c2c"   # target x1
    gray_circle = "#cccccc"

    # ── Geometry ──
    R = 2.5
    # Target point x1 — upper right
    theta_x1 = np.radians(55)
    x1 = R * np.array([np.cos(theta_x1), np.sin(theta_x1)])

    # RAFM source x0 — on the circle
    theta_rafm_x0 = np.radians(175)
    rafm_x0 = R * np.array([np.cos(theta_rafm_x0), np.sin(theta_rafm_x0)])

    # FM source x0 — inside the circle, lower-left
    fm_x0 = np.array([-1.7, -1.4])

    # ── Interpolation parameter ──
    t = 0.45

    # RAFM intermediate: slerp on circle
    d_theta = theta_x1 - theta_rafm_x0
    if d_theta > np.pi:
        d_theta -= 2 * np.pi
    elif d_theta < -np.pi:
        d_theta += 2 * np.pi
    theta_rafm_xt = theta_rafm_x0 + t * d_theta
    rafm_xt = R * np.array([np.cos(theta_rafm_xt), np.sin(theta_rafm_xt)])

    # FM intermediate: linear
    fm_xt = (1 - t) * fm_x0 + t * x1

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.8, 3.8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })

    # 1. Dashed circle
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R * np.cos(theta_circle), R * np.sin(theta_circle),
            color=gray_circle, linestyle='--', linewidth=1.0, zorder=1)

    # 2. RAFM geodesic arc (with arrowhead at x_t position)
    n_arc = 200
    arc_thetas = np.linspace(theta_rafm_x0, theta_rafm_x0 + d_theta, n_arc)
    arc_x = R * np.cos(arc_thetas)
    arc_y = R * np.sin(arc_thetas)
    ax.plot(arc_x, arc_y, color=blue, linewidth=1.8, zorder=3)

    # Arrow at RAFM x_t position along arc
    i_xt = int(t * n_arc)
    dx_b = arc_x[min(i_xt + 4, n_arc - 1)] - arc_x[max(i_xt - 4, 0)]
    dy_b = arc_y[min(i_xt + 4, n_arc - 1)] - arc_y[max(i_xt - 4, 0)]
    ax.annotate("", xy=(rafm_xt[0] + dx_b * 0.3, rafm_xt[1] + dy_b * 0.3),
                xytext=(rafm_xt[0] - dx_b * 0.3, rafm_xt[1] - dy_b * 0.3),
                arrowprops=dict(arrowstyle='->', color=blue, lw=2.0,
                                mutation_scale=18),
                zorder=5)

    # 3. FM straight path (with arrowhead at x_t position)
    ax.plot([fm_x0[0], x1[0]], [fm_x0[1], x1[1]],
            color=green, linewidth=1.8, zorder=3)

    # Arrow at FM x_t position along segment
    direction = x1 - fm_x0
    direction = direction / np.linalg.norm(direction)
    ax.annotate("", xy=(fm_xt[0] + direction[0] * 0.2, fm_xt[1] + direction[1] * 0.2),
                xytext=(fm_xt[0] - direction[0] * 0.2, fm_xt[1] - direction[1] * 0.2),
                arrowprops=dict(arrowstyle='->', color=green, lw=2.0,
                                mutation_scale=18),
                zorder=5)

    # 4. Endpoint points only (x0 and x1, no x_t dots)
    pt_size = 50
    ax.scatter(*x1, color=charcoal, s=pt_size, zorder=5, edgecolors='none')
    ax.scatter(*rafm_x0, color=blue, s=pt_size, zorder=5, edgecolors='none')
    ax.scatter(*fm_x0, color=green, s=pt_size, zorder=5, edgecolors='none')

    # 5. Text labels — large, no bold
    label_fs = 19
    # x1 (shared target)
    ax.annotate(r"$x_1$", xy=x1, xytext=(12, 8),
                textcoords='offset points',
                fontsize=label_fs, color=charcoal, zorder=6)

    # RAFM x0
    ax.annotate(r"$x_0$", xy=rafm_x0, xytext=(-22, -16),
                textcoords='offset points',
                fontsize=label_fs, color=blue, zorder=6)
    # RAFM x_t (near arrow)
    ax.annotate(r"$x_t$", xy=rafm_xt, xytext=(-24, 10),
                textcoords='offset points',
                fontsize=label_fs, color=blue, zorder=6)

    # FM x0
    ax.annotate(r"$x_0$", xy=fm_x0, xytext=(-22, -16),
                textcoords='offset points',
                fontsize=label_fs, color=green, zorder=6)
    # FM x_t (near arrow)
    ax.annotate(r"$x_t$", xy=fm_xt, xytext=(-22, 12),
                textcoords='offset points',
                fontsize=label_fs, color=green, zorder=6)

    # 6. Radial dashed line from origin at -45 degrees
    r_end = R * np.array([np.cos(np.radians(-45)), np.sin(np.radians(-45))])
    ax.plot([0, r_end[0]], [0, r_end[1]], color=gray_circle, linestyle=':',
            linewidth=0.8, zorder=1)
    ax.text(r_end[0] * 0.5 + 0.2, r_end[1] * 0.5 + 0.2, r"$R$",
            fontsize=15, color='#999999', ha='left', va='bottom', zorder=6)

    # 7. Equations below — large, no bold
    eq_y = -3.20
    ax.text(0.0, eq_y,
            r"$\psi_t^{\mathrm{RAFM}}(x_0, x_1) = R\,\gamma_t(u_0, u_1)$",
            fontsize=18, color=blue, ha='center', va='top', zorder=6)
    ax.text(0.0, eq_y - 0.80,
            r"$\psi_t^{\mathrm{FM}}(x_0, x_1) = (1-t)\,x_0 + t\,x_1$",
            fontsize=18, color=green, ha='center', va='top', zorder=6)

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_path_schematic.{ext}"),
                    bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Path schematic saved to {out_dir}/figure_path_schematic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_path_schematic()
