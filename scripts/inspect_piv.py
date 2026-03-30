"""Inspect the raw PIV dataset without modifying any existing code.

Parses DaVis .txt snapshots, computes velocity fields, vorticity, and
radial norm distribution. Saves figures and a summary JSON.

Usage:
    python scripts/inspect_piv.py --zip dataverse_files.zip --max-files 200 --seed 42
    python scripts/inspect_piv.py --raw-dir data/piv/raw --max-files 200
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NX = 545
NY = 740
FEATURE_DIM = 32  # final dimensionality (spatial subsample 4×8)
SAMPLE_NY = 8
SAMPLE_NX = 4


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_davis_txt(content: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one DaVis .txt file.

    Returns:
        x  (NY, NX) grid x-coordinates [mm]
        y  (NY, NX) grid y-coordinates [mm]
        Vx (NY, NX) x-velocity [m/s]
        Vy (NY, NX) y-velocity [m/s]
    """
    lines = content.decode(errors="replace").split("\n")
    xs, ys, vxs, vys = [], [], [], []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 4:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
            vxs.append(float(parts[2]))
            vys.append(float(parts[3]))
        except ValueError:
            continue

    n = NX * NY
    if len(xs) != n:
        raise ValueError(f"Expected {n} points, got {len(xs)}")

    x  = np.array(xs,  dtype=np.float32).reshape(NY, NX)
    y  = np.array(ys,  dtype=np.float32).reshape(NY, NX)
    Vx = np.array(vxs, dtype=np.float32).reshape(NY, NX)
    Vy = np.array(vys, dtype=np.float32).reshape(NY, NX)
    return x, y, Vx, Vy


def compute_vorticity(Vx: np.ndarray, Vy: np.ndarray) -> np.ndarray:
    """Vorticity = dVy/dx - dVx/dy via central finite differences."""
    return np.gradient(Vy, axis=1) - np.gradient(Vx, axis=0)


def subsample_vorticity(omega: np.ndarray) -> np.ndarray:
    """Uniform 4×8 spatial subsample → 32-D feature vector."""
    y_idx = np.linspace(0, NY - 1, SAMPLE_NY, dtype=int)
    x_idx = np.linspace(0, NX - 1, SAMPLE_NX, dtype=int)
    return omega[np.ix_(y_idx, x_idx)].flatten()


# ---------------------------------------------------------------------------
# File iterator
# ---------------------------------------------------------------------------

def iter_files(zip_path=None, raw_dir=None, max_files=None, seed=42):
    """Yield (name, bytes) for Serie_*.txt files."""
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(n for n in zf.namelist()
                           if Path(n).name.startswith("Serie_") and n.endswith(".txt"))
            rng = random.Random(seed)
            if max_files and len(names) > max_files:
                names = rng.sample(names, max_files)
            for name in names:
                with zf.open(name) as f:
                    yield Path(name).name, f.read()
    elif raw_dir is not None:
        raw_dir = Path(raw_dir)
        files = sorted(raw_dir.glob("Serie_*.txt"))
        rng = random.Random(seed)
        if max_files and len(files) > max_files:
            files = rng.sample(files, max_files)
        for fp in files:
            yield fp.name, fp.read_bytes()
    else:
        raise ValueError("Provide --zip or --raw-dir")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect raw PIV dataset.")
    parser.add_argument("--zip", default="dataverse_files.zip",
                        help="Path to dataverse_files.zip (default: dataverse_files.zip)")
    parser.add_argument("--raw-dir", default=None,
                        help="Directory with Serie_*.txt if already unzipped")
    parser.add_argument("--max-files", type=int, default=200,
                        help="Max number of snapshots to load (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig-dir", default="figures/piv_inspection")
    parser.add_argument("--out-dir", default="outputs/piv_inspection")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    out_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.zip) if args.zip and not args.raw_dir else None
    raw_dir  = Path(args.raw_dir) if args.raw_dir else None

    # ------------------------------------------------------------------
    # Load snapshots
    # ------------------------------------------------------------------
    print(f"Loading up to {args.max_files} snapshots ...")
    snapshots = []   # list of dicts with fields: Vx, Vy, omega, feat, name
    x_grid = y_grid = None

    for name, content in iter_files(zip_path, raw_dir, args.max_files, args.seed):
        try:
            x, y, Vx, Vy = parse_davis_txt(content)
            if np.any(np.isnan(Vx)) or np.any(np.isnan(Vy)):
                continue
            omega = compute_vorticity(Vx, Vy)
            feat  = subsample_vorticity(omega)
            snapshots.append({"name": name, "Vx": Vx, "Vy": Vy, "omega": omega, "feat": feat})
            if x_grid is None:
                x_grid, y_grid = x, y
        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"Loaded {len(snapshots)} snapshots.")
    if not snapshots:
        print("No snapshots loaded — check the path.")
        return

    features = np.stack([s["feat"] for s in snapshots])  # (N, 32)

    # Normalise features the same way as prepare_piv: /2.5, center
    features_norm = features / 2.5
    features_norm = features_norm - features_norm.mean(axis=0)

    # Radial norms
    R = np.linalg.norm(features_norm, axis=1)  # (N,)

    # ------------------------------------------------------------------
    # 1.  Velocity magnitude fields (mean over snapshots)
    # ------------------------------------------------------------------
    mean_Vx    = np.mean([s["Vx"]    for s in snapshots], axis=0)
    mean_Vy    = np.mean([s["Vy"]    for s in snapshots], axis=0)
    mean_omega = np.mean([s["omega"] for s in snapshots], axis=0)
    mean_speed = np.sqrt(mean_Vx**2 + mean_Vy**2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    xv, yv = x_grid[0, :], y_grid[:, 0]

    im0 = axes[0].pcolormesh(xv, yv, mean_Vx,    cmap="RdBu_r", shading="auto")
    axes[0].set_title("Mean Vx [m/s]")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(xv, yv, mean_speed, cmap="viridis",  shading="auto")
    axes[1].set_title("Mean ||V|| [m/s]")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(xv, yv, mean_omega, cmap="seismic",  shading="auto")
    axes[2].set_title("Mean Vorticity")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_fields.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved mean_fields.png")

    # ------------------------------------------------------------------
    # 2.  Radial norm distribution: histogram, CDF, survival, log-hist
    # ------------------------------------------------------------------
    R_gauss = np.random.default_rng(args.seed).standard_normal((10_000, FEATURE_DIM))
    R_gauss = np.linalg.norm(R_gauss, axis=1)
    # match std to PIV
    R_gauss = R_gauss * R.std() / R_gauss.std()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Histogram
    bins = np.linspace(0, R.max() * 1.05, 50)
    axes[0].hist(R,       bins=bins, density=True, alpha=0.7, label="PIV")
    axes[0].hist(R_gauss, bins=bins, density=True, alpha=0.5, label="Gaussian")
    axes[0].set_title("Radial histogram")
    axes[0].set_xlabel("R = ||x||")
    axes[0].legend()

    # CDF
    R_sorted  = np.sort(R)
    Rg_sorted = np.sort(R_gauss)
    qs = np.linspace(0, 1, len(R_sorted))
    axes[1].plot(R_sorted,  qs, label="PIV")
    axes[1].plot(Rg_sorted, np.linspace(0, 1, len(Rg_sorted)), label="Gaussian")
    axes[1].set_title("CDF of R")
    axes[1].set_xlabel("R")
    axes[1].legend()

    # Survival (log-scale y)
    axes[2].semilogy(R_sorted,  1 - qs + 1e-9, label="PIV")
    axes[2].semilogy(Rg_sorted, 1 - np.linspace(0, 1, len(Rg_sorted)) + 1e-9, label="Gaussian")
    axes[2].set_title("Survival P(R > r)")
    axes[2].set_xlabel("R")
    axes[2].legend()

    # Log-histogram
    log_R = np.log(R[R > 0])
    bins_log = np.linspace(log_R.min(), log_R.max(), 40)
    axes[3].hist(log_R, bins=bins_log, density=True)
    axes[3].set_title("Histogram of log(R)")
    axes[3].set_xlabel("log R")

    fig.tight_layout()
    fig.savefig(fig_dir / "radial_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved radial_distribution.png")

    # ------------------------------------------------------------------
    # 3.  Gallery: 3 low-R / 3 median-R / 3 high-R snapshots
    # ------------------------------------------------------------------
    order = np.argsort(R)
    n = len(order)
    gallery_idx = (
        list(order[:3])                               # low-R
        + list(order[n//2 - 1 : n//2 + 2])           # median-R
        + list(order[-3:])                            # high-R
    )
    labels = ["low-R"] * 3 + ["median-R"] * 3 + ["high-R"] * 3

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for ax, idx, lbl in zip(axes.flat, gallery_idx, labels):
        s = snapshots[idx]
        vmax = np.percentile(np.abs(s["omega"]), 98)
        ax.pcolormesh(xv, yv, s["omega"],
                      cmap="seismic", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"{lbl}  R={R[idx]:.2f}  ({s['name']})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Vorticity gallery — low / median / high R", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "gallery_by_R.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved gallery_by_R.png")

    # ------------------------------------------------------------------
    # 4.  Feature scatter (first 2 dims of normalised features)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(features_norm[:, 0], features_norm[:, 1],
                    c=R, cmap="plasma", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="R")
    ax.set_title("Feature space (dim 0 vs dim 1), coloured by R")
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved feature_scatter.png")

    # ------------------------------------------------------------------
    # 5.  Summary JSON
    # ------------------------------------------------------------------
    qs_R = np.quantile(R, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    summary = {
        "n_snapshots":    int(len(snapshots)),
        "feature_dim":    FEATURE_DIM,
        "grid_nx":        NX,
        "grid_ny":        NY,
        "R_mean":         float(R.mean()),
        "R_std":          float(R.std()),
        "R_min":          float(R.min()),
        "R_max":          float(R.max()),
        "R_q05":          float(qs_R[0]),
        "R_q25":          float(qs_R[1]),
        "R_q50":          float(qs_R[2]),
        "R_q75":          float(qs_R[3]),
        "R_q95":          float(qs_R[4]),
        "R_q99":          float(qs_R[5]),
        "R_gauss_mean":   float(R_gauss.mean()),
        "R_gauss_std":    float(R_gauss.std()),
        "pct_above_2std": float(np.mean(R > R.mean() + 2 * R.std())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved summary.json")

    # ------------------------------------------------------------------
    # 6.  README
    # ------------------------------------------------------------------
    readme = f"""# PIV Dataset Inspection

Generated by `scripts/inspect_piv.py`.

## Dataset
- **Source**: DOI 10.57745/DHJXM6 (PIV, cylinder at Re=3900)
- **Snapshots loaded**: {len(snapshots)} (out of up to {args.max_files} requested)
- **Grid**: {NX} × {NY} (x × y)
- **Feature dim**: {FEATURE_DIM} (4×8 uniform subsample of vorticity field, /2.5, centered)

## Radial norm R = ||x||  (feature-space)
| Stat | Value |
|------|-------|
| mean | {R.mean():.4f} |
| std  | {R.std():.4f} |
| q50  | {qs_R[2]:.4f} |
| q95  | {qs_R[4]:.4f} |
| q99  | {qs_R[5]:.4f} |
| max  | {R.max():.4f} |
| % > mean+2σ | {summary['pct_above_2std']*100:.1f}% |

## Figures
- `mean_fields.png`   — time-averaged Vx, speed, vorticity
- `radial_distribution.png` — R histogram, CDF, survival, log-histogram (vs Gaussian)
- `gallery_by_R.png`  — vorticity snapshots sorted by R (low/median/high)
- `feature_scatter.png` — 2D feature scatter coloured by R
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print("Saved README.md")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n=== PIV Inspection Summary ===")
    print(f"  Snapshots: {len(snapshots)}")
    print(f"  R  — mean={R.mean():.3f}  std={R.std():.3f}  "
          f"q50={qs_R[2]:.3f}  q99={qs_R[5]:.3f}  max={R.max():.3f}")
    print(f"  Gaussian R — mean={R_gauss.mean():.3f}  std={R_gauss.std():.3f}")
    frac_heavy = np.mean(R > R_gauss.max())
    print(f"  Fraction of PIV R beyond Gaussian max: {frac_heavy*100:.1f}%")
    print(f"\nFigures → {fig_dir}/")
    print(f"Outputs → {out_dir}/")


if __name__ == "__main__":
    main()
