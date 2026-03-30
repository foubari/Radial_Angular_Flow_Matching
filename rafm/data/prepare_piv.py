"""Prepare the public PIV dataset for use in RAFM experiments.

Dataset: "Non-time-resolved PIV dataset of flow over a circular cylinder
         at Reynolds number 3900", DOI 10.57745/DHJXM6

This script:
  1. Reads DaVis .txt files (x[mm], y[mm], Vx[m/s], Vy[m/s]) for each snapshot
  2. Computes vorticity (curl of velocity) on the 545×740 grid via finite differences
  3. Subsamples to 32 spatial locations (4×8 uniform grid over the domain)
  4. Applies MSGM normalisation: divide by 2.5, center
  5. Exports piv_d32.pt (32 dims) and piv_d16.pt (16 dims, first 16 components)

Usage:
    # First unzip the downloaded dataset:
    python -m rafm.data.prepare_piv --zip dataverse_files.zip --out_dir data/piv

    # Or if already unzipped:
    python -m rafm.data.prepare_piv --raw_dir data/piv/raw --out_dir data/piv

Grid structure of each DaVis file:
    Header: #DaVis 10.x 2C vector field 4 Nx Ny "x";"y";"Vx";"Vy"
    Nx=545 (x varies fast), Ny=740 (y varies slow)
    Ordering: row-major, y-constant blocks of Nx points
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


NX = 545   # x grid points (fast axis)
NY = 740   # y grid points (slow axis)

# 4×8 = 32 uniform spatial samples of vorticity
SAMPLE_NY = 8
SAMPLE_NX = 4


def parse_davis_txt(content: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse one DaVis .txt file. Returns (Vx, Vy) as (NY, NX) arrays."""
    lines = content.decode(errors="replace").split("\n")
    data = []
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 4:
            continue
        try:
            data.append((float(parts[2]), float(parts[3])))  # Vx, Vy
        except ValueError:
            continue

    if len(data) != NX * NY:
        raise ValueError(f"Expected {NX*NY} points, got {len(data)}")

    arr = np.array(data, dtype=np.float32)  # (NX*NY, 2)
    Vx = arr[:, 0].reshape(NY, NX)
    Vy = arr[:, 1].reshape(NY, NX)
    return Vx, Vy


def compute_vorticity(Vx: np.ndarray, Vy: np.ndarray) -> np.ndarray:
    """Vorticity = dVy/dx - dVx/dy via central finite differences. Returns (NY, NX)."""
    dVy_dx = np.gradient(Vy, axis=1)   # d/dx (fast axis)
    dVx_dy = np.gradient(Vx, axis=0)   # d/dy (slow axis)
    return dVy_dx - dVx_dy


def subsample_vorticity(omega: np.ndarray, ny: int = SAMPLE_NY, nx: int = SAMPLE_NX) -> np.ndarray:
    """Uniformly subsample vorticity field to ny×nx = 32 points."""
    y_idx = np.linspace(0, NY - 1, ny, dtype=int)
    x_idx = np.linspace(0, NX - 1, nx, dtype=int)
    return omega[np.ix_(y_idx, x_idx)].flatten()  # (ny*nx,)


def process_files(file_iter, total: int, grid_ny: int = SAMPLE_NY, grid_nx: int = SAMPLE_NX) -> np.ndarray:
    """Process an iterable of (name, content_bytes) pairs. Returns (N, grid_ny*grid_nx) array."""
    snapshots = []
    skipped = 0

    pbar = tqdm(file_iter, total=total, desc="Processing PIV frames")
    for name, content in pbar:
        if not name.startswith("Serie_") or not name.endswith(".txt"):
            continue
        try:
            Vx, Vy = parse_davis_txt(content)
            if np.any(np.isnan(Vx)) or np.any(np.isnan(Vy)):
                skipped += 1
                continue
            omega = compute_vorticity(Vx, Vy)
            feat = subsample_vorticity(omega, ny=grid_ny, nx=grid_nx)
            snapshots.append(feat)
        except Exception as e:
            skipped += 1
            pbar.set_postfix(skipped=skipped)

    print(f"  Processed {len(snapshots)} frames, skipped {skipped}")
    return np.stack(snapshots, axis=0)  # (N, 32)


def prepare_piv(out_dir: Path, zip_path: Path | None = None, raw_dir: Path | None = None,
                grids: list[tuple[int, int]] | None = None) -> None:
    """Prepare PIV data at one or more spatial resolutions.

    Args:
        grids: list of (ny, nx) tuples. Default: [(8, 4), (8, 8), (16, 16)]
               giving dims 32, 64, 256.
    """
    if grids is None:
        grids = [(8, 4), (8, 8), (16, 16)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for grid_ny, grid_nx in grids:
        dim = grid_ny * grid_nx
        print(f"\n--- Grid {grid_ny}x{grid_nx} = {dim}D ---")

        if zip_path is not None:
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if n.startswith("Serie_") and n.endswith(".txt")]
                def file_iter(names=names):
                    for name in names:
                        with zf.open(name) as f:
                            yield Path(name).name, f.read()
                data = process_files(file_iter(), total=len(names), grid_ny=grid_ny, grid_nx=grid_nx)
        elif raw_dir is not None:
            raw_dir_p = Path(raw_dir)
            files = sorted(raw_dir_p.glob("Serie_*.txt"))
            def file_iter(files=files):
                for f in files:
                    yield f.name, f.read_bytes()
            data = process_files(file_iter(), total=len(files), grid_ny=grid_ny, grid_nx=grid_nx)
        else:
            raise ValueError("Provide either --zip or --raw_dir")

        print(f"Raw data shape: {data.shape}")

        # MSGM normalisation: divide by 2.5, center
        data = data / 2.5
        data = data - data.mean(axis=0)

        tensor = torch.from_numpy(data).float()
        out_path = out_dir / f"piv_d{dim}.pt"
        torch.save(tensor, out_path)
        print(f"Saved {out_path} — shape {tensor.shape}")

        # Also save truncated versions for lower dims
        for d in [16, 32]:
            if d < dim:
                trunc_path = out_dir / f"piv_d{dim}_trunc{d}.pt"
                torch.save(tensor[:, :d], trunc_path)
                print(f"Saved {trunc_path} — shape {tensor[:, :d].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=None, help="Path to dataverse_files.zip")
    parser.add_argument("--raw_dir", default=None, help="Directory with Serie_*.txt files (if already unzipped)")
    parser.add_argument("--out_dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--grids", default="8x4,8x8,16x16",
                        help="Comma-separated NxM grids (default: 8x4,8x8,16x16 → d=32,64,256)")
    args = parser.parse_args()

    grids = []
    for g in args.grids.split(","):
        ny, nx = g.strip().split("x")
        grids.append((int(ny), int(nx)))

    prepare_piv(
        out_dir=args.out_dir,
        zip_path=Path(args.zip) if args.zip else None,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        grids=grids,
    )
