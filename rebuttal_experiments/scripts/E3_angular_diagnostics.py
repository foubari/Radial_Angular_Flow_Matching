"""E3 — Angular / direction-sensitive diagnostics (addresses reviewer R3).

Operates on SAVED samples.pt from any exp1-style results tree. Rebuilds the
(deterministic) dataset from a run's config.yaml to get held-out test data.
Everything is norm-independent (directions x/||x||) except the explicitly-labelled
"common-radial" diagnostic, which re-radialises every method with the TEST radial
law while keeping its generated directions, isolating angular quality.

Metrics per (method, seed):
  dir_sliced_w1      : sliced-Wasserstein-1 on unit directions (global, 500 proj)
  dir_mmd            : RBF-MMD on unit directions (median-heuristic bandwidth)
  nn_cos_gen2test    : mean nearest-neighbour cosine distance, gen->test dirs
  nn_cos_test2gen    : mean nearest-neighbour cosine distance, test->gen dirs (coverage)
  mean_angular_err   : mean angle (deg) to nearest test direction
  dir_sw_binN        : direction sliced-W1 within radial-quantile bin N (of test radii)
  cr_sliced_w1       : [common-radial diagnostic] full-space sliced-W1 after re-radialising
  cr_mmd             : [common-radial diagnostic] full-space RBF-MMD after re-radialising

Usage:
  python rebuttal_experiments/scripts/E3_angular_diagnostics.py \
      --results_root outputs/exp1_main_benchmark/student_t_d16_df3.0_cor \
      --out rebuttal_experiments/raw_results/E3_angular/student_t_d16.csv
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.exp0_source_diagnostics import DATASETS_FACTORY  # noqa: E402


def _rng(seed=0):
    return np.random.default_rng(seed)


def unit(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    keep = (n[:, 0] > eps) & np.isfinite(n[:, 0])
    return x[keep] / n[keep], keep


def sliced_w1(A, B, n_proj=500, seed=0, K=2000):
    """Sliced W1 between point sets A (n,d), B (m,d).

    W1 per projection = mean over a common K-point quantile grid of |Q_A - Q_B|.
    A fixed K (not min(n,m)) keeps np.quantile cheap and is ample for W1 accuracy.
    """
    rng = _rng(seed)
    d = A.shape[1]
    dirs = rng.standard_normal((d, n_proj))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    PA = A @ dirs
    PB = B @ dirs
    q = np.linspace(0.0, 1.0, K)
    qa = np.quantile(PA, q, axis=0)   # (K, n_proj)
    qb = np.quantile(PB, q, axis=0)
    return float(np.mean(np.abs(qa - qb)))


def rbf_mmd2(A, B, n=2000, seed=0):
    """RBF-MMD^2 with median-heuristic bandwidth. Uses scipy.cdist (memory-light)."""
    from scipy.spatial.distance import cdist
    rng = _rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    d2 = cdist(A, B, "sqeuclidean")
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / (med + 1e-12)
    Kab = np.exp(-gamma * d2)
    Kaa = np.exp(-gamma * cdist(A, A, "sqeuclidean"))
    Kbb = np.exp(-gamma * cdist(B, B, "sqeuclidean"))
    na, nb = len(A), len(B)
    mmd2 = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1)) \
         + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1)) \
         - 2 * Kab.mean()
    return float(max(mmd2, 0.0))


def nn_cosine(query, ref, n=3000, seed=0):
    """Mean nearest-neighbour cosine distance from query dirs to ref dirs (unit vecs)."""
    from sklearn.neighbors import NearestNeighbors
    rng = _rng(seed)
    q = query[rng.choice(len(query), min(n, len(query)), replace=False)]
    r = ref[rng.choice(len(ref), min(n, len(ref)), replace=False)]
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(r)
    dist, _ = nn.kneighbors(q)  # cosine distance = 1 - cos_sim
    return float(dist.mean()), float(np.degrees(np.arccos(np.clip(1 - dist, -1, 1))).mean())


def find_config(results_root: Path):
    for p in results_root.rglob("config.yaml"):
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, help="dir containing <method>/seed_*/samples.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_proj", type=int, default=500)
    ap.add_argument("--dataset_pt", default=None,
                    help="real dataset .pt; if given, test = chronological 60/20/20 test split of it (bypasses the synthetic factory)")
    args = ap.parse_args()

    root = Path(args.results_root)
    if args.dataset_pt:
        sys.path.insert(0, str(REPO / "rebuttal_experiments"))
        from lib.real_dataset import RealTabularDataset  # noqa
        test = RealTabularDataset(args.dataset_pt, "real").get_test_data().numpy().astype(np.float64)
    else:
        cfg_path = find_config(root)
        if cfg_path is None:
            raise SystemExit(f"No config.yaml under {root} to rebuild the dataset.")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg["dataset"]
        dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
        test = dataset.get_test_data().numpy().astype(np.float64)
    test_u, _ = unit(test)
    r_test = np.linalg.norm(test, axis=-1)
    # radial-quantile bin edges from test
    n_bins = 4
    edges = np.quantile(r_test, np.linspace(0, 1, n_bins + 1))

    rows = []
    method_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for mdir in method_dirs:
        for seed_dir in sorted(mdir.glob("seed_*")):
            spath = seed_dir / "samples.pt"
            if not spath.exists():
                continue
            samples = torch.load(spath, map_location="cpu").numpy().astype(np.float64)
            gen_u, keep = unit(samples)
            r_gen = np.linalg.norm(samples, axis=-1)
            row = {"method": mdir.name, "seed": seed_dir.name.replace("seed_", ""),
                   "n_gen": len(samples), "n_valid_dir": len(gen_u),
                   "nan_rate": float(np.mean(~np.isfinite(samples).all(axis=1)))}
            # direction metrics
            row["dir_sliced_w1"] = sliced_w1(gen_u, test_u, args.n_proj)
            row["dir_mmd"] = rbf_mmd2(gen_u, test_u)
            g2t, ang = nn_cosine(gen_u, test_u)
            row["nn_cos_gen2test"] = g2t
            row["mean_angular_err_deg"] = ang
            row["nn_cos_test2gen"], _ = nn_cosine(test_u, gen_u)
            # per radial-quantile bin direction SW (bin by each set's own radii into test edges)
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                gmask = (r_gen >= lo) & (r_gen < hi if b < n_bins - 1 else r_gen <= hi)
                tmask = (r_test >= lo) & (r_test < hi if b < n_bins - 1 else r_test <= hi)
                gu, _ = unit(samples[gmask]) if gmask.sum() > 10 else (np.empty((0, test.shape[1])), None)
                tu, _ = unit(test[tmask]) if tmask.sum() > 10 else (np.empty((0, test.shape[1])), None)
                row[f"dir_sw_bin{b}"] = sliced_w1(gu, tu, 200) if len(gu) > 10 and len(tu) > 10 else float("nan")
            # common-radial diagnostic: give this method the TEST radial law, keep its dirs
            rng = _rng(0)
            r_draw = r_test[rng.choice(len(r_test), len(gen_u), replace=True)]
            cr = gen_u * r_draw[:, None]
            row["cr_sliced_w1"] = sliced_w1(cr, test, args.n_proj)
            row["cr_mmd"] = rbf_mmd2(cr, test)
            rows.append(row)
            print(f"  {row['method']:<24} {row['seed']:<7} "
                  f"dir_sw1={row['dir_sliced_w1']:.4f} dir_mmd={row['dir_mmd']:.5f} "
                  f"ang_err={row['mean_angular_err_deg']:.2f}deg cr_sw1={row['cr_sliced_w1']:.4f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    keys = sorted({k for r in rows for k in r})
    keys = ["method", "seed"] + [k for k in keys if k not in ("method", "seed")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
