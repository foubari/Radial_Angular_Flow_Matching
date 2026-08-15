"""Standalone runner for a real tabular dataset (finance, weather, ...).

Reuses rafm Trainer/Sampler/metrics and the same method definitions as exp1, but for a
RealTabularDataset loaded from a .pt file. Methods: gaussian_fm, source_only_empirical,
rafm_empirical (no oracle for real data). Saves the exact same per-run layout as exp1.

Usage:
  python rebuttal_experiments/scripts/run_real.py --pt rebuttal_experiments/data/finance/finance_returns_raw.pt \
      --name finance_ff49 --out rebuttal_experiments/raw_results/E_finance --steps 10000 --batch 4096 --seeds 3
"""
import argparse, json, sys, time, platform
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rebuttal_experiments"))
from lib.real_dataset import RealTabularDataset  # noqa
from rafm.models.mlp import MLP  # noqa
from rafm.flow_matching.trainer import Trainer  # noqa
from rafm.flow_matching.sampler import Sampler  # noqa
from rafm.sources.gaussian import GaussianSource  # noqa
from rafm.sources.radial_empirical import RadialEmpiricalSource  # noqa
from rafm.paths.euclidean import EuclideanPath  # noqa
from rafm.paths.spherical_geodesic import SphericalGeodesicPath  # noqa
from rafm.metrics.radial import radial_metrics  # noqa
from rafm.metrics.distributional import distributional_metrics  # noqa
from rafm.metrics.angular import angular_metrics  # noqa
from rafm.metrics.stability import stability_metrics  # noqa
from rafm.utils.seeds import get_seed_list, set_all_seeds  # noqa

METHODS = [
    ("gaussian_fm", "gaussian", "euclidean"),
    ("source_only_empirical", "radial_empirical", "euclidean"),
    ("rafm_empirical", "radial_empirical", "spherical_geodesic"),
    ("angular_rafm", "radial_empirical", "spherical_geodesic"),   # scale-free angular target
]


def build_source(kind, train):
    if kind == "gaussian":
        return GaussianSource()
    return RadialEmpiricalSource(mode="ecdf").fit(train)


def build_path(kind):
    return EuclideanPath() if kind == "euclidean" else SphericalGeodesicPath()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nfe", type=int, default=128)
    ap.add_argument("--n_gen", type=int, default=10000)
    ap.add_argument("--methods", default=None, help="comma-separated subset")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "resmlp"])
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3, help="MLP hidden layers, or ResidualMLP blocks")
    ap.add_argument("--split", default="chrono", choices=["chrono", "random"])
    args = ap.parse_args()

    ds = RealTabularDataset(args.pt, args.name, split=args.split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = ds.get_test_data(); train = ds.get_train_data()
    n_gen = min(args.n_gen, ds.n_test) if ds.n_test < args.n_gen else args.n_gen
    seeds = get_seed_list(args.seeds, 42)
    methods = METHODS
    if args.methods:
        want = set(args.methods.split(","))
        methods = [m for m in METHODS if m[0] in want]

    print(f"dataset {args.name} dim={ds.dim} n_train={ds.n_train} n_test={ds.n_test} device={device}")
    for mname, skind, pkind in methods:
        print(f"\n== {mname} ==")
        for seed in seeds:
            set_all_seeds(seed)
            run_dir = Path(args.out) / args.name / mname / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "metrics.json").exists():   # resumable: skip completed runs
                print(f"  seed={seed} skip (metrics.json exists)", flush=True); continue
            cfg = {"lr": 1e-3, "batch_size": args.batch, "n_train_steps": args.steps,
                   "log_every": 200, "ckpt_every": 5000, "device": device,
                   "solver": "rk4", "nfe": args.nfe, "hidden_dim": args.hidden_dim,
                   "n_layers": args.n_layers, "arch": args.arch, "split": args.split,
                   "path": pkind, "source": skind, "n_gen_samples": n_gen,
                   "angular": (mname == "angular_rafm"),
                   "dataset": {"name": args.name, "dim": ds.dim}}
            if args.arch == "resmlp":
                from lib.resmlp import ResidualMLP
                model = ResidualMLP(input_dim=ds.dim, hidden_dim=args.hidden_dim, n_blocks=args.n_layers)
            else:
                model = MLP(input_dim=ds.dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
            source = build_source(skind, train)
            path = build_path(pkind)
            trainer = Trainer(model, path, source, ds, cfg, seed, run_dir)
            stats = trainer.train()
            sampler = Sampler(model, source, cfg)
            gen = sampler.sample(n_gen, ds.dim)
            samples = gen["samples"]
            torch.save(samples, run_dir / "samples.pt")
            m = {}
            m.update(radial_metrics(samples, test))
            m.update(distributional_metrics(samples, test, n_projections=500))
            m.update(angular_metrics(samples, test, n_bins=4, n_projections=200))
            m.update(stability_metrics(samples))
            m["nfe"] = gen["nfe"]; m["sample_time_s"] = gen["sample_time_s"]
            m["total_train_time_s"] = stats["total_train_time_s"]
            m["n_params"] = sum(p.numel() for p in model.parameters())
            (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
            (run_dir / "notes.md").write_text(
                f"{mname} seed={seed} on {args.name}\n"
                f"host={platform.node()} device={device}\n", encoding="utf-8")
            print(f"  seed={seed} radial_w1={m['radial_w1']:.4f} ks={m['ks_stat']:.4f} "
                  f"sliced_w1={m['sliced_w1']:.4f} nan={m['nan_rate']:.3f} "
                  f"train={m['total_train_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
