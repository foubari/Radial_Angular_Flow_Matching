"""Run the MSGM baseline on a real tabular dataset (.pt), same protocol as run_real.py.

Uses baselines.msgm_runner.run_msgm (which skips completed runs and resumes from
checkpoint — safe, never overwrites). Same chronological split, seeds, n_gen, metrics.

Usage:
  python rebuttal_experiments/scripts/run_msgm_real.py --pt <.pt> --name <name> \
      --out rebuttal_experiments/raw_results/E_finance --steps 10000 --batch 4096 --seeds 3
"""
import argparse, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "rebuttal_experiments"))
from lib.real_dataset import RealTabularDataset  # noqa
from baselines.msgm_runner import run_msgm  # noqa
from rafm.utils.seeds import get_seed_list  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n_gen", type=int, default=10000)
    ap.add_argument("--split", default="chrono", choices=["chrono", "random"])
    args = ap.parse_args()

    ds = RealTabularDataset(args.pt, args.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gen = min(args.n_gen, ds.n_test)
    cfg = {"lr": 1e-3, "batch_size": args.batch, "n_train_steps": args.steps,
           "ckpt_every": 1000, "hidden_dim": 128, "device": device,
           "n_gen_samples": n_gen, "n_projections_sw": 500, "n_angular_bins": 4}
    print(f"MSGM on {args.name} dim={ds.dim} n_train={ds.n_train} n_test={ds.n_test} n_gen={n_gen} device={device}")
    for seed in get_seed_list(args.seeds, 42):
        run_dir = Path(args.out) / args.name / "msgm" / f"seed_{seed}"
        run_msgm(ds, cfg, seed, run_dir)


if __name__ == "__main__":
    main()
