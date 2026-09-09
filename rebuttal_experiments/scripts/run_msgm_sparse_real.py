"""Run the SPARSE MSGM baseline on a real tabular dataset (.pt) — the sparse twin of run_msgm_real.py.
Uses baselines.msgm_sparse_runner.run_msgm_sparse (sdeflow MSGMsde(denseTensor=False) + same 18727 MLP),
with the SAME protocol/args as the dense run (chrono split, batch, steps, seeds, n_gen, nfe). Writes to
<out>/<name>/msgm_sparse/seed_* (distinct from the dense msgm/ dirs). Skips completed runs; resumes.

Usage (matches the dense finance/weather runs):
  python rebuttal_experiments/scripts/run_msgm_sparse_real.py \
    --pt rebuttal_experiments/data/finance/finance_returns_raw.pt --name finance_ff49 \
    --out outputs_msgm_sparse/rebuttal/E_finance --steps 10000 --batch 4096 --seeds 3
"""
import argparse, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "rebuttal_experiments"))
from lib.real_dataset import RealTabularDataset  # noqa
from baselines.msgm_sparse_runner import run_msgm_sparse  # noqa
from rafm.utils.seeds import get_seed_list  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n_gen", type=int, default=10000)
    ap.add_argument("--split", default="chrono", choices=["chrono", "random"])
    args = ap.parse_args()

    ds = RealTabularDataset(args.pt, args.name, split=args.split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_test = len(ds.get_test_data()); n_train = len(ds.get_train_data())
    n_gen = min(args.n_gen, n_test)
    cfg = {"lr": 1e-3, "batch_size": args.batch, "n_train_steps": args.steps,
           "ckpt_every": 1000, "hidden_dim": 128, "device": device,
           "n_gen_samples": n_gen, "n_projections_sw": 500, "n_angular_bins": 4}
    print(f"MSGM-sparse on {args.name} dim={ds.dim} n_train={n_train} n_test={n_test} n_gen={n_gen} "
          f"split={args.split} device={device}", flush=True)
    for seed in get_seed_list(args.seeds, 42):
        run_dir = Path(args.out) / args.name / "msgm_sparse" / f"seed_{seed}"
        run_msgm_sparse(ds, cfg, seed, run_dir)


if __name__ == "__main__":
    main()
