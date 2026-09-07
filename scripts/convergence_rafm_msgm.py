"""Convergence experiment: RAFM vs MSGM on Student-t d=32.

Trains both models for 10k steps, logging loss and computing
radial_w1 + sliced_w1 every eval_every steps on a fixed set of samples.

Usage:
    python scripts/convergence_rafm_msgm.py
    python scripts/convergence_rafm_msgm.py --eval_every 500
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.data.student_t import StudentT
from rafm.models.mlp import MLP
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.flow_matching.loss import cfm_loss
from rafm.flow_matching.sampler import Sampler
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.utils.seeds import set_all_seeds


def evaluate(model, source, test_data, cfg, dim, device):
    """Generate samples and compute metrics."""
    sample_cfg = dict(cfg)
    sample_cfg["path"] = "spherical_geodesic"
    sampler = Sampler(model, source, sample_cfg)
    gen = sampler.sample(5000, dim)
    samples = gen["samples"]
    r = radial_metrics(samples, test_data)
    d = distributional_metrics(samples, test_data, n_projections=200)
    return {"radial_w1": r["radial_w1"], "sliced_w1": d["sliced_w1"]}


def train_rafm(dataset, cfg, out_dir, eval_every, n_steps, seed=99999):
    """Train RAFM with periodic evaluation."""
    set_all_seeds(seed)
    dim = dataset.dim
    device = cfg["device"]

    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    source = RadialEmpiricalSource(mode="ecdf").fit(train_data)
    path = SphericalGeodesicPath()
    model = MLP(input_dim=dim, hidden_dim=cfg["hidden_dim"],
                n_layers=cfg["n_layers"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])

    train_gpu = train_data.to(device)
    n_train = train_gpu.shape[0]
    batch_size = cfg["batch_size"]

    log_path = out_dir / "rafm_convergence.csv"
    rows = []

    t0 = time.time()
    pbar = tqdm(range(1, n_steps + 1), desc="RAFM")
    for step in pbar:
        idx = torch.randint(n_train, (batch_size,), device=device)
        x1 = train_gpu[idx]
        optimizer.zero_grad(set_to_none=True)
        loss = cfm_loss(model, path, source, x1, device=device)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % eval_every == 0:
            elapsed = time.time() - t0
            metrics = evaluate(model, source, test_data, cfg, dim, device)
            row = {
                "step": step,
                "loss": loss.item(),
                "elapsed_s": elapsed,
                "radial_w1": metrics["radial_w1"],
                "sliced_w1": metrics["sliced_w1"],
            }
            rows.append(row)
            print(f"  RAFM step={step}: loss={loss.item():.4f} "
                  f"radial_w1={metrics['radial_w1']:.4f} "
                  f"sliced_w1={metrics['sliced_w1']:.4f}")

            # Save incrementally
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    pbar.close()
    print(f"RAFM done in {time.time() - t0:.1f}s")
    return rows


def train_msgm(dataset, cfg, out_dir, eval_every, n_steps, seed=99999):
    """Train MSGM with periodic evaluation."""
    set_all_seeds(seed)
    dim = dataset.dim
    device = cfg["device"]

    test_data = dataset.get_test_data()
    train_data = dataset.get_train_data()

    # Build MSGM model
    from baselines.msgm_adapter import MSGMAdapter
    adapter_cfg = {
        "dim": dim,
        "device": device,
        "n_train_steps": n_steps,
        "batch_size": cfg["batch_size"],
        "lr": 1e-3,
        "nfe": 128,
    }
    adapter = MSGMAdapter(train_data, adapter_cfg, device=device)
    adapter.build()

    # Build the reverse SDE (same as adapter.train() does internally)
    from SDEs import PluginReverseSDE
    model = adapter._model.to(device)
    gen_sde = PluginReverseSDE(adapter._sde, model, T=adapter._sde.T,
                                debias=False).to(device)
    optimizer = Adam(model.parameters(), lr=adapter_cfg["lr"])

    train_gpu = train_data.to(device)
    n_train = train_gpu.shape[0]
    batch_size = adapter_cfg["batch_size"]

    log_path = out_dir / "msgm_convergence.csv"
    rows = []

    t0 = time.time()
    pbar = tqdm(range(1, n_steps + 1), desc="MSGM")
    for step in pbar:
        idx = torch.randint(n_train, (batch_size,), device=device)
        x = train_gpu[idx]
        optimizer.zero_grad(set_to_none=True)
        loss = gen_sde.ssm(x).mean()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % eval_every == 0:
            elapsed = time.time() - t0
            # Generate samples with MSGM
            model.eval()
            with torch.no_grad():
                x0 = adapter._sde.latent_sample(5000, dim)
                from sde_scheme import rk4_stratonovich_sampler
                samples = rk4_stratonovich_sampler(
                    adapter._sde, x0, num_steps=128, keep_all_samples=False
                ).cpu()
            model.train()

            r = radial_metrics(samples, test_data)
            d = distributional_metrics(samples, test_data, n_projections=200)
            row = {
                "step": step,
                "loss": loss.item(),
                "elapsed_s": elapsed,
                "radial_w1": r["radial_w1"],
                "sliced_w1": d["sliced_w1"],
            }
            rows.append(row)
            print(f"  MSGM step={step}: loss={loss.item():.4f} "
                  f"radial_w1={r['radial_w1']:.4f} "
                  f"sliced_w1={d['sliced_w1']:.4f}")

            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    pbar.close()
    print(f"MSGM done in {time.time() - t0:.1f}s")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="outputs/convergence_rafm_msgm")
    parser.add_argument("--seed", type=int, default=99999,
                        help="Seed (different from exp1 seeds: 8925, 65457, 77395)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["rafm", "msgm"],
                        help="Run only this method (default: both)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "device": device,
        "hidden_dim": 128,
        "n_layers": 3,
        "lr": 1e-3,
        "batch_size": 256,
        "solver": "rk4",
        "nfe": 128,
    }

    print("Loading Student-t d=32...")
    dataset = StudentT(dim=32, df=3.0, correlated=True)

    seed = args.seed
    print(f"Seed: {seed} (exp1 used: 8925, 65457, 77395)")

    if args.method is None or args.method == "rafm":
        print("\n=== Training RAFM ===")
        train_rafm(dataset, cfg, out_dir, args.eval_every, args.n_steps, seed)

    if args.method is None or args.method == "msgm":
        print("\n=== Training MSGM ===")
        train_msgm(dataset, cfg, out_dir, args.eval_every, args.n_steps, seed)

    print(f"\nResults saved to {out_dir}/")
    print("  rafm_convergence.csv")
    print("  msgm_convergence.csv")


if __name__ == "__main__":
    main()
