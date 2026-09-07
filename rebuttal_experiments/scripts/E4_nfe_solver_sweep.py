"""E4 — NFE x solver sweep on a FIXED trained checkpoint (+ radius-drift diagnostics).

Fixes the paper's noisy Exp4 (which re-trained per point) by loading ONE checkpoint
and only varying the sampler. Also measures radius drift |||x_t|| - ||x_0||| along the
trajectory, and (for spherical models) compares tangent-projection ON vs OFF.

Reuses rafm model/source/_project_tangent without modifying rafm/.

Usage:
  python rebuttal_experiments/scripts/E4_nfe_solver_sweep.py \
      --run_dir rebuttal_experiments/raw_results/E1_reproduction/student_t_d16_df3.0_cor/rafm_empirical/seed_8925 \
      --path spherical_geodesic --out rebuttal_experiments/raw_results/E4_nfe_solver/rafm_empirical_d16.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.exp0_source_diagnostics import DATASETS_FACTORY  # noqa
from experiments.exp1_main_benchmark import build_source  # noqa
from rafm.models.mlp import MLP  # noqa
from rafm.flow_matching.sampler import _project_tangent  # noqa
from rafm.metrics.radial import radial_metrics  # noqa
from rafm.metrics.distributional import distributional_metrics  # noqa
from rafm.metrics.angular import angular_metrics  # noqa

SOLVER_ORDER = {"euler": 1, "heun": 2, "rk4": 4}
NFE_TARGETS = [1, 2, 5, 10, 20, 50, 100]


def integrate(model, x0, steps, solver, project, device, record_drift=True, angular=False):
    """Fixed-step integrator returning final x and drift stats."""
    dt = 1.0 / steps
    x = x0.clone()
    r0 = x0.norm(dim=-1)
    drift_max = torch.zeros(x.shape[0], device=device)

    def v(x, t):
        vv = model(x, t)
        if angular:                                        # reconstruct velocity v=||x||*A
            vv = x.norm(dim=-1, keepdim=True).clamp(min=1e-8) * vv
        return _project_tangent(vv, x) if project else vv

    for i in range(steps):
        t0 = torch.full((x.shape[0],), i * dt, device=device)
        if solver == "euler":
            x = x + dt * v(x, t0)
        elif solver == "heun":
            tn = torch.full((x.shape[0],), (i + 1) * dt, device=device)
            v1 = v(x, t0); xp = x + dt * v1; v2 = v(xp, tn)
            x = x + dt * 0.5 * (v1 + v2)
        else:  # rk4
            tm = torch.full((x.shape[0],), (i + 0.5) * dt, device=device)
            tn = torch.full((x.shape[0],), (i + 1) * dt, device=device)
            k1 = v(x, t0); k2 = v(x + dt / 2 * k1, tm)
            k3 = v(x + dt / 2 * k2, tm); k4 = v(x + dt * k3, tn)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if record_drift:
            drift_max = torch.maximum(drift_max, (x.norm(dim=-1) - r0).abs())
    drift_final = (x.norm(dim=-1) - r0).abs()
    return x, drift_final, drift_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--path", choices=["euclidean", "spherical_geodesic"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_gen", type=int, default=10000)
    ap.add_argument("--gen_seed", type=int, default=12345)
    ap.add_argument("--dataset_pt", default=None,
                    help="real dataset .pt; if given, use its chronological test/train split (bypasses synthetic factory)")
    ap.add_argument("--angular", action="store_true", help="angular checkpoint: reconstruct v=||x||*A during sampling")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = cfg["dataset"]["dim"]
    if args.dataset_pt:
        sys.path.insert(0, str(REPO / "rebuttal_experiments"))
        from lib.real_dataset import RealTabularDataset  # noqa
        dataset = RealTabularDataset(args.dataset_pt, cfg["dataset"]["name"])
    else:
        dataset = DATASETS_FACTORY[cfg["dataset"]["name"]](cfg["dataset"])
    test = dataset.get_test_data()
    train = dataset.get_train_data()

    # rebuild model + load checkpoint (strip any compile prefix just in case)
    model = MLP(input_dim=dim, hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("n_layers", 3), premodule=cfg.get("premodule"))
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()

    # source inferred from the METHOD directory name (run_dir.name is "seed_*").
    method_name = run_dir.parent.name
    src_name = "radial_empirical_ecdf" if ("empirical" in method_name or "angular" in method_name) else (
        "radial_oracle" if "oracle" in method_name else "gaussian")
    print(f"[E4] method={method_name} source={src_name} path={args.path}", flush=True)
    source = build_source(src_name, train, dataset)

    is_spherical = args.path == "spherical_geodesic"
    project_modes = [True, False] if is_spherical else [False]

    rows = []
    for solver, order in SOLVER_ORDER.items():
        for nfe_t in NFE_TARGETS:
            steps = max(1, round(nfe_t / order))
            actual_nfe = steps * order
            for project in project_modes:
                torch.manual_seed(args.gen_seed)
                x0 = source.sample(args.n_gen, dim, device=device)
                t0 = time.time()
                with torch.no_grad():
                    x, drift_final, drift_max = integrate(
                        model, x0, steps, solver, project, device, angular=args.angular)
                dt_s = time.time() - t0
                samples = x.cpu()
                m = {}
                m.update(radial_metrics(samples, test))
                m.update(distributional_metrics(samples, test, n_projections=200))
                m.update(angular_metrics(samples, test, n_bins=4, n_projections=100))
                nan = float(torch.isnan(samples).any(dim=1).float().mean())
                rows.append({
                    "solver": solver, "target_nfe": nfe_t, "actual_nfe": actual_nfe,
                    "steps": steps, "project_tangent": project,
                    "radial_w1": m["radial_w1"], "sliced_w1": m["sliced_w1"],
                    "angular_sw_mean": m["angular_sw_mean"], "ks_stat": m["ks_stat"],
                    "sample_time_s": dt_s, "nan_rate": nan,
                    "drift_final_mean": float(drift_final.mean()),
                    "drift_final_max": float(drift_final.max()),
                    "drift_traj_max_mean": float(drift_max.mean()),
                })
                print(f"  {solver:5} nfe={actual_nfe:<4} proj={project!s:5} "
                      f"radial_w1={m['radial_w1']:.4f} sliced_w1={m['sliced_w1']:.4f} "
                      f"drift_final={float(drift_final.mean()):.2e} nan={nan:.3f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
