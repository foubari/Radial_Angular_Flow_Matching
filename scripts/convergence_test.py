"""Quick convergence test: RAFM oracle + empirical at various n_train_steps.

Usage:
    python scripts/convergence_test.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.utils.io import load_config, save_samples
from rafm.utils.seeds import set_all_seeds
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.stability import stability_metrics
from experiments.exp1_main_benchmark import build_source, build_path
from experiments.exp0_source_diagnostics import DATASETS_FACTORY

STEPS_LIST = [10_000, 50_000, 100_000]
METHODS = [
    {"name": "rafm_oracle", "source": "radial_oracle", "path": "spherical_geodesic"},
    {"name": "rafm_empirical", "source": "radial_empirical_ecdf", "path": "spherical_geodesic"},
]
SEED = 42


def main():
    cfg = load_config("configs/exp1/studentt_d16.yaml")
    ds_cfg = cfg["dataset"]
    dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_base = Path("outputs/convergence_test") / dataset.name

    for method_cfg in METHODS:
        method_name = method_cfg["name"]
        source = build_source(method_cfg["source"], train_data, dataset)
        path = build_path(method_cfg["path"])

        for n_steps in STEPS_LIST:
            print(f"\n{'='*60}")
            print(f"{method_name} — {n_steps} steps")
            print(f"{'='*60}")

            set_all_seeds(SEED)
            run_cfg = dict(cfg)
            run_cfg["n_train_steps"] = n_steps
            run_cfg["device"] = device

            from rafm.models.mlp import MLP
            model = MLP(
                input_dim=dataset.dim,
                hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("n_layers", 3),
            )

            run_dir = out_base / method_name / f"steps_{n_steps}"
            run_dir.mkdir(parents=True, exist_ok=True)

            from rafm.flow_matching.trainer import Trainer
            trainer = Trainer(model, path, source, dataset, run_cfg, SEED, run_dir)
            train_stats = trainer.train()

            from rafm.flow_matching.sampler import Sampler
            sample_cfg = dict(run_cfg)
            sample_cfg["path"] = method_cfg["path"]
            sampler = Sampler(model, source, sample_cfg)
            gen_result = sampler.sample(10_000, dataset.dim)
            samples = gen_result["samples"]

            metrics = {}
            metrics.update(radial_metrics(samples, test_data))
            metrics.update(distributional_metrics(samples, test_data))
            metrics.update(stability_metrics(samples))
            metrics["n_train_steps"] = n_steps
            metrics["final_loss"] = train_stats["final_loss"]
            metrics["total_train_time_s"] = train_stats["total_train_time_s"]

            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            print(f"  loss={metrics['final_loss']:.2f}  "
                  f"radial_w1={metrics['radial_w1']:.4f}  "
                  f"ks={metrics['ks_stat']:.4f}  "
                  f"sliced_w1={metrics['sliced_w1']:.4f}  "
                  f"nan_rate={metrics['nan_rate']:.4f}  "
                  f"time={metrics['total_train_time_s']:.0f}s")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'method':<25} {'steps':>8} {'loss':>10} {'radial_w1':>10} {'ks_stat':>10} {'sliced_w1':>10} {'nan%':>8} {'time':>8}")
    print("-" * 95)
    for method_cfg in METHODS:
        for n_steps in STEPS_LIST:
            mf = out_base / method_cfg["name"] / f"steps_{n_steps}" / "metrics.json"
            if mf.exists():
                m = json.loads(mf.read_text())
                print(f"{method_cfg['name']:<25} {n_steps:>8} {m['final_loss']:>10.2f} "
                      f"{m['radial_w1']:>10.4f} {m['ks_stat']:>10.4f} "
                      f"{m['sliced_w1']:>10.4f} {m['nan_rate']*100:>7.1f}% "
                      f"{m['total_train_time_s']:>7.0f}s")


if __name__ == "__main__":
    main()
