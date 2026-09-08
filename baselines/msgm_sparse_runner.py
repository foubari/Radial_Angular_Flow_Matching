"""Runner for the MSGM-SPARSE baseline through our standard evaluation pipeline. Byte-for-byte the same
flow as baselines/msgm_runner.py (same datasets, seeds, metrics, resume/skip behaviour) — only the adapter
differs (MSGMSparseAdapter -> sparse generator). Writes metrics.json in the same schema as every method.
"""
import json
from pathlib import Path

import torch

from rafm.utils.io import save_samples
from rafm.utils.seeds import set_all_seeds
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics


def run_msgm_sparse(dataset, cfg: dict, seed: int, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "metrics.json").exists():
        print(f"  [msgm_sparse seed={seed}] already done, skipping")
        return

    set_all_seeds(seed)

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    try:
        from baselines.msgm_sparse_adapter import MSGMSparseAdapter
        adapter = MSGMSparseAdapter(train_data, cfg, device)
        adapter.build()
        train_stats = adapter.train(seed=seed, ckpt_dir=run_dir)
    except ImportError as e:
        print(f"  MSGM-sparse adapter failed (vendored sdeflow code not available?): {e}")
        return

    n_gen = cfg.get("n_gen_samples", 10_000)
    gen_result = adapter.sample(n_gen)
    samples = gen_result["samples"]

    save_samples(samples, run_dir / "samples.pt")

    metrics = {}
    metrics.update(radial_metrics(samples, test_data))
    metrics.update(distributional_metrics(
        samples, test_data, n_projections=cfg.get("n_projections_sw", 500)))
    metrics.update(angular_metrics(
        samples, test_data, n_bins=cfg.get("n_angular_bins", 4)))
    metrics.update(stability_metrics(samples))
    metrics["sample_time_s"] = gen_result["sample_time_s"]
    metrics["total_train_time_s"] = train_stats["total_train_time_s"]

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  [msgm_sparse seed={seed}] radial_w1={metrics['radial_w1']:.4f} "
          f"ks={metrics['ks_stat']:.4f}")
