"""Materialize reviewed configuration drafts using stdlib only; runs no models."""
import argparse
import json
from pathlib import Path


def pinned(inputs, name):
    item = inputs[name]
    return {"path": item["path"], "sha256": item["sha256"]}


def make_config(manifest, case):
    inputs = manifest["inputs"]
    kind = case["protocol"]
    downstream = kind in ("audio", "image")
    shape = [case["split"].get("n_total", 50000), case["dimension"]]
    split = case["split"]
    if kind == "audio":
        shape[0] = 12000
        counts = [10200, 1800, 0]
        split_kind = "random"
    elif kind == "image":
        shape[0] = 13394
        # Intentionally unfilled: never accept the combined train+val split by default.
        counts = [None, None, None]
        split_kind = "indices"
    else:
        counts = split["counts"]
        kinds = {"random_permutation": "random", "chronological": "chrono",
                 "contiguous_iid_order": "chrono"}
        if split["kind"] not in kinds:
            raise ValueError(f"Unrecognized vector split kind: {split['kind']}")
        split_kind = kinds[split["kind"]]
    data = {"input": {"path": case["cached_input_path"], "sha256": case["input_sha256"]},
            "shape": shape, "split": {"kind": split_kind, "seed": split.get("seed", split.get("generator_split_seed", 0)),
                                     **dict(zip(("n_train", "n_val", "n_test"), counts))},
            "transform": "piv_recenter" if case["family"] == "piv" else "train_center" if kind == "image" else "as_cached"}
    if "historical_split_hashes" in case:
        data["historical_split_hashes"] = case["historical_split_hashes"]
    if kind == "audio":
        data["external_test"] = pinned(inputs, "audio_test")
    elif kind == "image":
        data["labels"] = pinned(inputs, "dcae_labels")
        data["split"]["file"] = {"path": None, "sha256": None}
    model = {"kind": "mlp", "hidden_dim": 128, "n_layers": 3}
    if kind == "audio":
        model = {"kind": "audio_unet", "ch": 96, "class_dropout": 0.1}
    elif kind == "image":
        model = {"kind": "image_sit", "hidden": 384, "depth": 12, "heads": 6, "class_dropout": 0.1,
                 "source_dir": "/mnt/vast01/users/fouad.oubari/references/SiT-tflow"}
    training = {"steps": case["train_steps"], "batch_size": case["batch_size"],
                "optimizer": "AdamW" if downstream else "Adam", "lr": 2e-4 if kind == "audio" else 1e-4 if kind == "image" else 1e-3,
                "betas": [0.9, 0.95] if downstream else [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0, "gradient_clip": None,
                "ema": 0.999 if kind == "audio" else 0.9999 if kind == "image" else None,
                "precision": "bfloat16_autocast" if downstream else "float32",
                "batch_rule": "step_seeded" if downstream else "global_torch",
                "compile": not downstream, "checkpoint_every": 2000 if downstream else 5000,
                "log_every": 500 if downstream else 200, "class_dropout": 0.1 if downstream else 0.0}
    evaluation = {"n_samples": case["n_generated"], "model_evaluations": case["actual_nfe"],
                  "sample_batch_size": 128 if kind == "audio" else 512 if kind == "image" else case["n_generated"],
                  "sample_seed": 0 if downstream else None, "metric_seed": 0,
                  "metric_seed_note": "New evaluations use fixed shared projections; historical projection RNG state is unavailable."}
    if kind == "audio":
        evaluation["classifier"] = pinned(inputs, "audio_classifier")
    elif kind == "image":
        evaluation.update(device="cuda:0", real_n=3925, metric_seed=2020, real_reference_dir=inputs["dcae_reference"]["path"], expected_reference_sha256=None,
                          decoder_path=None, inception_weights_path=None,
                          generated_image_root="/mnt/vast01/users/fouad.oubari/data/tflow/generated_images")
    return {"schema_version": 1, "condition_id": case["id"], "kind": kind,
            "protocol_status": "resolved" if case["resolved_protocol"] else "blocked",
            "blocking_issues": case["blocking_issue_ids"], "seeds": case["model_seeds"],
            "data": data, "model": model, "training": training, "evaluation": evaluation,
            "sampler": {"t_min": 0.001, "rho": 7.0, "grid": "power_sigma", "sigma_min": 0.01},
            "tuning": {"seed": 46021, "nu": [3, 5, 7], "scale_multipliers": [0.5, 1, 2],
                       "stage1_fraction": 0.05, "continuation_fraction": 0.10, "n_finalists": 2,
                       "validation_generated_samples": 1000, "sample_seed": 61717, "projection_seed": 61719},
            "provenance": {"paper_sha256": manifest["paper"]["sha256"],
                           "baseline_commit": manifest["audit"]["commit"], "source_artifacts": case["source_artifacts"],
                           "cached_input_only": True, "existing_results_unchanged": True},
            "experiment_launch_status": "requires user approval of run plan"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/tflow/suite_manifest.json")
    parser.add_argument("--output", default="configs/tflow/prepared")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in manifest["benchmarks"]:
        cfg = make_config(manifest, case)
        resources = Path(args.manifest).parent / "image_resources.json"
        if cfg["kind"] == "image" and resources.exists():
            cfg["evaluation"].update({key:value for key,value in json.loads(resources.read_text()).items() if key not in ("resource_status","split_status")})
        path = output / (case["id"] + ".json")
        path.write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
        records.append({"condition": case["id"], "protocol_status": cfg["protocol_status"]})
    print(json.dumps({"prepared": len(records), "resolved": [x["condition"] for x in records if x["protocol_status"] == "resolved"],
                      "launched": 0}, indent=2))


if __name__ == "__main__":
    main()
