"""Explicit, validation-only 9-candidate successive-budget source selection.

This is an experiment command, not a lightweight check. Do not execute until
the user approves docs/run_plan.md. Test rows are never scored or decoded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from baselines.tflow_core import TFlowSourceConfig
from experiments.tflow.data import load_data, sha256
from experiments.tflow.run import (train, sample, load_trained, validate_config,
                                   json_hash, write_json, implementation_sha256,
                                   run_signature, sanitize_metrics, hardware)
from experiments.tflow.validation import source_candidates, selection_score, TUNING_SEED


def select(cfg, root):
    validate_config(cfg)
    implementation_hash = implementation_sha256(cfg)
    config_hash = json_hash(cfg)
    root = Path(root) / cfg["condition_id"]
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.json"
    tuning = cfg["tuning"]
    if (tuning["seed"] != TUNING_SEED or tuning["nu"] != [3,5,7]
        or tuning["scale_multipliers"] != [0.5,1,2]
        or tuning["stage1_fraction"] != 0.05 or tuning["continuation_fraction"] != 0.10
        or tuning["n_finalists"] != 2):
        raise ValueError("Configuration differs from the documented tuning budget")
    if (isinstance(tuning["validation_generated_samples"], bool)
        or not isinstance(tuning["validation_generated_samples"], int)
        or tuning["validation_generated_samples"] < 2
        or (cfg["kind"] != "vector" and tuning["validation_generated_samples"] % 10)):
        raise ValueError("Validation sample count must be at least two and class-balanced downstream")
    data = load_data(cfg, include_external_test=False)
    candidates = source_candidates(data.split("train"))
    if len(candidates) != 9:
        raise ValueError("The documented tuning grid requires exactly nine candidates")
    validation = data.split("val")
    steps = cfg["training"]["steps"]
    first_budget, final_budget = round(0.05 * steps), round(0.10 * steps)
    history = []

    def check_row(row, index, budget):
        choice = candidates[index]
        expected_run = json_hash(run_signature(cfg, TUNING_SEED,
                                               TFlowSourceConfig(choice["nu"], choice["scale"]),
                                               stage="tuning"))
        checkpoint = row.get("checkpoint", {})
        if (row.get("config_sha256") != config_hash
            or row.get("implementation_sha256") != implementation_hash
            or row.get("status") != "complete" or row.get("candidate_index") != index
            or row.get("source") != choice or row.get("steps") != budget
            or checkpoint.get("step") != budget or checkpoint.get("stage") != "tuning"
            or checkpoint.get("run_sha256") != expected_run
            or not re.fullmatch(r"[a-f0-9]{64}", checkpoint.get("sha256", ""))):
            raise ValueError("Cached validation score provenance mismatch")
        metrics, nonfinite = sanitize_metrics(row.get("metrics", {}))
        if nonfinite:
            raise ValueError("Cached validation score has nonfinite metrics")
        for name in ("selection_score", "radial_ks", "projected_ks_mean"):
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("Cached validation score is not a bounded CDF statistic")
        if (metrics.get("projection_seed") != tuning["projection_seed"]
            or metrics.get("n_projections") != 64
            or metrics.get("n_generated") != tuning["validation_generated_samples"]
            or metrics.get("n_validation") != len(validation)
            or not math.isclose(metrics["selection_score"],
                                0.5 * (metrics["radial_ks"] + metrics["projected_ks_mean"]),
                                rel_tol=0, abs_tol=1e-15)):
            raise ValueError("Cached validation metric protocol mismatch")
        for name in ("training_time_s", "sample_time_s"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Cached validation timing must be finite and nonnegative")
        return row

    if selection_path.exists():
        saved = json.loads(selection_path.read_text())
        if saved.get("config_sha256") != config_hash or saved.get("implementation_sha256") != implementation_hash:
            raise ValueError("Existing source selection has a different configuration or implementation")
        if (saved.get("status") != "frozen" or saved.get("selection_split") != "validation"
            or saved.get("test_data_used_for_selection") is not False
            or saved.get("condition_id") != cfg["condition_id"]
            or saved.get("tuning_seed") != TUNING_SEED
            or len(saved.get("stage1_trials", [])) != 9
            or len(saved.get("finalist_trials", [])) != 2):
            raise ValueError("Existing source selection is not a complete validation-only selection")
        rows = [check_row(row, index, first_budget) for index, row in enumerate(saved["stage1_trials"])]
        finalists = sorted(rows, key=lambda row: (row["metrics"]["selection_score"], row["candidate_index"]))[:2]
        for row, first in zip(saved["finalist_trials"], finalists):
            check_row(row, first["candidate_index"], final_budget)
        winner = min(saved["finalist_trials"], key=lambda row: (row["metrics"]["selection_score"], row["candidate_index"]))
        if saved.get("selected") != winner["source"] or saved.get("selection_score") != winner["metrics"]:
            raise ValueError("Frozen source does not match the documented validation winner")
        return saved

    def trial(index, budget):
        choice = candidates[index]
        source = TFlowSourceConfig(choice["nu"], choice["scale"])
        output = root / f"candidate_{index:02d}"
        if (output / "failed.json").exists():
            raise RuntimeError(f"Candidate {index} previously failed; review preserved failure")
        cached_score = output / f"validation_step{budget}.json"
        if cached_score.exists():
            # A finalist checkpoint may already be at 10%. Its immutable 5%
            # score remains valid: never request a shorter training budget or
            # substitute the later checkpoint for the original 5% evaluation.
            return check_row(json.loads(cached_score.read_text()), index, budget)
        try:
            train(cfg, data, output, TUNING_SEED, source, budget=budget, stage="tuning")
            model, saved = load_trained(cfg, data, output, TUNING_SEED, source, stage="tuning")
            if saved["step"] != budget:
                raise ValueError("Validation must evaluate the exact requested tuning checkpoint")
            generated = sample(cfg, model, source, tuning["validation_generated_samples"], seed=tuning["sample_seed"])
            score = selection_score(generated["samples"], validation, projection_seed=tuning["projection_seed"])
            row = {"candidate_index":index, "source":choice, "steps":budget, "status":"complete",
                   "metrics":score, "config_sha256":config_hash, "implementation_sha256":implementation_hash,
                   "checkpoint":{"path":str((output / "checkpoint.pt").resolve()),
                                 "sha256":sha256(output / "checkpoint.pt"), "step":saved["step"],
                                 "run_sha256":saved["run_sha256"], "stage":saved["stage"]},
                   "training_time_s":saved["train_time_s"], "sample_time_s":generated["sample_time_s"]}
            check_row(row, index, budget)
            write_json(output / f"validation_step{budget}.json", row)
            del model
            return row
        except Exception as error:
            write_json(output / "failed.json", {"status":"failed", "candidate_index":index,
                       "steps":budget, "source":choice, "config_sha256":config_hash,
                       "implementation_sha256":implementation_hash, "stage":"tuning", "hardware":hardware(),
                       "error_type":type(error).__name__, "error":str(error)})
            # No automatic rerun or solver/scale workaround after a failure.
            raise

    for index in range(len(candidates)):
        history.append(trial(index, first_budget))
        write_json(root / "tuning_history.json", {"status":"in_progress", "trials":history,
                   "config_sha256":config_hash, "implementation_sha256":implementation_hash})
    order = sorted(history, key=lambda row:(row["metrics"]["selection_score"],row["candidate_index"]))
    finalists = [trial(row["candidate_index"], final_budget) for row in order[:2]]
    winner = min(finalists, key=lambda row:(row["metrics"]["selection_score"],row["candidate_index"]))
    result = {"status":"frozen", "selection_split":"validation", "config_sha256":config_hash,
              "condition_id":cfg["condition_id"], "tuning_seed":TUNING_SEED, "implementation_sha256":implementation_hash,
              "selected":winner["source"], "stage1_trials":history, "finalist_trials":finalists,
              "selection_score":winner["metrics"], "full_training_equivalents":0.55,
              "test_data_used_for_selection":False, "final_training":"restart all three paper seeds from scratch"}
    write_json(selection_path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs_tflow/tuning")
    args = parser.parse_args()
    print(json.dumps(select(json.loads(Path(args.config).read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
