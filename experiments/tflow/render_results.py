"""Render new t-Flow additions without rebuilding historical paper tables.

This module only reads JSON and formats reports. It never imports torch, loads
checkpoints, samples data, trains models, or recomputes historical metrics.
Every expected condition and seed remains visible. An incomplete or failed
condition has no aggregate, even when two seeds succeeded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
MAIN_VECTOR = (
    "student_t_d16_df3.0_cor", "student_t_d32_df3.0_cor",
    "piv_d64", "piv_d256", "finance_ff49", "weather_au_wind",
)
TABLE3_VECTOR = MAIN_VECTOR[:4]
IMAGE_ID = "imagenette_dcae"

# Canonical reporting keys -> supported native per-domain paths.
ALIASES = {
    "vector": {},
    "audio": {
        "digit_acc": ("digit_acc", "content.digit_acc"),
        "energy_KS": ("energy_KS", "energy.ks"),
        "radial_w1": ("radial_w1", "energy.radial_w1"),
        "cov>q90": ("cov>q90", "energy.cov_gt_q90"),
        "cov>q95": ("cov>q95", "energy.cov_gt_q95"),
        "cov>q99": ("cov>q99", "energy.cov_gt_q99"),
        "cov<q10": ("cov<q10", "energy.cov_lt_q10"),
        "PIT": ("PIT", "energy.pit_mean"),
    },
    "image": {
        "fid": ("fid", "image.fid"),
        "kid": ("kid", "image.kid"),
        "kid_std": ("kid_std", "image.kid_std"),
        "radial_w1": ("radial_w1", "latent.radial_w1"),
        "ks": ("ks", "latent.ks"),
        "sliced_w1": ("sliced_w1", "latent.sliced_w1"),
        "precision": ("precision", "image.precision"),
        "recall": ("recall", "image.recall"),
        "density": ("density", "image.density"),
        "coverage": ("coverage", "image.coverage"),
    },
}
COMMON_METRICS = ("total_train_time_s", "sample_time_s", "nfe")
REQUIRED_METRICS = {
    "vector": ("radial_w1", "ks_stat", "sliced_w1", "mmd",
               "q950_err", "q990_err", "q995_err", "tail_exc_95", "tail_exc_99",
               "angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3",
               "angular_sw_mean", "nan_rate", "exploding_norm_rate", "invalid_rate", "inf_rate"),
    "audio": ("digit_acc", "energy_KS", "radial_w1", "cov>q90", "cov>q95", "cov>q99", "cov<q10", "PIT"),
    "image": ("fid", "kid", "kid_std", "radial_w1", "ks", "sliced_w1", "precision", "recall", "density", "coverage"),
}
REQUIRED_METRICS = {kind: metrics + COMMON_METRICS for kind, metrics in REQUIRED_METRICS.items()}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_metrics(metrics: dict, prefix: str = "") -> dict[str, Any]:
    """Keep native scalar keys, including diagnostic metrics outside table columns."""
    result = {}
    for key, value in metrics.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_metrics(value, full))
        else:
            result[full] = value
    return result


def normalize_metrics(metrics: dict, protocol: str) -> dict[str, Any]:
    flat = flatten_metrics(metrics)
    for canonical, aliases in ALIASES[protocol].items():
        matches = [(key, flat[key]) for key in aliases if key in flat]
        if matches:
            # Ambiguous duplicate aliases are a corrupt artifact, not a choice of
            # whichever value happens to be encountered first.
            if any(value != matches[0][1] for _, value in matches[1:]):
                raise ValueError(f"Conflicting values for metric {canonical}")
            flat[canonical] = matches[0][1]
            for key, _ in matches:
                if key != canonical:
                    del flat[key]
    # The downstream evaluator returns both measurements and provenance. Keep
    # provenance in each native_metrics payload, not in mean/SD columns.
    flat = {key: value for key, value in flat.items()
            if key not in {"method", "n", "real_n", "cfg"}
            and not key.startswith(("protocol.", "reference.", "class_counts."))}
    return flat


def _bad_numbers(value: Any, prefix: str = "metrics") -> list[str]:
    if _numeric(value):
        return [prefix] if not math.isfinite(value) else []
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _bad_numbers(v, f"{prefix}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _bad_numbers(v, f"{prefix}[{i}]")]
    return []


def _safe_json(value: Any) -> Any:
    """A failed artifact may contain NaN; its original file is never rewritten."""
    if _numeric(value) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


def _validate_condition(condition: dict) -> None:
    if not SAFE_ID.fullmatch(condition.get("id", "")):
        raise ValueError("A condition id must be a safe single path component")
    seeds = condition.get("model_seeds", [])
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int for s in seeds):
        raise ValueError(f"{condition['id']}: exactly three distinct expected model seeds are required")
    if condition.get("protocol") not in REQUIRED_METRICS:
        raise ValueError(f"{condition['id']}: unsupported reporting protocol")


def inspect_run(path: Path, condition: dict, seed: int) -> dict:
    """Read the exact final-run envelope; never infer success from metrics alone."""
    base = {"seed": seed, "path": str(path), "status": "pending", "issues": []}
    if not path.exists():
        base["issues"] = ["missing final result.json"]
        return base
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("result envelope is not an object")
    except (OSError, ValueError) as error:
        return {**base, "status": "invalid", "issues": [f"cannot read result: {error}"]}
    issues = []
    for key, expected in (("schema_version", 1), ("method", "tflow"),
                          ("condition_id", condition["id"]), ("seed", seed)):
        if raw.get(key) != expected or (key == "seed" and type(raw.get(key)) is not int):
            issues.append(f"{key} does not match expected {expected!r}")
    if raw.get("stage", "final") != "final":
        issues.append("tuning records cannot be included as final results")
    if raw.get("status") not in ("complete", "failed"):
        issues.append("status must be complete or failed")
    if not SHA256.fullmatch(str(raw.get("config_sha256", ""))):
        issues.append("missing or invalid config_sha256")
    if not SHA256.fullmatch(str(raw.get("implementation_sha256", ""))):
        issues.append("missing or invalid implementation_sha256")
    implementation = None
    if raw.get("status") == "complete":
        try:
            implementation = json.loads(path.with_name("implementation.json").read_text())
            if not isinstance(implementation, dict) or not implementation:
                raise ValueError("implementation manifest must be a nonempty object")
            implementation_hash = hashlib.sha256(
                json.dumps(implementation, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if implementation_hash != raw.get("implementation_sha256"):
                issues.append("implementation_sha256 does not match implementation.json")
        except (OSError, TypeError, ValueError) as error:
            issues.append(f"cannot verify implementation.json: {error}")
    cfg = raw.get("config")
    if not isinstance(cfg, dict) or not cfg:
        issues.append("missing resolved config")
    else:
        try:
            actual_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if actual_hash != raw.get("config_sha256"):
                issues.append("config_sha256 does not match the canonical config payload")
        except (TypeError, ValueError):
            issues.append("config cannot be represented as finite canonical JSON")
        if cfg.get("protocol_status") != "resolved" or cfg.get("blocking_issues"):
            issues.append("record config has unresolved protocol issues")
        if cfg.get("condition_id") != condition["id"] or cfg.get("kind") != condition["protocol"]:
            issues.append("record config condition/kind does not match the suite")
        if cfg.get("seeds") != condition["model_seeds"]:
            issues.append("record config final seeds do not match the suite")
        training = cfg.get("training", {})
        evaluation = cfg.get("evaluation", {})
        if not isinstance(training, dict):
            issues.append("training configuration is not an object")
            training = {}
        if not isinstance(evaluation, dict):
            issues.append("evaluation configuration is not an object")
            evaluation = {}
        for key, expected in (("steps", condition.get("train_steps")), ("batch_size", condition.get("batch_size"))):
            if expected is not None and training.get(key) != expected:
                issues.append(f"training.{key} differs from the suite budget")
        for key, expected in (("n_samples", condition.get("n_generated")), ("model_evaluations", condition.get("actual_nfe"))):
            if expected is not None and evaluation.get(key) != expected:
                issues.append(f"evaluation.{key} differs from the suite protocol")
        expected_input = condition.get("input_sha256")
        data_config = cfg.get("data", {})
        input_config = data_config.get("input", {}) if isinstance(data_config, dict) else {}
        input_hash = input_config.get("sha256") if isinstance(input_config, dict) else None
        if not SHA256.fullmatch(str(input_hash or "")):
            issues.append("record config does not pin its cached input SHA-256")
        elif expected_input and input_hash != expected_input:
            issues.append("record cached input hash differs from the suite")
    if raw.get("status") == "complete" and (not isinstance(raw.get("hardware"), dict) or not raw["hardware"]):
        issues.append("missing hardware provenance")
    source = raw.get("source", {})
    if not isinstance(source, dict):
        source = {}
    for key in ("nu", "scale"):
        value = source.get(key)
        lower = 2 if key == "nu" else 0
        if not _numeric(value) or not math.isfinite(value) or value <= lower:
            issues.append(f"missing or invalid source.{key}")
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        issues.append("metrics must be an object")
    bad = _bad_numbers(metrics)
    if bad:
        issues.append("nonfinite values: " + ", ".join(bad))
    try:
        normalized = normalize_metrics(metrics, condition["protocol"])
    except ValueError as error:
        normalized = {}
        issues.append(str(error))
    if raw.get("status") == "complete":
        if raw.get("nonfinite_metric_fields"):
            issues.append("undefined/nonfinite metric fields: final run must be recorded as failed")
        for key in REQUIRED_METRICS[condition["protocol"]]:
            if not _numeric(normalized.get(key)) or not math.isfinite(normalized[key]):
                issues.append(f"missing finite required metric {key}")
        # Historical exploding_norm_rate counts finite radii above 100 times
        # the generated median; invalid_rate also includes that tail flag.
        # These are reportable diagnostics, not reasons to discard heavy tails.
        for key in ("nan_rate", "inf_rate"):
            if _numeric(normalized.get(key)) and normalized[key] != 0:
                issues.append(f"nonzero {key}: final run must be recorded as failed")
        for key in ("total_train_time_s", "sample_time_s"):
            if _numeric(normalized.get(key)) and normalized[key] < 0:
                issues.append(f"negative measured timing {key}")
        if condition.get("actual_nfe") is not None and normalized.get("nfe") != condition["actual_nfe"]:
            issues.append("recorded nfe differs from the matched evaluation budget")
    status = "invalid" if issues else raw["status"]
    return {
        **base, "status": status, "issues": issues,
        "metrics": _safe_json(normalized), "native_metrics": _safe_json(metrics), "source": _safe_json(source),
        "config": _safe_json(raw.get("config")), "config_sha256": raw.get("config_sha256"),
        "implementation_sha256": raw.get("implementation_sha256"),
        "implementation": _safe_json(implementation),
        "checkpoint": _safe_json(raw.get("checkpoint")),
        "sample_artifact": _safe_json(raw.get("sample_artifact")),
        "sampler": _safe_json(raw.get("sampler")),
        "downstream_evaluation": _safe_json(raw.get("downstream_evaluation")),
        "hardware": _safe_json(raw.get("hardware")), "failure": _safe_json(raw.get("failure")),
        "nonfinite_metric_fields": raw.get("nonfinite_metric_fields", []),
        "recorded_status": raw.get("status"),
    }


def aggregate_condition(condition: dict, results_root: Path) -> dict:
    _validate_condition(condition)
    seeds = condition["model_seeds"]
    runs = [inspect_run(results_root / condition["id"] / f"seed_{s}" / "result.json", condition, s)
            for s in seeds]
    complete = [r for r in runs if r["status"] == "complete"]
    issues = []
    expected_paths = {results_root / condition["id"] / f"seed_{s}" / "result.json" for s in seeds}
    unexpected = sorted(str(p) for p in (results_root / condition["id"]).glob("seed_*/result.json")
                        if p not in expected_paths)
    if unexpected:
        issues.append("unexpected final-seed artifacts: " + ", ".join(unexpected))
    if len(complete) == len(seeds):
        settings = {(r["source"]["nu"], r["source"]["scale"]) for r in complete}
        if len(settings) != 1:
            issues.append("final seeds do not share the same frozen source nu and scale")
        if len({r["config_sha256"] for r in complete}) != 1:
            issues.append("final seeds do not share the same frozen resolved configuration")
        if len({r["implementation_sha256"] for r in complete}) != 1:
            issues.append("final seeds do not share the same implementation fingerprint")
    if condition.get("blocking_issue_ids") or not condition.get("resolved_protocol", False):
        issues.append("suite condition has unresolved input/protocol issues")
    status = ("failed" if any(r["status"] in ("failed", "invalid") for r in runs)
              else "pending" if len(complete) != len(seeds)
              else "invalid" if issues else "complete")
    aggregate = {}
    metric_coverage = {}
    metric_keys = sorted({k for r in complete for k, v in r["metrics"].items() if _numeric(v)})
    for key in metric_keys:
        vals = [r.get("metrics", {}).get(key) for r in runs]
        count = sum(_numeric(v) and math.isfinite(v) for v in vals)
        metric_coverage[key] = {"available": count, "expected": len(seeds)}
        if status == "complete" and count == len(seeds):
            aggregate[key] = {"mean": statistics.mean(vals), "std": statistics.pstdev(vals),
                              "vals": vals, "n": len(vals)}
    return {
        "schema_version": 1, "method": "tflow", "condition_id": condition["id"],
        "status": status, "expected_seeds": seeds, "n_expected": len(seeds),
        "n_complete": len(complete), "n_failed": sum(r["status"] in ("failed", "invalid") for r in runs),
        "n_pending": sum(r["status"] == "pending" for r in runs), "issues": issues,
        "protocol": condition["protocol"], "paper_scope": condition.get("paper_scope"),
        "runs": runs, "aggregate_full_precision": aggregate, "aggregate": aggregate,
        "metric_coverage": metric_coverage,
        "historical_values_recomputed": False,
    }


def collect_suite(manifest: dict, results_root: Path) -> dict:
    conditions = [c for c in manifest["benchmarks"] if c.get("required", True)]
    ids = [c["id"] for c in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate conditions in suite manifest")
    if not ids:
        raise ValueError("The suite has no required conditions")
    results = {c["id"]: aggregate_condition(c, results_root) for c in conditions}
    count = sum(r["status"] == "complete" for r in results.values())
    unexpected = sorted(str(p) for p in results_root.glob("*/seed_*/result.json")
                        if p.parent.parent.name not in ids)
    return {
        "schema_version": 1, "method": "tflow",
        "status": "complete" if count == len(conditions) and not unexpected else "incomplete",
        "n_conditions": len(conditions), "n_complete_conditions": count,
        "n_expected_runs": sum(r["n_expected"] for r in results.values()),
        "n_complete_runs": sum(r["n_complete"] for r in results.values()),
        "paper": manifest.get("paper", {}), "conditions": results,
        "unexpected_condition_artifacts": unexpected,
        "historical_values_recomputed": False,
    }


def _cell(row: dict | None, metric: str, *, latex: bool = False) -> str:
    if row is None:
        return "NOT IN MANIFEST"
    if row["status"] != "complete":
        return f"{row['status'].upper()} ({row['n_complete']}/{row['n_expected']})"
    stats = row["aggregate_full_precision"].get(metric)
    if stats is None:
        n = row["metric_coverage"].get(metric, {}).get("available", 0)
        return f"PENDING METRIC ({n}/{row['n_expected']})"
    pm = r"$\pm$" if latex else " ± "
    return f"{stats['mean']:.6g}{pm}{stats['std']:.6g}"


def _escape(value: str) -> str:
    return (value.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#"))


def table_addition(report: dict, ids: tuple[str, ...], metrics: tuple[str, ...], title: str) -> str:
    """Standalone t-Flow-only tabular: no legacy row is read, rounded or replaced."""
    lines = ["% Standalone addition only. Existing paper values are unchanged.",
             "% " + title, r"\begin{tabular}{l" + "r" * len(metrics) + "}",
             r"\hline", "Condition / t-Flow & " + " & ".join(_escape(m) for m in metrics) + r" \\",
             r"\hline"]
    for id in ids:
        row = report["conditions"].get(id)
        lines.append(_escape(id) + " & " + " & ".join(_cell(row, m, latex=True) for m in metrics) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_bundle(report: dict) -> dict[str, str]:
    """Pure formatting; callers decide when approved results may be written."""
    text = ["# New t-Flow results", "",
            f"Coverage: **{report['n_complete_conditions']}/{report['n_conditions']} conditions**, "
            f"**{report['n_complete_runs']}/{report['n_expected_runs']} final seeds**.", "",
            "Existing reported paper values are unchanged. These tables contain only new t-Flow additions. "
            "A failed or incomplete condition has no mean or standard deviation. Standard deviations use the population convention.", "",
            "| Condition | Status | Complete | Failed | Pending | Issues |",
            "|---|---|---:|---:|---:|---|"]
    for id, row in report["conditions"].items():
        run_issues = [f"seed {r['seed']}: {'; '.join(r['issues']) or str(r.get('failure'))}"
                      for r in row["runs"] if r["status"] != "complete"]
        issues = "; ".join(row["issues"] + run_issues).replace("|", r"\|").replace("\n", " ")
        text.append(f"| `{id}` | {row['status']} | {row['n_complete']} | {row['n_failed']} | {row['n_pending']} | {issues} |")
    if report.get("unexpected_condition_artifacts"):
        text.extend(["", "Unmapped final-result artifacts (suite remains incomplete):", ""])
        text.extend(f"- `{p}`" for p in report["unexpected_condition_artifacts"])
    text.extend(["", "## Full per-condition metrics", ""])
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(["condition_id", "status", "metric", "mean", "population_std", "n", "expected_n"])
    for id, row in report["conditions"].items():
        text.extend([f"### {id}", "", f"Status: {row['status']}; {row['n_complete']}/{row['n_expected']} complete final seeds.", ""])
        if row["status"] != "complete":
            writer.writerow([id, row["status"], "", "", "", row["n_complete"], row["n_expected"]])
            text.extend(["Aggregate withheld. Per-seed status and provenance are retained in the JSON report.", ""])
            continue
        text.extend(["| Metric | Mean ± population SD | Seeds |", "|---|---:|---:|"])
        for metric in sorted(row["metric_coverage"]):
            stats = row["aggregate_full_precision"].get(metric)
            text.append(f"| `{metric}` | {_cell(row, metric)} | {row['metric_coverage'][metric]['available']}/{row['n_expected']} |")
            writer.writerow([id, "complete" if stats else "pending_metric", metric,
                             stats["mean"] if stats else "", stats["std"] if stats else "",
                             stats["n"] if stats else row["metric_coverage"][metric]["available"], row["n_expected"]])
        text.append("")
    return {
        "tflow_results.json": json.dumps(_safe_json(report), indent=2, allow_nan=False) + "\n",
        "tflow_results.md": "\n".join(text).rstrip() + "\n",
        "tflow_metrics.csv": csv_buffer.getvalue(),
        "table1_tflow_addition.tex": table_addition(report, MAIN_VECTOR, ("sliced_w1",), "Table 1: t-Flow Sliced W1 column"),
        "table2_tflow_addition.tex": table_addition(report, (IMAGE_ID,), ("fid", "kid", "radial_w1", "precision", "coverage"), "Table 2: t-Flow image row"),
        "table3_tflow_addition.tex": table_addition(report, TABLE3_VECTOR, ("radial_w1", "ks_stat", "sliced_w1", "total_train_time_s"), "Table 3: t-Flow vector rows"),
        "table4_tflow_addition.tex": table_addition(report, (IMAGE_ID,), ("fid", "kid", "radial_w1", "ks", "precision", "recall", "density", "coverage"), "Table 4: t-Flow full image row"),
    }


def write_bundle(report: dict, output_dir: Path, *, allow_pending: bool = False) -> None:
    if report["status"] != "complete" and not allow_pending:
        raise ValueError("Incomplete suite: no result bundle written; use --pending only for an explicit status preview")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty; existing artifacts are never overwritten")
    bundle = render_bundle(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in bundle.items():
        (output_dir / name).write_text(content)
    conditions_dir = output_dir / "conditions"
    conditions_dir.mkdir()
    for id, row in report["conditions"].items():
        (conditions_dir / f"{id}.json").write_text(json.dumps(_safe_json(row), indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/tflow/suite_manifest.json"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pending", action="store_true", help="Explicit preview with pending/failed cells; never average surviving seeds")
    args = parser.parse_args(argv)
    try:
        report = collect_suite(json.loads(args.manifest.read_text()), args.results_root)
        write_bundle(report, args.output_dir, allow_pending=args.pending)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Report not written: {error}\n")
    print(f"{report['status']}: {report['n_complete_conditions']}/{report['n_conditions']} conditions; "
          f"{report['n_complete_runs']}/{report['n_expected_runs']} final seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
