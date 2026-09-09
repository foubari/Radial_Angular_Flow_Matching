"""Summarize measured AudioMNIST gain results, including baseline discrepancies.

Only JSON/text files are read. No ML import, tensor load, experiment or figure
generation occurs. Historical rows are copied unchanged from the paper renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.poc_audio.render_gain_results import PUBLISHED, PAPER_SHA256, PAPER_SOURCE

SEEDS = (8925, 1234, 7)
METHOD = "fixed_spher_empirical_gain"
METRICS = ("digit_acc", "energy_KS", "radial_w1", "cov>q90", "cov>q95", "cov>q99", "cov<q10", "PIT")
ENERGY_KEYS = {"energy_KS": "ks", "radial_w1": "radial_w1", "cov>q90": "cov_gt_q90",
               "cov>q95": "cov_gt_q95", "cov>q99": "cov_gt_q99", "cov<q10": "cov_lt_q10", "PIT": "pit_mean"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON constant: {value}")))


def fingerprint(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def full_metrics(row, phase):
    raw = row.get(phase, {}).get("unrounded", {})
    energy = raw.get("energy", {})
    result = {"digit_acc": raw.get("digit_acc"), **{key: energy.get(native) for key, native in ENERGY_KEYS.items()}}
    for key in ("gen_energy_mean", "data_energy_mean"):
        if key in energy:
            result[key] = energy[key]
    if "mean_confidence" in raw:
        result["mean_confidence"] = raw["mean_confidence"]
    return result


def aggregate(rows, phase):
    metrics = sorted(set.intersection(*(set(row[phase]) for row in rows)))
    return {key: {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "vals": vals, "n": 3}
            for key in metrics for vals in [[row[phase][key] for row in rows]] if all(finite(v) for v in vals)}


def summarize(input_path, reference_path):
    """Keep measurement completeness separate from historical reproduction."""
    result_dir = input_path.parent
    archived = read_json(reference_path)
    stage = archived.get("24000", {})
    for method in ("fixed_spher", "std_rafm", "angular_rafm"):
        if stage.get(method, {}).get("n_seeds") != 3:
            raise ValueError(f"Missing historical three-seed row {method}")
    artifact = read_json(input_path) if input_path.exists() else {}
    failure_path = result_dir / "failure.json"
    failure = read_json(failure_path) if failure_path.exists() else None
    if not artifact and not failure and not any(result_dir.glob("seed_*/eval.json")):
        raise FileNotFoundError(f"No measured aggregate, seed results or failure yet in {result_dir}")
    issues = []
    if artifact and (artifact.get("schema_version") != 1 or artifact.get("method") != METHOD):
        issues.append("Aggregate schema/method does not match the fixed-spherical gain experiment")
    if artifact and artifact.get("status") not in ("complete", "baseline_mismatch", "pending", "in_progress"):
        issues.append(f"Aggregate records failure or unsupported status: {artifact.get('status')!r}")
    by_seed = {}
    for row in artifact.get("runs", []):
        seed = row.get("seed")
        if type(seed) is not int or seed not in SEEDS or seed in by_seed:
            issues.append(f"Unexpected or duplicate aggregate seed: {seed!r}")
        else:
            by_seed[seed] = row
    for path in sorted(result_dir.glob("seed_*/eval.json")):
        row = read_json(path)
        seed = row.get("seed")
        if type(seed) is not int or seed not in SEEDS or path.parent.name != f"seed_{seed}":
            issues.append(f"Unexpected seed artifact: {path}")
        elif seed in by_seed and row != by_seed[seed]:
            issues.append(f"Aggregate/per-seed artifact disagreement for seed {seed}")
        else:
            by_seed[seed] = row
    runs = []
    mismatch = artifact.get("status") == "baseline_mismatch" or artifact.get("archived_baselines_reproduced") is False
    reproduction_checks = []
    for index, seed in enumerate(SEEDS):
        raw = by_seed.get(seed)
        if raw is None:
            runs.append({"seed": seed, "status": "pending", "issues": ["Missing measured seed result"]})
            continue
        run_issues = []
        if raw.get("step") != 24000 or raw.get("n") != 2000:
            run_issues.append("Run does not match step 24000 / 2000 generated samples")
        checks = raw.get("invariance", {})
        changed = checks.get("prediction_disagreements")
        if checks.get("passed") is not True or type(changed) is not int or changed != 0:
            run_issues.append("Invariance or zero-changed-prediction check failed or missing")
        phases = {phase: full_metrics(raw, phase) for phase in ("baseline", "posthoc")}
        for phase, values in phases.items():
            for metric, value in values.items():
                if not finite(value):
                    run_issues.append(f"Missing/nonfinite full-precision {phase}.{metric}")
        comparisons = raw.get("archived_baseline_comparison", {})
        for metric in ("digit_acc", "energy_KS", "cov>q95", "cov>q99", "cov<q10", "PIT"):
            cell = comparisons.get(metric, {})
            expected = stage["fixed_spher"].get(metric, {}).get("vals", [])
            rounded = (raw.get("baseline", {}).get("content", {}).get("digit_acc") if metric == "digit_acc"
                       else raw.get("baseline", {}).get("energy", {}).get(ENERGY_KEYS[metric]))
            matches = (len(expected) == 3 and finite(rounded)
                       and math.isclose(rounded, expected[index], rel_tol=1e-9, abs_tol=1e-12))
            reproduction_checks.append({"seed": seed, "metric": metric, "archived": expected[index] if len(expected) == 3 else None,
                                        "recomputed_rounded": rounded, "matches": matches, "recorded_comparison": cell})
            mismatch |= not matches or cell.get("matches") is False
        runs.append({"seed": seed, "status": "failed" if run_issues else "complete", "issues": run_issues,
                     **phases, "invariance": checks, "runtime": raw.get("runtime"),
                     "checkpoint": raw.get("checkpoint"), "metadata": raw.get("metadata"),
                     "samples": raw.get("samples"), "paired_gains": raw.get("paired_gains"),
                     "raw_result": raw})
    complete = not issues and not failure and all(row["status"] == "complete" for row in runs)
    aggregate_allowed = complete and artifact.get("status") in ("complete", "baseline_mismatch")
    status = ("failed" if issues or failure or any(row["status"] == "failed" for row in runs)
              else "pending" if not aggregate_allowed else "baseline_mismatch" if mismatch else "complete")
    stats = {phase: aggregate(runs, phase) if aggregate_allowed else {} for phase in ("baseline", "posthoc")}
    if artifact.get("aggregate_full_precision") and aggregate_allowed:
        for metric in METRICS:
            claimed = artifact["aggregate_full_precision"].get(metric, {})
            actual = stats["posthoc"][metric]
            if claimed.get("vals") != actual["vals"] or any(not finite(claimed.get(k)) or not math.isclose(claimed[k], actual[k], rel_tol=1e-12, abs_tol=1e-15) for k in ("mean", "std")):
                issues.append(f"Stored posthoc aggregate disagrees with full-precision seed values: {metric}")
        if issues:
            status, stats = "failed", {"baseline": {}, "posthoc": {}}
    return {"schema_version": 1, "method": METHOD, "status": status, "recorded_status": artifact.get("status", "aggregate_missing"),
            "expected_seeds": list(SEEDS), "n_complete": sum(row["status"] == "complete" for row in runs),
            "issues": issues, "failure": failure, "runs": runs, "aggregate_full_precision": stats,
            "archived_baselines_reproduced": status == "complete", "baseline_mismatch": mismatch,
            "reproduction_checks": reproduction_checks,
            "prediction_disagreements": [{"seed": row["seed"], "count": row.get("invariance", {}).get("prediction_disagreements")} for row in runs],
            "all_prediction_disagreements_zero": complete and all(row["invariance"]["prediction_disagreements"] == 0 for row in runs),
            "historical_rows_unchanged": {key: stage[key] for key in ("fixed_spher", "std_rafm", "angular_rafm")},
            "paper_rows_unchanged": PUBLISHED,
            "hardware": artifact.get("hardware"), "runtime": artifact.get("runtime"), "protocol": artifact.get("protocol"),
            "split": artifact.get("split"), "inputs": artifact.get("inputs"), "source_sha256": artifact.get("source_sha256"),
            "provenance": {"aggregate": fingerprint(input_path) if input_path.exists() else {"path": str(input_path.resolve()), "status": "missing"},
                           "reference_aggregate": fingerprint(reference_path), "summary_script": fingerprint(Path(__file__)),
                           "failure": fingerprint(failure_path) if failure_path.exists() else None,
                           "paper_sha256": PAPER_SHA256, "paper_source": PAPER_SOURCE},
            "figure2": {"eligible_without_discrepancy_note": status == "complete",
                        "eligible_with_audited_discrepancy_note": status == "baseline_mismatch",
                        "discrepancy_note_option": "--baseline-discrepancy-note PATH",
                        "renderer": "experiments/poc_audio/render_gain_results.py", "generated_by_this_script": False}}


def cell(stats, metric, latex=False):
    item = stats.get(metric)
    if item is None:
        return "PENDING / FAILED"
    sep = r" $\pm$ " if latex else " ± "
    return f"{item['mean']:.6g}{sep}{item['std']:.6g}"


def render(summary):
    status = summary["status"]
    lines = [f"# AudioMNIST empirical-gain comparison: {status}", "",
             f"Complete measured seeds: {summary['n_complete']}/3. Source status: `{summary['recorded_status']}`.", "",
             "Historical rows are preserved. New means use full-precision per-seed values and population standard deviations.", ""]
    if summary["baseline_mismatch"]:
        lines += ["**Baseline mismatch:** the newly evaluated fixed-spherical baseline does not reproduce the archived rounded metrics. These measured rows are discrepancy-labelled and do not establish exact historical reproduction. The current paired before/after measurements remain directly comparable.", ""]
    if status in ("pending", "failed"):
        lines += ["Missing or failed seeds are explicit. No mean over surviving seeds is reported.", ""]
    lines += ["| Metric | Current fixed spherical | Current fixed spherical + gain |", "|---|---:|---:|"]
    metrics = sorted(set(METRICS) | set(summary["aggregate_full_precision"]["baseline"]) | set(summary["aggregate_full_precision"]["posthoc"]))
    for metric in metrics:
        lines.append(f"| {metric} | {cell(summary['aggregate_full_precision']['baseline'], metric)} | {cell(summary['aggregate_full_precision']['posthoc'], metric)} |")
    lines += ["", "| Seed | Status | Original accuracy | Gain accuracy | Changed predictions | Generation seconds | Paired evaluation seconds |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        runtime = row.get("runtime") or {}
        lines.append(f"| {row['seed']} | {row['status']} | {row.get('baseline', {}).get('digit_acc', 'missing')} | {row.get('posthoc', {}).get('digit_acc', 'missing')} | {row.get('invariance', {}).get('prediction_disagreements', 'missing')} | {runtime.get('generation_s', 'missing')} | {runtime.get('paired_evaluation_s', 'missing')} |")
    lines += ["", "| Seed | Metric | Original full precision | Gain full precision |", "|---|---|---:|---:|"]
    for row in summary["runs"]:
        for metric in metrics:
            lines.append(f"| {row['seed']} | {metric} | {row.get('baseline', {}).get(metric, 'missing')} | {row.get('posthoc', {}).get(metric, 'missing')} |")
    lines += ["", f"All changed-prediction counts verified zero: `{summary['all_prediction_disagreements_zero']}`.", "",
              "Published fixed-spherical accuracy remains **0.810 ± 0.013**. RAFM-Vel and RAFM-Ang remain separate historical comparisons.", "",
              "| Historical row | Archived accuracy mean | Archived population SD |", "|---|---:|---:|"]
    for key, label in (("fixed_spher", "Fixed spherical"), ("std_rafm", "RAFM-Vel"), ("angular_rafm", "RAFM-Ang")):
        row = summary["historical_rows_unchanged"][key]["digit_acc"]
        lines.append(f"| {label} | {row['mean']} | {row['std']} |")
    for title, payload in (("Baseline reproduction checks", summary["reproduction_checks"]), ("Issues and failure", {"issues": summary["issues"], "failure": summary["failure"]}),
                           ("Hardware", summary["hardware"]), ("Runtime", summary["runtime"]), ("Protocol", summary["protocol"]), ("Provenance", summary["provenance"])):
        lines += ["", f"## {title}", "", "```json", json.dumps(payload, indent=2, allow_nan=False), "```"]
    lines += ["", "Per-seed full-precision metrics, checkpoint/sample fingerprints, implementation sources and paired comparisons are retained in `comparison.json`. No figure is generated by this summary script.", ""]
    tex = [f"% Measured comparison status: {status}; original paper rows unchanged.", r"\begin{tabular}{lrrrr}", r"\hline", r"Method & Accuracy & Energy KS & $P(r>q_{.95})$ & $P(r>q_{.99})$ \\", r"\hline"]
    for _, label, *values in PUBLISHED:
        tex.append(label + " & " + " & ".join(mean + r" $\pm$ " + std for mean, std in values) + r" \\")
    tex.append(r"\hline")
    qualifier = "baseline discrepancy" if summary["baseline_mismatch"] else status
    for phase, label in (("baseline", "Current fixed spherical"), ("posthoc", "Current fixed spherical + gain")):
        tex.append(f"{label} ({qualifier}) & " + " & ".join(cell(summary["aggregate_full_precision"][phase], metric, latex=True) for metric in ("digit_acc", "energy_KS", "cov>q95", "cov>q99")) + r" \\")
    tex += [r"\hline", r"\end{tabular}", f"% {summary['n_complete']}/3 measured seeds; new rows use full-precision values and population SD."]
    if summary["baseline_mismatch"]:
        tex.append("% Original baseline reproduction failed; new rows describe current paired measurements, not exact historical reproduction.")
    return {"comparison.json": json.dumps(summary, indent=2, allow_nan=False) + "\n", "comparison.md": "\n".join(lines), "table5_audio_gain_comparison.tex": "\n".join(tex) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs_audio_gain/fixed_spherical_empirical_gain_v1/aggregate.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "experiments/poc_audio/stage2_3seed.json")
    parser.add_argument("--output-dir", type=Path, help="New/empty report directory; defaults to INPUT_DIR/comparison")
    args = parser.parse_args()
    summary = summarize(args.input, args.reference)
    bundle = render(summary)
    output = args.output_dir or args.input.parent / "comparison"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in bundle.items():
        (output / name).write_text(contents)
    print(json.dumps({"status": summary["status"], "n_complete": summary["n_complete"], "output_dir": str(output.resolve()),
                      "baseline_reproduced": summary["archived_baselines_reproduced"], "changed_predictions": summary["prediction_disagreements"]}, indent=2))


if __name__ == "__main__":
    main()
