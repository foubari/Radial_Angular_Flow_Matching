"""Render paper Table 5 / Figure 2 with measured AudioMNIST additions only.

An explicit --pending preview renders published measurements alone. This module
imports no ML library and never loads model checkpoints or generated tensors.
An audited discrepancy note can explicitly permit a complete measured gain
result whose original-checkpoint reevaluation differs from the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SEEDS = (8925, 1234, 7)
PAPER_SHA256 = "97f709df2a45e4acf4ba186379c60e9c75120d95e09af782d7a18c8b56d00367"
PAPER_SOURCE = "RAFM_ICLR_2027.pdf, page 26 Table 5; page 8 Figure 2"
METRICS = ("digit_acc", "energy_KS", "cov>q95", "cov>q99")
REFERENCE_LABELS = {"gaussian_euclidean": "Gaussian FM", "matched_euclidean": "Matched source",
                    "rafm": "RAFM-Vel", "angular_rafm": "RAFM-Ang"}
# These strings preserve the visually checked PDF exactly, including its precision.
# The final sparse MSGM row is in Table 5, but Figure 2 contains the five FM controls.
PUBLISHED = (
    ("gaussian", "Gaussian FM", ("0.734", "0.004"), ("0.117", "0.008"), ("0.0587", "0.0041"), ("0.0195", "0.0032")),
    ("matched", "Matched source", ("0.750", "0.018"), ("0.095", "0.010"), ("0.0397", "0.0013"), ("0.0118", "0.0002")),
    ("fixed_spher", "Fixed spherical", ("0.810", "0.013"), ("0.592", "0.000"), ("0.0000", "0.0000"), ("0.0000", "0.0000")),
    ("std_rafm", "RAFM-Vel", ("0.711", "0.014"), ("0.0218", "0.0000"), ("0.0500", "0.0000"), ("0.0145", "0.0000")),
    ("angular_rafm", "RAFM-Ang", ("0.764", "0.025"), ("0.0218", "0.0000"), ("0.0500", "0.0000"), ("0.0145", "0.0000")),
    ("msgm_sparse", "MSGM (sparse)", ("0.100", "0.000"), ("0.0340", "0.0000"), ("0.0465", "0.0000"), ("0.0105", "0.0000")),
)
STYLE = {
    "gaussian": ("Gaussian FM", "#888888", "s"),
    "matched": ("Matched-Euclidean", "#41955A", "s"),
    "fixed_spher": ("Fixed spherical", "#A16BB1", "^"),
    "std_rafm": ("RAFM-Vel", "#1976B5", "D"),
    "angular_rafm": ("RAFM-Ang", "#D4473B", "o"),
    "fixed_spher_empirical_gain": ("Fixed spherical + empirical gain", "#7541A0", "*"),
    "tflow": ("t-Flow", "#C48A23", "P"),
}


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    if len(values) != 3 or not all(finite_number(v) for v in values):
        raise ValueError("three finite measured values are required")
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "vals": values, "n": 3}


def published_points(reference):
    """Read per-seed points, and cross-check their mean/SD against the paper."""
    stage = reference.get("24000", {})
    points = {}
    for key, _label, *published in PUBLISHED[:-1]:
        row = stage.get(key, {})
        if row.get("n_seeds") != 3:
            raise ValueError(f"{key}: original three-seed 24000 row missing")
        points[key] = {}
        for metric, (printed_mean, printed_std) in zip(METRICS, published):
            cell = stats(row.get(metric, {}).get("vals", []))
            digits = len(printed_mean.split(".")[1])
            if f"{cell['mean']:.{digits}f}" != printed_mean or f"{cell['std']:.{digits}f}" != printed_std:
                raise ValueError(f"{key}/{metric}: archived points disagree with the PDF Table 5")
            points[key][metric] = cell
    return points


def measured_metrics(row, kind):
    if kind == "fixed_spher_empirical_gain":
        raw = row.get("posthoc", {}).get("unrounded", {})
        energy = raw.get("energy", {})
        return {"digit_acc": raw.get("digit_acc"), "energy_KS": energy.get("ks"),
                "cov>q95": energy.get("cov_gt_q95"), "cov>q99": energy.get("cov_gt_q99")}
    # Optional t-Flow rows use an explicit reporting envelope with canonical keys.
    return {key: row.get("metrics", {}).get(key) for key in METRICS}


def validate_invariance(run):
    checks = run.get("invariance", {})
    if checks.get("passed") is not True or checks.get("prediction_disagreements") != 0 or checks.get("accuracy_difference") != 0:
        raise ValueError("empirical-gain classifier invariance has not passed")
    if checks.get("logits_close") is not True or checks.get("logits_finite") is not True:
        raise ValueError("empirical-gain logit checks have not passed")
    for key in ("direction_l2_max", "radius_relative_error_max"):
        value = checks.get(key)
        if not finite_number(value) or not 0 <= value <= 2e-6:
            raise ValueError(f"empirical-gain {key} failed")
    before = run.get("baseline", {}).get("unrounded", {})
    after = run.get("posthoc", {}).get("unrounded", {})
    count = after.get("correct_count")
    if not isinstance(count, int) or not 0 <= count <= 2000 or before.get("correct_count") != count or after.get("digit_acc") != count / 2000:
        raise ValueError("empirical-gain exact before/after counts are inconsistent")
    if checks.get("correct_before") != count or checks.get("correct_after") != count:
        raise ValueError("empirical-gain reported invariance counts disagree with measured counts")


def validate_reference_controls(payload):
    groups = {}
    for row in payload.get("reference_controls", []):
        method = row.get("method")
        if method not in REFERENCE_LABELS:
            raise ValueError("unknown complete-sample reference control")
        groups.setdefault(method, []).append(row)
        if row.get("step") != 24000 or row.get("n") != 2000:
            raise ValueError("reference controls must use complete 2000-sample final-checkpoint evaluation")
        checksum = row.get("checkpoint", {}).get("sha256", "")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("reference control checkpoint SHA-256 missing")
        validate_invariance(row)
    for method, rows in groups.items():
        if [row.get("seed") for row in rows] != list(SEEDS):
            raise ValueError(f"{method}: complete-sample controls require all three original seeds")
    return groups


def validate_tflow_condition(payload):
    """Consume the suite renderer's actual condition report and recheck its runs.

    The suite report carries each original result.json path. Re-reading that
    JSON preserves checkpoint/sample provenance dropped from its compact rows;
    no model, tensor, or checkpoint deserialization is involved.
    """
    required = {"schema_version": 1, "method": "tflow", "condition_id": "audiomnist_stft",
                "protocol": "audio", "status": "complete", "expected_seeds": list(SEEDS),
                "n_expected": 3, "n_complete": 3, "n_failed": 0, "n_pending": 0}
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"tflow condition report: {key} must equal {value!r}")
    if payload.get("issues"):
        raise ValueError("tflow condition report has unresolved issues")
    runs = payload.get("runs", [])
    if [row.get("seed") for row in runs] != list(SEEDS):
        raise ValueError("tflow condition report requires all three original seeds in order")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.tflow.render_results import inspect_run
    config_hashes, implementation_hashes, sources, measured_rows = set(), set(), set(), []
    for row in runs:
        if row.get("status") != "complete" or row.get("recorded_status") != "complete" or row.get("issues"):
            raise ValueError("tflow condition report contains an unsuccessful final seed")
        cfg = row.get("config", {})
        evaluation, training = cfg.get("evaluation", {}), cfg.get("training", {})
        if training.get("steps") != 24000 or training.get("batch_size") != 32:
            raise ValueError("tflow audio training budget differs from the paper")
        if evaluation.get("sample_seed") != 0 or evaluation.get("n_samples") != 2000 or evaluation.get("model_evaluations") != 160:
            raise ValueError("tflow audio evaluation protocol differs from the paper")
        if cfg.get("model", {}).get("kind") != "audio_unet" or cfg.get("model", {}).get("ch") != 96:
            raise ValueError("tflow audio must use the matched UNet backbone")
        for pin in (cfg.get("data", {}).get("input", {}), cfg.get("data", {}).get("external_test", {}), evaluation.get("classifier", {})):
            digest = pin.get("sha256", "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("tflow audio input/test/classifier hashes must all be pinned")
        path = Path(row.get("path", ""))
        condition = {"id": "audiomnist_stft", "protocol": "audio", "model_seeds": list(SEEDS),
                     "train_steps": 24000, "batch_size": 32, "n_generated": 2000, "actual_nfe": 160,
                     "input_sha256": cfg["data"]["input"]["sha256"]}
        inspected = inspect_run(path, condition, row["seed"])
        if inspected["status"] != "complete" or inspected.get("config") != cfg or inspected.get("config_sha256") != row.get("config_sha256"):
            raise ValueError(f"tflow seed {row['seed']}: original final result/config provenance failed")
        if inspected.get("implementation_sha256") != row.get("implementation_sha256"):
            raise ValueError("tflow condition report differs from its original implementation fingerprint")
        if inspected.get("metrics") != row.get("metrics") or inspected.get("source") != row.get("source"):
            raise ValueError("tflow condition report differs from its original measured result")
        raw = json.loads(path.read_text())
        if raw.get("checkpoint", {}).get("step") != 24000:
            raise ValueError("tflow audio requires the final 24000-step checkpoint")
        for artifact in (raw.get("checkpoint", {}), raw.get("sample_artifact", {})):
            digest = artifact.get("sha256", "")
            if not artifact.get("path") or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("tflow original result requires checkpoint and generated-sample hashes")
        config_hashes.add(row["config_sha256"])
        implementation_hashes.add(row["implementation_sha256"])
        sources.add((row["source"]["nu"], row["source"]["scale"]))
        measured = measured_metrics(row, "tflow")
        if not all(finite_number(v) and 0 <= v <= 1 for v in measured.values()) or measured["energy_KS"] <= 0:
            raise ValueError("tflow audio has missing, nonfinite or invalid measured metrics")
        measured_rows.append(measured)
    if len(config_hashes) != 1 or len(implementation_hashes) != 1 or len(sources) != 1:
        raise ValueError("tflow final seeds must share one frozen configuration, implementation and source")
    return {metric: stats([row[metric] for row in measured_rows]) for metric in METRICS}


def discrepancy_note(path):
    """Snapshot the supplied audit note; its contents accompany the new figure."""
    path = Path(path).resolve()
    contents = path.read_bytes().decode("utf-8")
    if not contents.strip():
        raise ValueError("baseline discrepancy note must contain a nonempty audited explanation")
    return {"path": str(path), "sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
            "content": contents}


def validate_discrepancy_note(note):
    if not isinstance(note, dict) or not isinstance(note.get("content"), str) or not note["content"].strip():
        raise ValueError("a nonempty audited baseline discrepancy note is required")
    if not isinstance(note.get("path"), str) or not Path(note["path"]).is_absolute():
        raise ValueError("baseline discrepancy note must retain its absolute source path")
    if note.get("sha256") != hashlib.sha256(note["content"].encode("utf-8")).hexdigest():
        raise ValueError("baseline discrepancy note content does not match its SHA-256")


def validate_measured(payload, kind, baseline_discrepancy_note=None):
    """Reject pending, partial, mismatched, nonfinite, and failed-control results."""
    if kind == "tflow":
        return validate_tflow_condition(payload)
    allow_discrepancy = (kind == "fixed_spher_empirical_gain"
                         and payload.get("status") == "baseline_mismatch"
                         and baseline_discrepancy_note is not None)
    if baseline_discrepancy_note is not None:
        validate_discrepancy_note(baseline_discrepancy_note)
    if (payload.get("status") != "complete" and not allow_discrepancy) or payload.get("method") != kind:
        raise ValueError(f"{kind}: only a complete measured result is eligible")
    if allow_discrepancy and payload.get("archived_baselines_reproduced") is not False:
        raise ValueError("baseline_mismatch must explicitly record that archived baselines were not reproduced")
    protocol = payload.get("protocol", {})
    required = {"training_seeds": list(SEEDS), "checkpoint_step": 24000,
                "n_gen": 2000, "sample_seed": 0, "model_evaluations": 160}
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"{kind}: protocol {key} must equal {value!r}")
    runs = payload.get("runs", [])
    if [run.get("seed") for run in runs] != list(SEEDS):
        raise ValueError(f"{kind}: all three original seeds, in order, are required")
    for run in runs:
        if run.get("step") != 24000 or run.get("n") != 2000:
            raise ValueError(f"{kind}: incomplete per-seed sample/checkpoint provenance")
        checksum = run.get("checkpoint", {}).get("sha256", "")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"{kind}: missing checkpoint SHA-256")
        measured = measured_metrics(run, kind)
        if not all(finite_number(value) and 0 <= value <= 1 for value in measured.values()):
            raise ValueError(f"{kind}: missing/nonfinite/out-of-range measured metrics")
        if measured["energy_KS"] <= 0:
            raise ValueError(f"{kind}: energy KS must be positive for the logarithmic Figure 2 axis")
        if kind == "fixed_spher_empirical_gain":
            validate_invariance(run)
    if kind == "fixed_spher_empirical_gain" and payload.get("archived_baselines_reproduced") is not True and not allow_discrepancy:
        raise ValueError("original fixed-spherical baselines have not been reproduced")
    if kind == "fixed_spher_empirical_gain":
        validate_reference_controls(payload)
    return {metric: stats([measured_metrics(row, kind)[metric] for row in runs]) for metric in METRICS}


def cell(mean, std, digits=4):
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def table5_tex(additions, pending=False, baseline_discrepancy=False):
    lines = ["% Original rows: " + PAPER_SOURCE, "% PDF SHA256: " + PAPER_SHA256,
             "% Requires booktabs. New rows are computed from full-precision measured values.",
             r"\begin{table}[t]\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Digit accuracy $\uparrow$ & Energy KS $\downarrow$ & Mass above $Q_g(0.95)$ & Mass above $Q_g(0.99)$ \\",
             r"\midrule"]
    for _key, label, *values in PUBLISHED:
        lines.append(label + " & " + " & ".join(f"${mean} \\pm {std}$" for mean, std in values) + r" \\")
    if additions:
        lines.append(r"\midrule")
    for key, measured in additions.items():
        label = STYLE[key][0]
        if baseline_discrepancy and key == "fixed_spher_empirical_gain":
            label += r" $^{\dagger}$"
        lines.append(label + " & " + " & ".join(cell(measured[m]["mean"], measured[m]["std"], 3 if m == "digit_acc" else 4) for m in METRICS) + r" \\")
    if pending:
        lines.extend([r"\midrule", r"Fixed spherical + empirical gain & \multicolumn{4}{c}{Pending evaluation; no new measurements} \\"])
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    caption = ("AudioMNIST at 24,000 steps: mean and population standard deviation over three training seeds. "
               "Original Table 5 values are retained. RAFM-Ang versus RAFM-Vel remains the angular-target ablation. ")
    caption += ("The empirical-gain control is pending; this is a preview of published measurements."
                if pending else "The added fixed-spherical control draws an independent training-ECDF gain after generation, without retraining or changing the trained radius.")
    if baseline_discrepancy:
        caption += (r" $^{\dagger}$Original-checkpoint reevaluation differs from the archived baseline. "
                    "The added gain row reports actual new measurements with this discrepancy disclosed; "
                    "historical values remain unchanged and baseline reproduction is not established. "
                    "The audited discrepancy note is preserved in the reporting manifest.")
    lines.extend(["\\caption{" + caption + "}", r"\label{tab:audio-with-empirical-gain}", r"\end{table}", ""])
    return "\n".join(lines)


def table6_tex(payload=None):
    """Published Table 6 and new controls stay explicitly distinct."""
    published = "\n".join([
        "% Published Table 6, RAFM_ICLR_2027.pdf page 26; unchanged historical controls.",
        r"\begin{table}[t]\centering\small", r"\begin{tabular}{lll}", r"\toprule",
        r"Check & Evaluation set & Result \\", r"\midrule",
        r"Classifier validation & 1,200 held-out real clips & 95.5\% accuracy \\",
        r"Radius replacement & 1,500 generated clips; four reference checkpoints & Accuracy unchanged at reported precision \\",
        r"\bottomrule\end{tabular}",
        r"\caption{Published AudioMNIST evaluator checks. The radius replacement check covers Gaussian FM, Matched source, fixed spherical, and RAFM-Vel.}",
        r"\label{tab:audio-published-controls}\end{table}", "",
    ])
    if payload is None:
        return published, "% New complete-sample empirical-gain checks are pending; no results reported.\n"
    lines = ["% New measured 2,000-sample checks; distinct from historical Table 6.",
             r"\begin{table}[t]\centering\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & Seed & Clips & Correct before/after & Changed predictions & Max direction error & Max relative radius error \\", r"\midrule"]
    combined = [("Fixed spherical", row) for row in payload["runs"]]
    combined.extend((REFERENCE_LABELS[row["method"]], row) for row in payload.get("reference_controls", []))
    for label, row in combined:
        check = row["invariance"]
        lines.append(f"{label} & {row['seed']} & {row['n']} & {check['correct_before']}/{check['correct_after']} & "
                     f"{check['prediction_disagreements']} & {check['direction_l2_max']:.2e} & "
                     f"{check['radius_relative_error_max']:.2e}" + r" \\")
    lines.extend([r"\bottomrule\end{tabular}",
                  r"\caption{Complete-sample independent empirical-gain checks using direct normalized-STFT classification. Every generated clip is checked; all logits pass the stated numerical tolerance. Only methods with original artifacts and complete three-seed measurements are included; historical Table 6 results are not substituted.}",
                  r"\label{tab:audio-complete-gain-controls}\end{table}", ""])
    return published, "\n".join(lines)


def figure_disclosure(baseline_discrepancy):
    if not baseline_discrepancy:
        return ""
    return ("* Original-checkpoint reevaluation differs from archive; new gain points are measured.\n"
            "Historical points remain unchanged; baseline reproduction is not established.")


def figure_caption(pending=False, baseline_discrepancy=False):
    caption = ("AudioMNIST at 24,000 training steps. Hollow markers show individual training seeds; "
               "filled markers and error bars show the mean and population standard deviation. "
               "Historical Figure 2 points remain unchanged. RAFM-Ang and RAFM-Vel are separate comparisons. ")
    caption += ("The independent empirical-gain control is pending; no new point is shown."
                if pending else "The added fixed-spherical + empirical-gain points are measured from all three original checkpoint seeds.")
    if baseline_discrepancy:
        caption += (" Original-checkpoint reevaluation differs from the archived baseline; the new gain points "
                    "are actual measurements with this discrepancy disclosed. Baseline reproduction is not established. "
                    "The audited discrepancy note is preserved in reporting_manifest.json.")
    return caption


def render_scatter(points, output, pending, baseline_discrepancy=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for key, metrics in points.items():
        label, color, marker = STYLE[key]
        if baseline_discrepancy and key == "fixed_spher_empirical_gain":
            label += " (measured*)"
        ks, accuracy = metrics["energy_KS"], metrics["digit_acc"]
        ax.scatter(ks["vals"], accuracy["vals"], facecolors="none", edgecolors=color,
                   marker=marker, s=62 if marker != "*" else 115, linewidths=1.2, zorder=3)
        ax.errorbar(ks["mean"], accuracy["mean"], xerr=ks["std"], yerr=accuracy["std"],
                    fmt=marker, color=color, markersize=8 if marker != "*" else 12,
                    capsize=3, linewidth=1.2, label=label, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Energy KS (lower is better)")
    ax.set_ylabel("Digit accuracy (higher is better)")
    ax.set_title("Published AudioMNIST results — empirical-gain control pending" if pending else "AudioMNIST controls at 24,000 steps")
    ax.grid(True, alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.text(.5, .09 if baseline_discrepancy else .01,
             "Hollow: individual training seeds. Filled: mean ± population SD. Upper-left is better.", ha="center", fontsize=8)
    if baseline_discrepancy:
        fig.text(.5, .012, figure_disclosure(True), ha="center", fontsize=7.5, color="#813719")
    fig.tight_layout(rect=(0, .13 if baseline_discrepancy else .03, 1, 1))
    basename = "figure2_audio_pending" if pending else "figure2_audio_controls"
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{basename}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-aggregate", type=Path, required=True)
    ap.add_argument("--posthoc-result", type=Path)
    ap.add_argument("--tflow-result", type=Path)
    ap.add_argument("--baseline-discrepancy-note", type=Path,
                    help="audited nonempty note permitting a three-seed fixed-gain baseline_mismatch; archived values stay unchanged")
    ap.add_argument("--pending", action="store_true", help="explicit existing-only preview; no new points")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.pending and (args.posthoc_result or args.tflow_result or args.baseline_discrepancy_note):
        ap.error("--pending previews contain only published measurements")
    if not args.pending and not args.posthoc_result:
        ap.error("a validated --posthoc-result is required; use --pending for an existing-only preview")
    if args.output_dir.exists():
        ap.error("output directory must be new; existing assets are never overwritten")
    points = published_points(json.loads(args.reference_aggregate.read_text()))
    note = discrepancy_note(args.baseline_discrepancy_note) if args.baseline_discrepancy_note else None
    additions, payload = {}, None
    inputs = {"reference_aggregate": {"path": str(args.reference_aggregate.resolve()), "sha256": sha256(args.reference_aggregate)}}
    for kind, path in (("fixed_spher_empirical_gain", args.posthoc_result), ("tflow", args.tflow_result)):
        if path is None:
            continue
        data = json.loads(path.read_text())
        additions[kind] = validate_measured(data, kind, baseline_discrepancy_note=note if kind == "fixed_spher_empirical_gain" else None)
        if kind == "tflow" and data.get("condition_id") == "audiomnist_stft" and payload is not None:
            cfg = data["runs"][0]["config"]
            pins = {"train_file": cfg["data"]["input"], "test_file": cfg["data"]["external_test"],
                    "classifier": cfg["evaluation"]["classifier"]}
            if any(payload.get("inputs", {}).get(key, {}).get("sha256") != value["sha256"] for key, value in pins.items()):
                raise ValueError("tflow and empirical-gain evaluations do not use the same pinned audio data/classifier")
        inputs[kind] = {"path": str(path.resolve()), "sha256": sha256(path)}
        if kind == "fixed_spher_empirical_gain":
            payload = data
    points.update(additions)
    baseline_discrepancy = payload is not None and payload.get("status") == "baseline_mismatch"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "table5_audio_with_gain.tex").write_text(table5_tex(additions, pending=args.pending, baseline_discrepancy=baseline_discrepancy))
    published_controls, measured_controls = table6_tex(payload)
    (args.output_dir / "table6_published_controls.tex").write_text(published_controls)
    (args.output_dir / "table6_complete_gain_controls.tex").write_text(measured_controls)
    caption = figure_caption(args.pending, baseline_discrepancy)
    (args.output_dir / "figure2_caption.txt").write_text(caption + "\n")
    render_scatter(points, args.output_dir, args.pending, baseline_discrepancy=baseline_discrepancy)
    completed_controls = validate_reference_controls(payload) if payload else {}
    manifest = {"status": "pending_preview" if args.pending else "measured_addition_with_baseline_discrepancy" if baseline_discrepancy else "measured_addition",
                "paper_source": PAPER_SOURCE, "paper_sha256": PAPER_SHA256,
                "inputs": inputs, "plotted_methods": list(points),
                "new_measurements": additions, "existing_table_values_preserved": True,
                "baseline_discrepancy": baseline_discrepancy,
                "baseline_discrepancy_note": note,
                "recorded_posthoc_status": payload.get("status") if payload else None,
                "archived_baselines_reproduced": payload.get("archived_baselines_reproduced") if payload else None,
                "figure_disclosure": figure_disclosure(baseline_discrepancy),
                "figure_caption": caption,
                "complete_reference_controls": list(completed_controls),
                "missing_controls": [REFERENCE_LABELS[key] for key in REFERENCE_LABELS if key not in completed_controls]}
    (args.output_dir / "reporting_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {manifest['status']} assets to {args.output_dir}")


if __name__ == "__main__":
    main()
