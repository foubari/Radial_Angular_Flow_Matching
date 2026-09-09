"""Reporting checks use small JSON fixtures, never generators or benchmarks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("render_gain_results", ROOT / "experiments/poc_audio/render_gain_results.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def measured_fixture():
    """Synthetic fixture for validation tests only; never an experiment result."""
    rows = []
    for seed in report.SEEDS:
        rows.append({
            "seed": seed, "step": 24000, "n": 2000,
            "checkpoint": {"sha256": "a" * 64},
            "baseline": {"unrounded": {"correct_count": 1600, "digit_acc": .8}},
            "posthoc": {"unrounded": {"correct_count": 1600, "digit_acc": .8,
                "energy": {"ks": .025, "cov_gt_q95": .05, "cov_gt_q99": .01}}},
            "invariance": {"passed": True, "prediction_disagreements": 0,
                "accuracy_difference": 0, "logits_close": True, "logits_finite": True,
                "direction_l2_max": 1e-7, "radius_relative_error_max": 1e-7,
                "correct_before": 1600, "correct_after": 1600},
        })
    return {"status": "complete", "method": "fixed_spher_empirical_gain",
            "archived_baselines_reproduced": True,
            "protocol": {"training_seeds": list(report.SEEDS), "checkpoint_step": 24000,
                         "n_gen": 2000, "sample_seed": 0, "model_evaluations": 160},
            "runs": rows}


def test_published_seed_points_match_visually_verified_paper_table():
    source = ROOT / "experiments/poc_audio/stage2_3seed.json"
    points = report.published_points(json.loads(source.read_text()))
    assert set(points) == {"gaussian", "matched", "fixed_spher", "std_rafm", "angular_rafm"}
    assert points["fixed_spher"]["digit_acc"]["vals"] == [.805, .798, .828]
    assert points["std_rafm"]["energy_KS"]["vals"] == [.0218] * 3


def test_modified_published_values_are_rejected():
    source = ROOT / "experiments/poc_audio/stage2_3seed.json"
    value = json.loads(source.read_text())
    value["24000"]["fixed_spher"]["digit_acc"]["vals"][0] = .99
    with pytest.raises(ValueError, match="disagree with the PDF"):
        report.published_points(value)


def test_pending_table_preserves_all_six_published_rows_and_adds_no_measurement():
    text = report.table5_tex({}, pending=True)
    for _, label, *metrics in report.PUBLISHED:
        assert label in text
        for mean, std in metrics:
            assert f"${mean} \\pm {std}$" in text
    assert "Pending evaluation; no new measurements" in text
    assert "RAFM-Ang versus RAFM-Vel" in text


@pytest.mark.parametrize("status", ["pending", "baseline_mismatch", "failed"])
def test_noncomplete_measurements_cannot_be_plotted(status):
    fixture = measured_fixture()
    fixture["status"] = status
    with pytest.raises(ValueError, match="only a complete"):
        report.validate_measured(fixture, "fixed_spher_empirical_gain")


def audited_note(tmp_path):
    path = tmp_path / "discrepancy-note.md"
    path.write_bytes(b"Unit-test audit fixture: checkpoint reevaluation differs from archive.\r\n")
    return report.discrepancy_note(path)


def test_baseline_discrepancy_opt_in_preserves_actual_metrics_and_historical_rows(tmp_path):
    fixture = measured_fixture()
    fixture.update(status="baseline_mismatch", archived_baselines_reproduced=False)
    note = audited_note(tmp_path)
    measured = report.validate_measured(fixture, "fixed_spher_empirical_gain", baseline_discrepancy_note=note)
    assert measured["digit_acc"]["vals"] == [.8] * 3
    text = report.table5_tex({"fixed_spher_empirical_gain": measured}, baseline_discrepancy=True)
    for _, label, *metrics in report.PUBLISHED:
        historical_row = label + " & " + " & ".join(f"${mean} \\pm {std}$" for mean, std in metrics)
        historical_row += " " + chr(92) * 2
        assert historical_row in text
    assert "Original-checkpoint reevaluation differs" in text
    assert "actual new measurements" in text
    assert "baseline reproduction is not established" in text
    assert "new gain points are measured" in report.figure_disclosure(True)
    assert report.figure_disclosure(False) == ""


@pytest.mark.parametrize("corruption", ["prediction", "missing_seed", "nonfinite", "protocol", "failed_status", "claimed_reproduction"])
def test_discrepancy_note_does_not_bypass_scientific_checks(tmp_path, corruption):
    fixture = measured_fixture()
    fixture.update(status="baseline_mismatch", archived_baselines_reproduced=False)
    if corruption == "prediction":
        fixture["runs"][0]["invariance"]["prediction_disagreements"] = 1
    elif corruption == "missing_seed":
        fixture["runs"].pop()
    elif corruption == "nonfinite":
        fixture["runs"][0]["posthoc"]["unrounded"]["energy"]["ks"] = float("nan")
    elif corruption == "protocol":
        fixture["protocol"]["model_evaluations"] = 40
    elif corruption == "failed_status":
        fixture["status"] = "failed"
    else:
        fixture["archived_baselines_reproduced"] = True
    with pytest.raises(ValueError):
        report.validate_measured(fixture, "fixed_spher_empirical_gain", baseline_discrepancy_note=audited_note(tmp_path))


def test_discrepancy_note_is_nonempty_and_hashes_exact_file_bytes(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text(" \n\t")
    with pytest.raises(ValueError, match="nonempty"):
        report.discrepancy_note(empty)
    note = audited_note(tmp_path)
    assert note["sha256"] == hashlib.sha256(Path(note["path"]).read_bytes()).hexdigest()
    note["content"] += "modified"
    fixture = measured_fixture()
    fixture.update(status="baseline_mismatch", archived_baselines_reproduced=False)
    with pytest.raises(ValueError, match="SHA-256"):
        report.validate_measured(fixture, "fixed_spher_empirical_gain", baseline_discrepancy_note=note)


def test_cli_opt_in_saves_audit_note_and_requests_visible_figure_disclosure(tmp_path, monkeypatch):
    fixture = measured_fixture()
    fixture.update(status="baseline_mismatch", archived_baselines_reproduced=False)
    result = tmp_path / "fixture-result.json"
    result.write_text(json.dumps(fixture))
    note = audited_note(tmp_path)
    output = tmp_path / "report"
    calls = []
    monkeypatch.setattr(report, "render_scatter", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["render_gain_results", "--reference-aggregate", str(ROOT / "experiments/poc_audio/stage2_3seed.json"),
                                     "--posthoc-result", str(result), "--baseline-discrepancy-note", note["path"], "--output-dir", str(output)])
    report.main()
    manifest = json.loads((output / "reporting_manifest.json").read_text())
    assert manifest["status"] == "measured_addition_with_baseline_discrepancy"
    assert manifest["baseline_discrepancy_note"] == note
    assert manifest["recorded_posthoc_status"] == "baseline_mismatch"
    assert manifest["archived_baselines_reproduced"] is False
    assert calls[0][1]["baseline_discrepancy"] is True
    assert "baseline reproduction is not established" in manifest["figure_disclosure"]
    assert (output / "figure2_caption.txt").read_text().strip() == manifest["figure_caption"]
    assert "Original-checkpoint reevaluation differs" in manifest["figure_caption"]


def test_two_seeds_and_changed_predictions_are_rejected():
    fixture = measured_fixture()
    incomplete = copy.deepcopy(fixture)
    incomplete["runs"].pop()
    with pytest.raises(ValueError, match="all three"):
        report.validate_measured(incomplete, "fixed_spher_empirical_gain")
    fixture["runs"][0]["invariance"]["prediction_disagreements"] = 2
    # The unchanged total accuracy does not excuse changed individual predictions.
    with pytest.raises(ValueError, match="invariance"):
        report.validate_measured(fixture, "fixed_spher_empirical_gain")


def test_aggregate_is_derived_from_measured_runs_not_supplied_means():
    fixture = measured_fixture()
    fixture["aggregate"] = {"digit_acc": {"mean": .999, "std": 0}}
    result = report.validate_measured(fixture, "fixed_spher_empirical_gain")
    assert result["digit_acc"]["mean"] == .8
    assert result["digit_acc"]["vals"] == [.8] * 3
    fixture["runs"][0]["posthoc"]["unrounded"]["energy"]["ks"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        report.validate_measured(fixture, "fixed_spher_empirical_gain")


def test_old_1500_controls_and_new_complete_checks_are_distinct():
    fixture = measured_fixture()
    old, new = report.table6_tex(fixture)
    assert "1,500 generated clips" in old
    assert "1,200 held-out real clips" in old
    assert "95.5\\% accuracy" in old
    assert "1,500" not in new
    assert "2,000-sample" in new
    for seed in report.SEEDS:
        assert f"{seed} & 2000" in new


def test_tflow_requires_actual_suite_provenance_not_a_custom_score_envelope():
    fixture = measured_fixture()
    fixture["method"] = "tflow"
    for row in fixture["runs"]:
        row["metrics"] = {"digit_acc": .7, "energy_KS": .03, "cov>q95": .05, "cov>q99": .01}
    with pytest.raises(ValueError, match="condition report"):
        report.validate_measured(fixture, "tflow")


def test_optional_reference_controls_are_complete_and_rendered_separately():
    fixture = measured_fixture()
    controls = copy.deepcopy(fixture["runs"])
    for row in controls:
        row["method"] = "rafm"
    fixture["reference_controls"] = controls
    assert list(report.validate_reference_controls(fixture)) == ["rafm"]
    _, text = report.table6_tex(fixture)
    assert "RAFM-Vel & 8925 & 2000" in text
    fixture["reference_controls"] = controls[:2]
    with pytest.raises(ValueError, match="all three"):
        report.validate_measured(fixture, "fixed_spher_empirical_gain")


def actual_suite_audio_fixture(tmp_path):
    """Pass synthetic metadata through the real suite collector; no experiments."""
    sys.path.insert(0, str(ROOT))
    from experiments.tflow.render_results import collect_suite
    condition = {"id": "audiomnist_stft", "protocol": "audio", "model_seeds": list(report.SEEDS),
                 "resolved_protocol": True, "blocking_issue_ids": [], "train_steps": 24000,
                 "batch_size": 32, "n_generated": 2000, "actual_nfe": 160, "input_sha256": "a" * 64}
    cfg = {"condition_id": "audiomnist_stft", "kind": "audio", "protocol_status": "resolved",
           "blocking_issues": [], "seeds": list(report.SEEDS),
           "training": {"steps": 24000, "batch_size": 32}, "model": {"kind": "audio_unet", "ch": 96},
           "evaluation": {"n_samples": 2000, "model_evaluations": 160, "sample_seed": 0,
                          "classifier": {"path": "fixture_classifier.pt", "sha256": "b" * 64}},
           "data": {"input": {"path": "fixture_train.pt", "sha256": "a" * 64},
                    "external_test": {"path": "fixture_test.pt", "sha256": "e" * 64}}}
    for seed, accuracy in zip(report.SEEDS, (.7, .72, .74)):
        directory = tmp_path / "audiomnist_stft" / f"seed_{seed}"
        directory.mkdir(parents=True)
        implementation = {"files": {"unit_fixture.py": "f" * 64}, "python": "fixture",
                          "torch": "fixture; no model executed", "numpy": "fixture"}
        (directory / "implementation.json").write_text(json.dumps(implementation))
        raw = {"schema_version": 1, "method": "tflow", "condition_id": "audiomnist_stft", "seed": seed,
               "status": "complete", "stage": "final", "config": cfg,
               "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest(),
               "implementation_sha256": hashlib.sha256(json.dumps(implementation, sort_keys=True, allow_nan=False).encode()).hexdigest(),
               "source": {"nu": 5., "scale": .5}, "hardware": {"description": "unit fixture; no GPU used"},
               "checkpoint": {"path": "fixture_checkpoint.pt", "step": 24000, "sha256": "c" * 64},
               "sample_artifact": {"path": "fixture_samples.pt", "sha256": "d" * 64},
               "metrics": {"digit_acc": accuracy, "energy_KS": .03, "radial_w1": .01,
                           "cov>q90": .1, "cov>q95": .05, "cov>q99": .01, "cov<q10": .1, "PIT": .5,
                           "nfe": 160, "invalid_rate": 0., "sample_time_s": 1., "total_train_time_s": 20.}}
        (directory / "result.json").write_text(json.dumps(raw))
    return collect_suite({"benchmarks": [condition]}, tmp_path)["conditions"]["audiomnist_stft"]


def test_actual_suite_condition_report_adds_measured_tflow_without_schema_rewrite(tmp_path):
    fixture = actual_suite_audio_fixture(tmp_path)
    assert fixture["status"] == "complete"
    measured = report.validate_measured(fixture, "tflow")
    assert measured["digit_acc"]["vals"] == [.7, .72, .74]
    assert measured["digit_acc"]["mean"] == pytest.approx(.72)
    text = report.table5_tex({"tflow": measured})
    assert "t-Flow" in text and "$0.720" in text
    fixture["n_complete"] = 2
    with pytest.raises(ValueError, match="n_complete"):
        report.validate_measured(fixture, "tflow")


def test_discrepancy_opt_in_does_not_relax_tflow_status_checks(tmp_path):
    fixture = actual_suite_audio_fixture(tmp_path)
    fixture["status"] = "baseline_mismatch"
    with pytest.raises(ValueError, match="status"):
        report.validate_measured(fixture, "tflow", baseline_discrepancy_note=audited_note(tmp_path))


def test_suite_condition_requires_original_hashes_and_unchanged_raw_measurements(tmp_path):
    fixture = actual_suite_audio_fixture(tmp_path)
    path = Path(fixture["runs"][0]["path"])
    raw = json.loads(path.read_text())
    raw["checkpoint"]["sha256"] = "missing"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checkpoint and generated-sample hashes"):
        report.validate_measured(fixture, "tflow")
    raw["checkpoint"]["sha256"] = "c" * 64
    raw["metrics"]["digit_acc"] = .99
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="differs from its original"):
        report.validate_measured(fixture, "tflow")
