"""Pure JSON/report correctness tests; no models, datasets or experiments."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.tflow.render_results import (
    REQUIRED_METRICS, aggregate_condition, collect_suite, normalize_metrics, render_bundle, write_bundle,
)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.condition = {
            "id": "piv_d64", "protocol": "vector", "required": True,
            "paper_scope": "main_and_appendix", "model_seeds": [8925, 77395, 65457],
            "resolved_protocol": True, "blocking_issue_ids": [], "train_steps": 10000,
            "batch_size": 256, "n_generated": 10000, "actual_nfe": 512,
            "input_sha256": "a" * 64,
        }
        self.config = {
            "condition_id": "piv_d64", "kind": "vector", "protocol_status": "resolved",
            "blocking_issues": [], "seeds": self.condition["model_seeds"],
            "training": {"steps": 10000, "batch_size": 256},
            "evaluation": {"n_samples": 10000, "model_evaluations": 512},
            "data": {"input": {"sha256": "a" * 64, "path": "cached_input.pt"}},
        }
        self.domain_metrics = {
            "vector": {"radial_w1": 1., "ks_stat": .01, "sliced_w1": 1., "mmd": .02,
                       "q950_err": .1, "q990_err": .2, "q995_err": .3,
                       "tail_exc_95": .001, "tail_exc_99": .002,
                       "angular_sw_bin0": .1, "angular_sw_bin1": .2,
                       "angular_sw_bin2": .3, "angular_sw_bin3": .4, "angular_sw_mean": .25,
                       "nan_rate": 0., "invalid_rate": 0., "exploding_norm_rate": 0., "inf_rate": 0.},
            "audio": {"digit_acc": .7, "energy_KS": .01, "radial_w1": 1., "cov>q90": .1,
                      "cov>q95": .05, "cov>q99": .01, "cov<q10": .1, "PIT": .5},
            "image": {"fid": 100., "kid": .02, "kid_std": .001, "radial_w1": 1.,
                      "ks": .01, "sliced_w1": 1., "precision": .5, "recall": .5,
                      "density": .6, "coverage": .7},
        }

    def record(self, seed, value, **changes):
        config = copy.deepcopy(self.config)
        implementation = changes.pop("implementation", {"files": {"fixture.py": "c" * 64},
                                                        "python": "unit-test fixture"})
        metrics = {**self.domain_metrics[config["kind"]], "total_train_time_s": 20.,
                   "sample_time_s": 1., "nfe": 512, "radial_w1": value}
        if "sliced_w1" in metrics:
            metrics["sliced_w1"] = value
        result = {
            "schema_version": 1, "method": "tflow", "condition_id": "piv_d64", "seed": seed,
            "status": "complete", "config": config,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "implementation_sha256": hashlib.sha256(json.dumps(implementation, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "hardware": {"gpu": "unit-test fixture; no GPU used"}, "source": {"nu": 5.0, "scale": .5},
            "metrics": metrics,
        }
        result.update(changes)
        path = self.root / "piv_d64" / f"seed_{seed}" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name("implementation.json").write_text(json.dumps(implementation))
        path.write_text(json.dumps(result))
        return path

    def all_complete(self):
        for seed, value in zip(self.condition["model_seeds"], (1.0, 2.0, 3.0)):
            self.record(seed, value)

    def report(self):
        return collect_suite({"benchmarks": [self.condition], "paper": {"sha256": "b" * 64}}, self.root)

    def test_complete_uses_all_three_and_population_sd(self):
        self.all_complete()
        row = aggregate_condition(self.condition, self.root)
        self.assertEqual(row["status"], "complete")
        stat = row["aggregate_full_precision"]["sliced_w1"]
        self.assertEqual(stat["vals"], [1.0, 2.0, 3.0])
        self.assertEqual(stat["n"], 3)
        self.assertEqual(stat["mean"], 2.0)
        self.assertAlmostEqual(stat["std"], statistics.pstdev([1.0, 2.0, 3.0]))

    def test_missing_seed_is_pending_without_survivor_mean(self):
        self.record(8925, 1.0)
        self.record(77395, 2.0)
        report = self.report()
        row = report["conditions"]["piv_d64"]
        self.assertEqual((row["status"], row["n_complete"], row["n_pending"]), ("pending", 2, 1))
        self.assertEqual(row["aggregate_full_precision"], {})
        bundle = render_bundle(report)
        self.assertIn("PENDING (2/3)", bundle["table1_tflow_addition.tex"])
        self.assertNotIn("1.5", bundle["table1_tflow_addition.tex"])
        target = self.root / "should_not_exist"
        with self.assertRaises(ValueError):
            write_bundle(report, target)
        self.assertFalse(target.exists())

    def test_failed_seed_kept_with_failure_and_no_aggregate(self):
        self.all_complete()
        self.record(65457, 3.0, status="failed", failure={"type": "FloatingPointError", "message": "one invalid sample"})
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual((row["status"], row["n_complete"], row["n_failed"]), ("failed", 2, 1))
        self.assertEqual(row["aggregate"], {})
        self.assertEqual(row["runs"][2]["failure"]["message"], "one invalid sample")

    def test_nested_nonfinite_metric_cannot_hide_in_complete_result(self):
        self.all_complete()
        path = self.root / "piv_d64/seed_65457/result.json"
        data = json.loads(path.read_text())
        data["metrics"]["diagnostics"] = {"nested": [1.0, float("nan")]}
        path.write_text(json.dumps(data))
        report = self.report()
        self.assertEqual(report["conditions"]["piv_d64"]["status"], "failed")
        encoded = render_bundle(report)["tflow_results.json"]
        json.loads(encoded, parse_constant=lambda value: self.fail(f"invalid JSON constant {value}"))
        self.assertIn("nonfinite values", encoded)

    def test_forged_hash_or_input_identity_rejected(self):
        self.all_complete()
        self.record(8925, 1.0, config_sha256="f" * 64)
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["runs"][0]["status"], "invalid")
        self.assertTrue(any("canonical config" in issue for issue in row["runs"][0]["issues"]))

    def test_wrong_condition_and_unresolved_protocol_not_aggregated(self):
        self.all_complete()
        self.record(8925, 1.0, condition_id="wrong_condition")
        self.assertEqual(self.report()["conditions"]["piv_d64"]["status"], "failed")
        self.all_complete()
        self.condition["resolved_protocol"] = False
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["status"], "invalid")
        self.assertEqual(row["aggregate"], {})

    def test_source_must_be_frozen_across_final_seeds(self):
        self.all_complete()
        self.record(65457, 3.0, source={"nu": 7.0, "scale": .5})
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["status"], "invalid")
        self.assertIn("frozen source", row["issues"][0])
        self.assertEqual(row["aggregate_full_precision"], {})

    def test_implementation_fingerprint_is_required_and_verified(self):
        for corruption in ("missing_hash", "forged_hash", "missing_manifest", "modified_manifest"):
            with self.subTest(corruption=corruption):
                self.all_complete()
                path = self.root / "piv_d64/seed_65457/result.json"
                data = json.loads(path.read_text())
                if corruption == "missing_hash":
                    data.pop("implementation_sha256")
                elif corruption == "forged_hash":
                    data["implementation_sha256"] = "d" * 64
                elif corruption == "missing_manifest":
                    path.with_name("implementation.json").unlink()
                else:
                    path.with_name("implementation.json").write_text('{"changed": true}')
                path.write_text(json.dumps(data))
                row = self.report()["conditions"]["piv_d64"]
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["aggregate"], {})
                self.assertIn("implementation", "; ".join(row["runs"][2]["issues"]))

    def test_three_valid_implementation_fingerprints_must_match(self):
        self.all_complete()
        self.record(65457, 3., implementation={"files": {"fixture.py": "d" * 64}, "python": "fixture"})
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["status"], "invalid")
        self.assertEqual(row["n_complete"], 3)
        self.assertEqual(row["aggregate"], {})
        self.assertIn("implementation fingerprint", "; ".join(row["issues"]))

    def test_every_domain_metric_and_timing_is_required(self):
        for kind in ("vector", "audio", "image"):
            self.condition["protocol"] = self.config["kind"] = kind
            self.all_complete()
            self.assertEqual(self.report()["conditions"]["piv_d64"]["status"], "complete")
            for metric in REQUIRED_METRICS[kind]:
                with self.subTest(kind=kind, missing_metric=metric):
                    path = self.record(65457, 3.)
                    data = json.loads(path.read_text())
                    data["metrics"].pop(metric)
                    path.write_text(json.dumps(data))
                    row = self.report()["conditions"]["piv_d64"]
                    self.assertEqual(row["status"], "failed")
                    self.assertEqual(row["aggregate"], {})
                    self.assertIn(f"required metric {metric}", "; ".join(row["runs"][2]["issues"]))

    def test_undefined_angular_bin_and_declared_nonfinite_fields_prevent_success(self):
        self.all_complete()
        path = self.record(65457, 3.)
        data = json.loads(path.read_text())
        data["metrics"]["angular_sw_bin3"] = None
        path.write_text(json.dumps(data))
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["aggregate"], {})
        self.record(65457, 3., nonfinite_metric_fields=["angular_sw_bin3"])
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["status"], "failed")
        self.assertIn("undefined/nonfinite", "; ".join(row["runs"][2]["issues"]))

    def test_nonzero_nonfinite_sample_rate_prevents_success(self):
        self.all_complete()
        for metric in ("nan_rate", "inf_rate"):
            with self.subTest(metric=metric):
                path = self.record(65457, 3.)
                data = json.loads(path.read_text())
                data["metrics"][metric] = .0001
                path.write_text(json.dumps(data))
                row = self.report()["conditions"]["piv_d64"]
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["aggregate"], {})

    def test_finite_heavy_tail_flags_remain_reportable_for_all_three_seeds(self):
        rates = [.0001, .0003, .0002]
        for seed, rate in zip(self.condition["model_seeds"], rates):
            path = self.record(seed, 1.)
            data = json.loads(path.read_text())
            data["metrics"].update(exploding_norm_rate=rate, invalid_rate=rate)
            path.write_text(json.dumps(data))
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual((row["status"], row["n_complete"]), ("complete", 3))
        for metric in ("exploding_norm_rate", "invalid_rate"):
            stats = row["aggregate_full_precision"][metric]
            self.assertEqual(stats["vals"], rates)
            self.assertEqual(stats["n"], 3)
            self.assertAlmostEqual(stats["mean"], statistics.mean(rates))
            self.assertAlmostEqual(stats["std"], statistics.pstdev(rates))
            self.assertEqual([run["native_metrics"][metric] for run in row["runs"]], rates)

    def test_native_downstream_aliases_and_conflicting_values(self):
        audio = normalize_metrics({"content": {"digit_acc": .7}, "energy": {"ks": .02, "cov_gt_q95": .05}}, "audio")
        self.assertEqual(audio["digit_acc"], .7)
        self.assertEqual(audio["energy_KS"], .02)
        self.assertEqual(audio["cov>q95"], .05)
        image = normalize_metrics({"latent": {"ks": .04}, "image": {"fid": 100}}, "image")
        self.assertEqual(image["ks"], .04)
        self.assertEqual(image["fid"], 100)
        with self.assertRaises(ValueError):
            normalize_metrics({"fid": 99, "image": {"fid": 100}}, "image")

    def test_additions_never_read_or_overwrite_historical_values(self):
        self.all_complete()
        report = self.report()
        old = self.root / "historical"
        old.mkdir()
        reference = old / "table1.tex"
        reference.write_text("immutable historical values")
        with self.assertRaises(FileExistsError):
            write_bundle(report, old)
        self.assertEqual(reference.read_text(), "immutable historical values")
        target = self.root / "new_additions"
        write_bundle(report, target)
        self.assertTrue((target / "table1_tflow_addition.tex").exists())
        self.assertEqual(json.loads((target / "conditions/piv_d64.json").read_text())["status"], "complete")
        self.assertFalse(report["historical_values_recomputed"])

    def test_all_expected_conditions_stay_visible_even_without_results(self):
        other = {**self.condition, "id": "piv_d256"}
        report = collect_suite({"benchmarks": [self.condition, other]}, self.root)
        self.assertEqual((report["n_conditions"], report["n_expected_runs"], report["n_complete_runs"]), (2, 6, 0))
        self.assertEqual(set(report["conditions"]), {"piv_d64", "piv_d256"})
        self.assertIn("piv_d256", render_bundle(report)["tflow_results.md"])

    def test_unexpected_seed_or_path_component_is_refused(self):
        self.condition["model_seeds"] = [8925, 8925, 7]
        with self.assertRaises(ValueError):
            aggregate_condition(self.condition, self.root)
        self.condition["model_seeds"] = [8925, 77395, 65457]
        self.condition["id"] = "../outside"
        with self.assertRaises(ValueError):
            aggregate_condition(self.condition, self.root)

    def test_extra_measured_seed_is_not_silently_ignored(self):
        self.all_complete()
        self.record(42, 4.0)
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["status"], "invalid")
        self.assertEqual(row["aggregate"], {})
        self.assertIn("unexpected final-seed", row["issues"][0])

    def test_wrong_run_budget_is_rejected_even_with_valid_config_hash(self):
        self.all_complete()
        self.config["training"]["steps"] = 500
        self.record(8925, 1.0)
        row = self.report()["conditions"]["piv_d64"]
        self.assertEqual(row["runs"][0]["status"], "invalid")
        self.assertIn("training.steps differs", "; ".join(row["runs"][0]["issues"]))


if __name__ == "__main__":
    unittest.main()
