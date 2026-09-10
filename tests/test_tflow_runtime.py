"""Runtime contracts using tiny fixtures/analytic callbacks; never optimizer steps."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from baselines.tflow_core import TFlowSourceConfig
from experiments.tflow import data as inputs
from experiments.tflow import run
from experiments.tflow.validation import selection_score, source_candidates


def config_fixture():
    return {
        "schema_version": 1, "condition_id": "unit_fixture", "kind": "vector",
        "protocol_status": "resolved", "blocking_issues": [],
        "seeds": [8925, 77395, 65457],
        "model": {"kind": "mlp", "hidden_dim": 4, "n_layers": 1},
        "data": {"shape": [6, 2]},
        "training": {"steps": 10000, "optimizer": "Adam", "lr": 0.001,
                     "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
                     "gradient_clip": None, "ema": None, "batch_size": 2,
                     "checkpoint_every": 100, "log_every": 100,
                     "batch_rule": "global_torch", "precision": "float32"},
        "evaluation": {"model_evaluations": 4, "n_samples": 5,
                       "sample_batch_size": 2, "sample_seed": 0, "metric_seed": 0},
        "sampler": {"t_min": 0.1, "grid": "linear", "rho": 7, "sigma_min": 0.01},
    }


def test_configuration_rejects_nonfinite_and_invalid_budgets():
    cfg = config_fixture()
    run.validate_config(cfg)
    for section, key, value in (("training", "lr", float("nan")),
                                ("training", "eps", 0),
                                ("evaluation", "model_evaluations", 0),
                                ("evaluation", "model_evaluations", 3),
                                ("evaluation", "sample_batch_size", -1)):
        bad = copy.deepcopy(cfg)
        bad[section][key] = value
        with pytest.raises(ValueError):
            run.validate_config(bad)


def test_nested_nonfinite_metrics_are_explicit_and_json_safe(tmp_path):
    raw = {"image": {"fid": np.float64(float("nan")), "kid": 0.2},
           "bins": [0.5, torch.tensor(float("inf"))], "undefined": None}
    clean, bad = run.sanitize_metrics(raw)
    assert bad == ["image.fid", "bins[1]", "undefined"]
    assert clean["image"] == {"fid": None, "kid": 0.2}
    path = tmp_path / "failure.json"
    run.write_json(path, {"metrics": clean, "nonfinite_metric_fields": bad})
    assert json.loads(path.read_text())["metrics"]["bins"] == [0.5, None]


def test_run_identity_changes_with_implementation_and_source(monkeypatch):
    cfg, source = config_fixture(), TFlowSourceConfig(3, 1)
    fingerprint = {"files": {"core.py": "a"}}
    monkeypatch.setattr(run, "implementation_manifest", lambda cfg: fingerprint)
    original = run.json_hash(run.run_signature(cfg, 8925, source))
    fingerprint["files"]["core.py"] = "b"
    assert run.json_hash(run.run_signature(cfg, 8925, source)) != original
    fingerprint["files"]["core.py"] = "a"
    assert run.json_hash(run.run_signature(cfg, 8925, TFlowSourceConfig(5, 1))) != original
    assert run.json_hash(run.run_signature(cfg, 8925, source, "tuning")) != original


def test_frozen_selection_requires_matching_code_and_validation_split(monkeypatch, tmp_path):
    cfg = config_fixture()
    monkeypatch.setattr(run, "implementation_sha256", lambda cfg: "a" * 64)
    selection = {"status": "frozen", "config_sha256": run.json_hash(cfg),
                 "implementation_sha256": "a" * 64, "selection_split": "validation",
                 "selected": {"nu": 3, "scale": 0.8}}
    path = tmp_path / "selection.json"
    run.write_json(path, selection)
    assert run.source_from_selection(cfg, path) == TFlowSourceConfig(3, 0.8)
    selection["implementation_sha256"] = "b" * 64
    run.write_json(path, selection)
    with pytest.raises(ValueError, match="implementation"):
        run.source_from_selection(cfg, path)
    selection["implementation_sha256"] = "a" * 64
    selection["selection_split"] = "test"
    run.write_json(path, selection)
    with pytest.raises(ValueError, match="validation"):
        run.source_from_selection(cfg, path)


def test_cpu_rng_checkpoint_roundtrip_includes_python_numpy_torch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = run.rng_state()
    try:
        first = (torch.rand(3), np.random.random(3), random.random())
        run.restore_rng(saved)
        second = (torch.rand(3), np.random.random(3), random.random())
        assert torch.equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert first[2] == second[2]
    finally:
        run.restore_rng(saved)


def test_sampling_counts_all_batches_and_restores_model_mode(monkeypatch):
    cfg = config_fixture()
    initial = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    monkeypatch.setattr(run, "sample_student_t", lambda shape, source, device: initial.to(device))

    class AnalyticNoise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, t):
            assert not self.training
            self.calls += 1
            # Constant drift +1, requiring no fitted parameters.
            return x - t[:, None]

    model = AnalyticNoise().train()
    result = run.sample(cfg, model, TFlowSourceConfig(), 5)
    torch.testing.assert_close(result["samples"], initial + 0.9)
    assert result["nfe"] == 4
    assert result["n_batches"] == 3
    assert result["model_calls_total"] == model.calls == 12
    assert result["time_grid"].device.type == "cpu"
    assert result["time_grid"].tolist() == pytest.approx([0.1, 0.55, 1.0])
    assert model.training


def test_cached_sample_paths_distinguish_checkpoint_bytes(monkeypatch, tmp_path):
    cfg = config_fixture()
    cfg["evaluation"]["samples_root"] = str(tmp_path)
    monkeypatch.setattr(run, "implementation_sha256", lambda cfg: "a" * 64)
    first = run.sample_artifact_path(cfg, 8925, TFlowSourceConfig(), checkpoint_sha256="b" * 64)
    second = run.sample_artifact_path(cfg, 8925, TFlowSourceConfig(), checkpoint_sha256="c" * 64)
    assert first != second
    assert first.is_relative_to(tmp_path)
    assert not first.exists()


def test_completed_resume_preserves_loss_and_executes_no_optimizer_step(monkeypatch, tmp_path):
    cfg, source = config_fixture(), TFlowSourceConfig()
    monkeypatch.setattr(run, "compute_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(run, "implementation_sha256", lambda cfg: "a" * 64)
    monkeypatch.setattr(run, "implementation_manifest", lambda cfg: {"files": {"unit": "a" * 64}})
    monkeypatch.setattr(run, "hardware", lambda: {"kind": "unit fixture"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class NoStepOptimizer:
        def __init__(self, parameters, **kwargs):
            self.state = {}

        def load_state_dict(self, state):
            assert state == {}

        def step(self):
            pytest.fail("a completed resume must never execute an optimizer step")

    monkeypatch.setattr(run, "build_model", lambda *args: FakeModel())
    monkeypatch.setattr(torch.optim, "Adam", NoStepOptimizer)
    state = run.rng_state()
    saved = {"run_sha256": run.json_hash(run.run_signature(cfg, 8925, source)),
             "model": {"weight": torch.zeros(1)}, "ema": None, "optimizer": {},
             "step": 10000, "train_time_s": 123.0, "last_loss": 2.5,
             "rng": state, "stage": "final"}
    torch.save(saved, tmp_path / "checkpoint.pt")
    data = inputs.CachedData(torch.ones(6, 2), None,
                            {"train": torch.arange(4), "val": torch.tensor([4]), "test": torch.tensor([5])},
                            torch.zeros(2))
    try:
        result = run.train(cfg, data, tmp_path, 8925, source)
        assert result["final_loss"] == 2.5
        assert result["total_train_time_s"] == 123.0
        assert result["training_step"] == 10000
    finally:
        run.restore_rng(state)


def test_invalid_run_identity_does_not_overwrite_config(monkeypatch, tmp_path):
    cfg = config_fixture()
    original = {"other": "run"}
    run.write_json(tmp_path / "config.json", original)
    monkeypatch.setattr(run, "compute_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(run, "implementation_sha256", lambda cfg: "a" * 64)
    with pytest.raises(ValueError, match="different run"):
        run.train(cfg, None, tmp_path, 8925, TFlowSourceConfig())
    assert json.loads((tmp_path / "config.json").read_text()) == original


def test_fractional_split_indices_are_rejected_before_cast(monkeypatch):
    monkeypatch.setattr(inputs, "load_pinned", lambda spec: {
        "train": torch.tensor([0.1, 1.0]), "val": torch.tensor([2]), "test": torch.tensor([3])})
    with pytest.raises(ValueError, match="integer vector"):
        inputs.split_indices(4, {"kind": "indices", "file": {}, "n_train": 2, "n_val": 1, "n_test": 1})


@pytest.mark.parametrize("dim", [49, 64])
def test_loader_shape_and_training_only_centering_with_tiny_vectors(monkeypatch, dim):
    # Synthetic contracts only; real cached inputs are checked by check_cached_inputs.py.
    values = torch.arange(6, dtype=torch.float32)[:, None].expand(6, dim).clone()
    monkeypatch.setattr(inputs, "load_pinned", lambda spec: values)
    cfg = {"kind": "vector", "data": {"input": {}, "shape": [6, dim],
           "split": {"kind": "chrono", "n_train": 3, "n_val": 1, "n_test": 2},
           "transform": "train_center"}}
    result = inputs.load_data(cfg)
    assert result.values.shape == (6, dim)
    torch.testing.assert_close(result.mean, torch.ones(dim))
    torch.testing.assert_close(result.split("test")[:, 0], torch.tensor([3.0, 4.0]))


def test_audio_tuning_does_not_read_external_test(monkeypatch):
    calls = []

    def pinned(spec):
        calls.append(spec["path"])
        if spec["path"] != "/train-fixture.pt":
            pytest.fail("tuning read the external test resource")
        return {"x": torch.ones(6, 2), "digit": torch.arange(6)}

    monkeypatch.setattr(inputs, "load_pinned", pinned)
    cfg = {"kind": "audio", "data": {"input": {"path": "/train-fixture.pt"},
           "external_test": {"path": "/test-fixture.pt"}, "shape": [6, 2],
           "split": {"kind": "chrono", "n_train": 3, "n_val": 3, "n_test": 0},
           "transform": "as_cached"}}
    result = inputs.load_data(cfg, include_external_test=False)
    assert calls == ["/train-fixture.pt"]
    assert result.external_test is None and result.external_gains is None


def test_dataset_manifest_distinguishes_split_order_and_external_test():
    data = inputs.CachedData(torch.arange(12).float().reshape(6, 2), None,
                             {"train": torch.tensor([0, 1, 2]),
                              "val": torch.tensor([3, 4, 5]), "test": torch.empty(0, dtype=torch.long)},
                             torch.zeros(2), external_test=torch.tensor([[50., 60.]]),
                             external_gains=torch.tensor([2.]))
    first = run.dataset_manifest(data)
    assert first["external_test_loaded"] is True
    assert first["split_indices"]["test"]["shape"] == [0]
    assert first["splits"]["test"]["shape"] == [1, 2]
    data.indices["train"] = torch.tensor([2, 1, 0])
    changed = run.dataset_manifest(data)
    assert first["values"] == changed["values"]
    assert first["splits"]["train"]["sha256"] != changed["splits"]["train"]["sha256"]
    assert first["split_indices"]["train"]["sha256"] != changed["split_indices"]["train"]["sha256"]
    assert first["splits"]["val"] == changed["splits"]["val"]


def test_peak_memory_retains_measured_maximum_across_resume(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 300)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 400)
    assert run.peak_memory(torch.device("cuda"), {"allocated_bytes": 500, "reserved_bytes": 200}) == {
        "allocated_bytes": 500, "reserved_bytes": 400}
    assert run.peak_memory(torch.device("cpu")) is None


def test_finite_gradients_keeps_none_semantics_and_rejects_nonfinite():
    model = torch.nn.Linear(2, 2)
    assert run.finite_gradients(model)
    model.weight.grad = torch.ones_like(model.weight)
    assert run.finite_gradients(model)
    model.bias.grad = torch.tensor([0., float("nan")])
    assert not run.finite_gradients(model)


def test_heavy_finite_rows_do_not_overflow_validation_norms():
    values = torch.tensor([[1e30, -1e30], [2e30, 1e30], [-1e30, 3e30]], dtype=torch.float32)
    choices = source_candidates(values)
    assert all(np.isfinite(choice["scale"]) for choice in choices)
    assert selection_score(values, values, n_projections=4)["selection_score"] == 0
