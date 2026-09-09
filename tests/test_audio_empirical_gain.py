"""Small correctness checks; no training, checkpoint loading, or benchmark runs."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rafm.sources.radial_empirical import RadialEmpiricalSource

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/poc_audio/audio_empirical_gain.py"
SPEC = importlib.util.spec_from_file_location("audio_empirical_gain", SCRIPT)
audio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audio)


def metadata(seed=8925):
    return {"R0": 2.3, "D": 16254, "args": {
        "method": "fixed_spherical", "seed": seed, "split_seed": 0,
        "arch": "unet", "ch": 96, "ncls": 10, "angular": False,
        "steps": 24000, "batch": 32, "lr": 2e-4, "ema": .999,
        "class_dropout": .1, "depth": 5,
    }}


@pytest.mark.parametrize("key,value", [("method", "rafm"), ("seed", 7),
                                      ("split_seed", 1), ("angular", True), ("steps", 12000)])
def test_refuses_different_training_protocol(key, value):
    meta = metadata()
    meta["args"][key] = value
    with pytest.raises(ValueError):
        audio.validate_metadata(meta, 8925)


def test_refuses_missing_or_fabricated_radius():
    for radius in (None, float("nan"), 0, -1):
        meta = metadata()
        meta["R0"] = radius
        with pytest.raises(ValueError):
            audio.validate_metadata(meta, 8925)


def test_original_evaluator_legacy_defaults_preserve_metadata():
    meta = metadata()
    del meta["args"]["ncls"]
    del meta["args"]["angular"]
    before = json.dumps(meta, sort_keys=True)
    audio.validate_metadata(meta, 8925)
    assert json.dumps(meta, sort_keys=True) == before


@pytest.mark.parametrize("key,value", [("ncls", 11), ("depth", 4), ("angular", True)])
def test_explicit_legacy_or_architecture_conflicts_are_refused(key, value):
    meta = metadata()
    meta["args"][key] = value
    with pytest.raises(ValueError):
        audio.validate_metadata(meta, 8925)


def test_preflight_does_not_load_checkpoint_or_create_outputs(tmp_path, monkeypatch):
    def disallow_load(*args, **kwargs):
        raise AssertionError("preflight must not deserialize tensors")
    monkeypatch.setattr(torch, "load", disallow_load)
    shared = {}
    for name in ("train_file", "test_file", "classifier"):
        path = tmp_path / name
        path.write_bytes(b"placeholder; not a tensor file")
        shared[name] = str(path)
    reference = {"n_seeds": 3, **{key: {"vals": [.1, .1, .1]} for key in
                 ("digit_acc", "energy_KS", "cov>q95", "cov>q99", "cov<q10", "PIT")}}
    ref_file = tmp_path / "reference.json"
    ref_file.write_text(json.dumps({"24000": {"fixed_spher": reference}}))
    runs = []
    for seed in audio.SEEDS:
        checkpoint, meta = tmp_path / f"{seed}.pt", tmp_path / f"{seed}.json"
        checkpoint.write_bytes(b"not loaded during preflight")
        meta.write_text(json.dumps(metadata(seed)))
        runs.append((str(seed), str(checkpoint), str(meta)))
    args = argparse.Namespace(**shared, reference_aggregate=str(ref_file),
                              output_dir=str(tmp_path / "new_results"), run=runs)
    _, validated, _ = audio.preflight(args)
    assert list(validated) == list(audio.SEEDS)
    assert not Path(args.output_dir).exists()
    args.run = runs[:2]
    with pytest.raises(ValueError, match="all three"):
        audio.preflight(args)


def test_exact_ecdf_draw_is_isolated_and_matches_original_draw_order():
    training = torch.tensor([[1., 0.], [4., 0.], [9., 0.]])
    source = RadialEmpiricalSource(mode="ecdf").fit(training)
    before = torch.random.get_rng_state().clone()
    gains = audio.sample_training_gains(source, 17, seed=0)
    assert torch.equal(before, torch.random.get_rng_state())
    quantiles = torch.rand(17, generator=torch.Generator().manual_seed(0))
    assert torch.equal(gains, torch.quantile(training.norm(dim=1), quantiles))
    # Interpolation is essential: this is not empirical-index resampling.
    assert not torch.isin(gains, training.norm(dim=1)).any()
    replay = audio.replay_original_source_radii(source, 17, dimension=2, seed=0)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(replay, gains, rtol=2e-6, atol=1e-6)


def test_postprocessing_preserves_direction_and_sets_canonical_radius():
    y = torch.tensor([[3., 4.], [-4., 3.], [0., -5.]])
    gains = torch.tensor([.2, 2., 8.])
    x = audio.apply_gains(y, gains)
    torch.testing.assert_close(x.norm(dim=1), gains)
    torch.testing.assert_close(x / x.norm(dim=1, keepdim=True), y / 5)
    assert torch.equal(y, torch.tensor([[3., 4.], [-4., 3.], [0., -5.]]))


@pytest.mark.parametrize("y,g", [([[0., 0.]], [1.]), ([[1., 0.]], [0.]),
                                ([[float("inf"), 0.]], [1.]), ([[1., 0.]], [float("nan")])])
def test_invalid_direction_or_gain_is_rejected(y, g):
    with pytest.raises(ValueError):
        audio.apply_gains(torch.tensor(y), torch.tensor(g))


def test_prediction_changes_fail_even_when_aggregate_accuracy_is_identical():
    y = torch.tensor([[1., 0.], [0., 1.]])
    gains = torch.tensor([2., 3.])
    x = audio.apply_gains(y, gains)
    before = torch.tensor([[2., 0.], [0., 2.]])
    after = before.flip(0)
    check = audio.invariant_checks(y, x, gains, torch.tensor([0, 0]), before, after)
    assert check["accuracy_difference"] == 0
    assert check["prediction_disagreements"] == 2
    assert not check["passed"]


def test_complete_control_checks_actual_classifier_normalized_inputs():
    # A tiny classifier keeps this a unit check; both distinct gain levels must
    # leave each normalized STFT classifier input and prediction unchanged.
    class DirectionClassifier(torch.nn.Module):
        def forward(self, x):
            flat = x.flatten(1)
            return torch.stack((flat[:, 0], flat[:, 1], flat[:, 2]), dim=1)

    y = torch.zeros(3, audio.DIM)
    y[torch.arange(3), torch.arange(3)] = 2.5
    gains = torch.tensor([.3, 2., 6.])
    x, _, _, baseline, posthoc, check = audio.evaluate_gain_invariance(
        y, gains, torch.arange(3), DirectionClassifier(), np.array([.2, 1., 3., 7.]))
    assert check["passed"]
    assert check["prediction_disagreements"] == 0
    assert baseline["unrounded"]["correct_count"] == posthoc["unrounded"]["correct_count"] == 3
    assert baseline["content"]["digit_acc"] == posthoc["content"]["digit_acc"] == 1
    torch.testing.assert_close(x.norm(dim=1), gains)
    assert posthoc["energy"]["gen_energy_mean"] == round(float(gains.mean()), 3)


def test_positive_squared_energy_has_same_ks_and_test_quantile_tails():
    gains = np.array([.5, 1., 1., 2., 8., 9.])
    test = np.array([.4, .8, 1.2, 1.6, 3., 4.])
    _, raw = audio.energy_metrics(gains, test)
    _, squared = audio.energy_metrics(gains ** 2, test ** 2)
    assert raw["ks"] == squared["ks"]
    assert raw["cov_gt_q95"] == 2 / 6
    assert raw["cov_gt_q99"] == 2 / 6
    assert raw["cov_lt_q10"] == 1 / 6


def test_partial_seed_set_cannot_be_aggregated():
    with pytest.raises(ValueError, match="complete passing"):
        audio.aggregate_runs([])


def test_reference_controls_require_all_three_original_seeds_and_correct_target(tmp_path):
    entries = []
    for seed in audio.SEEDS:
        checkpoint, meta_file = tmp_path / f"ref_{seed}.pt", tmp_path / f"ref_{seed}.json"
        checkpoint.write_bytes(b"not loaded by metadata preflight")
        meta = metadata(seed)
        meta["args"]["method"] = "rafm"
        meta["args"]["angular"] = True
        meta_file.write_text(json.dumps(meta))
        entries.append(("angular_rafm", str(seed), str(checkpoint), str(meta_file)))
    with pytest.raises(ValueError, match="three distinct"):
        audio.preflight_reference_runs(entries[:2])
    validated = audio.preflight_reference_runs(entries)
    assert [row["seed"] for row in validated] == list(audio.SEEDS)
    assert all(row["method"] == "rafm" and row["angular"] is True for row in validated)
    wrong_label = [("rafm", *entry[1:]) for entry in entries]
    with pytest.raises(ValueError, match="angular"):
        audio.preflight_reference_runs(wrong_label)


def test_original_rk4_has_160_model_evaluations_and_preserves_zero_field_state():
    class ZeroField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, time, labels):
            self.calls += 1
            return torch.zeros_like(x)

    model = ZeroField()
    initial = torch.ones(1, audio.DIM)
    result = audio.rk4_original(model, initial, torch.tensor([0]), spherical=True, angular=True)
    assert model.calls == 160
    assert torch.equal(result, initial)


def test_evaluate_refuses_login_execution_before_loading_inputs(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
        audio.evaluate(None, None, None, None)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    audio.require_compute_allocation()


def test_tensor_datasets_default_to_data_root_and_distinct_report_hashes(tmp_path):
    first = argparse.Namespace(output_dir=tmp_path / "report_a")
    second = argparse.Namespace(output_dir=tmp_path / "report_b")
    a, b = audio.sample_run_directory(first), audio.sample_run_directory(second)
    assert a.parent == Path("/mnt/vast01/users/fouad.oubari/data/tflow/audio_gain")
    assert b.parent == a.parent
    assert a != b
    assert audio.sample_run_directory(first) == a
    first.samples_root = tmp_path / "explicit_sample_override"
    assert audio.sample_run_directory(first).parent == first.samples_root


def test_tensor_artifact_records_absolute_path_checksum_and_size(tmp_path):
    path = tmp_path / "artifact_metadata_fixture"
    path.write_bytes(b"abc")
    artifact = audio.tensor_artifact(path)
    assert artifact["path"] == str(path.resolve())
    assert artifact["size_bytes"] == 3
    assert artifact["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
