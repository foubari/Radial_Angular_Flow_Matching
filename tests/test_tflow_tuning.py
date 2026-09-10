"""Selection control-flow tests with a fake trainer; no optimizer/model runs."""
import copy
import json
from types import SimpleNamespace

import pytest

from baselines.tflow_core import TFlowSourceConfig
from experiments.tflow import prepare_configs, run, sanity, tune


def tuning_config():
    return {
        "schema_version": 1, "condition_id": "mock_selection", "kind": "vector",
        "protocol_status": "resolved", "blocking_issues": [],
        "seeds": [8925, 77395, 65457],
        "model": {"kind": "mlp", "hidden_dim": 4, "n_layers": 1},
        "data": {"shape": [6, 2]},
        "training": {"steps": 10000, "optimizer": "Adam", "lr": 0.001,
                     "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
                     "gradient_clip": None, "ema": None, "batch_size": 2,
                     "checkpoint_every": 100, "log_every": 100,
                     "batch_rule": "global_torch", "precision": "float32"},
        "evaluation": {"model_evaluations": 4, "n_samples": 10,
                       "sample_batch_size": 2, "sample_seed": 0, "metric_seed": 0},
        "sampler": {"t_min": 0.1, "grid": "linear", "rho": 7, "sigma_min": 0.01},
        "tuning": {"seed": 46021, "nu": [3, 5, 7], "scale_multipliers": [0.5, 1, 2],
                   "stage1_fraction": 0.05, "continuation_fraction": 0.10,
                   "n_finalists": 2, "validation_generated_samples": 1000,
                   "sample_seed": 61717, "projection_seed": 61719},
    }


@pytest.fixture
def fake_selection(monkeypatch):
    cfg = tuning_config()
    calls, sampling_calls, split_reads, load_options = [], [], [], []
    checkpoints = {}
    train_rows, validation_rows = [[1, 0]] * 4, [[2, 0]] * 2

    class TrainValidationOnly:
        values = SimpleNamespace(shape=(6, 2))

        def split(self, name):
            split_reads.append(name)
            assert name in ("train", "val"), "Selection must never inspect test rows"
            return train_rows if name == "train" else validation_rows

    data = TrainValidationOnly()

    def load_data(config, *, include_external_test):
        assert config == cfg
        load_options.append(include_external_test)
        assert include_external_test is False
        return data

    choices = [{"nu": nu, "scale": multiplier, "scale_multiplier": multiplier,
                "reference_scale": 1.0, "training_median_radius": 2.0}
               for nu in (3, 5, 7) for multiplier in (0.5, 1, 2)]

    def candidates(values):
        assert values is train_rows
        return copy.deepcopy(choices)

    def candidate_index(source):
        return next(i for i, choice in enumerate(choices)
                    if choice["nu"] == source.nu and choice["scale"] == source.scale)

    def train(config, loaded, output, seed, source, *, budget, stage):
        assert config == cfg and loaded is data and seed == 46021 and stage == "tuning"
        index = candidate_index(source)
        before = checkpoints.get(index, 0)
        assert budget >= before, "A cached 5% score must not request a rewind of a 10% checkpoint"
        calls.append((index, before, budget))
        checkpoints[index] = budget
        output.mkdir(parents=True, exist_ok=True)
        # Only an identity marker: no tensors, models, optimizer or training.
        run.write_json(output / "checkpoint.pt", {"candidate": index, "step": budget})

    def load_trained(config, loaded, output, seed, source, *, stage):
        index = candidate_index(source)
        assert loaded is data and stage == "tuning"
        saved = {"step": checkpoints[index], "stage": stage,
                 "run_sha256": run.json_hash(run.run_signature(config, seed, source, stage)),
                 "train_time_s": checkpoints[index] / 1000}
        return {"candidate": index, "step": saved["step"]}, saved

    def sample(config, model, source, n, *, seed):
        assert config == cfg and seed == 61717
        assert model["candidate"] == candidate_index(source)
        sampling_calls.append((model["candidate"], model["step"], n))
        return {"samples": {**model, "n_generated": n}, "sample_time_s": 0.25}

    def score(samples, validation, *, projection_seed):
        assert validation is validation_rows and projection_seed == 61719
        index = samples["candidate"]
        # Stage one selects 0,1; the larger-budget winner is 1.
        value = index / 10 if samples["step"] == 500 else (0.02 if index == 0 else 0.01)
        return {"selection_score": value, "radial_ks": value, "projected_ks_mean": value,
                "projection_seed": projection_seed, "n_projections": 64,
                "n_generated": samples["n_generated"], "n_validation": len(validation)}

    monkeypatch.setattr(tune, "load_data", load_data)
    monkeypatch.setattr(tune, "source_candidates", candidates)
    monkeypatch.setattr(tune, "train", train)
    monkeypatch.setattr(tune, "load_trained", load_trained)
    monkeypatch.setattr(tune, "sample", sample)
    monkeypatch.setattr(tune, "selection_score", score)
    monkeypatch.setattr(tune, "implementation_sha256", lambda cfg: "a" * 64)
    monkeypatch.setattr(run, "implementation_sha256", lambda cfg: "a" * 64)
    monkeypatch.setattr(tune, "hardware", lambda: {"kind": "mock control-flow test"})
    return SimpleNamespace(cfg=cfg, data=data, choices=choices, calls=calls,
                           sampling_calls=sampling_calls, split_reads=split_reads,
                           load_options=load_options, checkpoints=checkpoints)


def test_nine_plus_two_budget_validation_only_and_fresh_final_seed(fake_selection, tmp_path):
    fake = fake_selection
    result = tune.select(fake.cfg, tmp_path)
    assert fake.calls == [(i, 0, 500) for i in range(9)] + [(0, 500, 1000), (1, 500, 1000)]
    assert sum(after - before for _, before, after in fake.calls) == 5500
    assert result["full_training_equivalents"] == 0.55
    assert len(fake.sampling_calls) == 11
    assert result["selected"] == fake.choices[1]
    assert result["test_data_used_for_selection"] is False
    assert result["selection_split"] == "validation"
    assert fake.load_options == [False]
    assert set(fake.split_reads) == {"train", "val"}
    path = tmp_path / fake.cfg["condition_id"] / "selection.json"
    assert run.source_from_selection(fake.cfg, path) == TFlowSourceConfig(3, 1)
    assert all(row["checkpoint"]["stage"] == "tuning" for row in result["finalist_trials"])


def test_resume_uses_five_percent_scores_with_ten_percent_checkpoints(fake_selection, tmp_path):
    fake = fake_selection
    result = tune.select(fake.cfg, tmp_path)
    # Equivalent to stopping after score writes but before the selection freeze.
    (tmp_path / fake.cfg["condition_id"] / "selection.json").unlink()
    assert fake.checkpoints[0] == fake.checkpoints[1] == 1000
    fake.calls.clear()
    fake.sampling_calls.clear()
    resumed = tune.select(fake.cfg, tmp_path)
    assert resumed == result
    assert fake.calls == fake.sampling_calls == []
    # A subsequent frozen-result read also performs no training or sampling.
    assert tune.select(fake.cfg, tmp_path) == result
    assert fake.calls == fake.sampling_calls == []


@pytest.mark.parametrize("corruption", ["nonfinite", "run_hash", "steps", "status"])
def test_corrupt_cached_score_is_refused_without_retry(fake_selection, tmp_path, corruption):
    fake = fake_selection
    tune.select(fake.cfg, tmp_path)
    directory = tmp_path / fake.cfg["condition_id"]
    (directory / "selection.json").unlink()
    path = directory / "candidate_00" / "validation_step500.json"
    row = json.loads(path.read_text())
    if corruption == "nonfinite":
        row["metrics"]["selection_score"] = float("nan")
    elif corruption == "run_hash":
        row["checkpoint"]["run_sha256"] = "b" * 64
    elif corruption == "steps":
        row["checkpoint"]["step"] = 1000
    else:
        row["status"] = "failed"
    path.write_text(json.dumps(row))
    fake.calls.clear()
    with pytest.raises(ValueError, match="Cached validation"):
        tune.select(fake.cfg, tmp_path)
    assert fake.calls == []


@pytest.mark.parametrize("corruption", ["implementation", "test_split", "wrong_winner"])
def test_frozen_selection_identity_and_winner_are_verified(fake_selection, tmp_path, corruption):
    fake = fake_selection
    result = tune.select(fake.cfg, tmp_path)
    if corruption == "implementation":
        result["implementation_sha256"] = "b" * 64
    elif corruption == "test_split":
        result["selection_split"] = "test"
    else:
        result["selected"] = fake.choices[0]
    run.write_json(tmp_path / fake.cfg["condition_id"] / "selection.json", result)
    fake.calls.clear()
    with pytest.raises(ValueError):
        tune.select(fake.cfg, tmp_path)
    assert fake.calls == []


def test_wrong_checkpoint_step_fails_before_validation_sampling(fake_selection, monkeypatch, tmp_path):
    original = tune.load_trained

    def wrong_step(*args, **kwargs):
        model, saved = original(*args, **kwargs)
        saved["step"] += 1
        return model, saved

    monkeypatch.setattr(tune, "load_trained", wrong_step)
    with pytest.raises(ValueError, match="exact requested tuning checkpoint"):
        tune.select(fake_selection.cfg, tmp_path)
    assert fake_selection.sampling_calls == []
    report = json.loads((tmp_path / "mock_selection" / "candidate_00" / "failed.json").read_text())
    assert report["implementation_sha256"] == "a" * 64
    assert report["config_sha256"] == run.json_hash(fake_selection.cfg)
    assert report["stage"] == "tuning"
    before = list(fake_selection.calls)
    with pytest.raises(RuntimeError, match="previously failed"):
        tune.select(fake_selection.cfg, tmp_path)
    assert fake_selection.calls == before


def test_sanity_setup_failure_is_preserved_without_checkpoint(fake_selection, monkeypatch, tmp_path):
    monkeypatch.setattr(sanity, "implementation_sha256", lambda cfg: "a" * 64)
    monkeypatch.setattr(sanity, "compute_device", lambda: "cpu")
    monkeypatch.setattr(sanity, "hardware", lambda: {"kind": "mock setup test"})
    monkeypatch.setattr(sanity, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(sanity, "dataset_manifest", lambda data: {"kind": "mock data manifest"})

    def load_data(cfg, *, include_external_test):
        assert include_external_test is False
        return fake_selection.data

    def broken_model(*args):
        raise RuntimeError("Deliberate model setup failure")

    monkeypatch.setattr(sanity, "load_data", load_data)
    monkeypatch.setattr(sanity, "build_model", broken_model)
    with pytest.raises(RuntimeError, match="Deliberate model setup failure"):
        sanity.check(fake_selection.cfg, tmp_path)
    output = tmp_path / fake_selection.cfg["condition_id"]
    report = json.loads((output / "sanity.json").read_text())
    assert report["status"] == "failed"
    assert report["config_sha256"] == run.json_hash(fake_selection.cfg)
    assert report["implementation_sha256"] == "a" * 64
    assert report["checkpoint_written"] is False
    assert report["losses"] == []
    assert not list(output.glob("*.pt"))
    with pytest.raises(FileExistsError):
        sanity.check(fake_selection.cfg, tmp_path)


def test_config_materialization_preserves_split_seed_and_rejects_unknown_kind():
    manifest = {"inputs": {}, "paper": {"sha256": "a" * 64}, "audit": {"commit": "b" * 40}}
    case = {"id": "unit_case", "protocol": "vector", "dimension": 2, "family": "unit",
            "split": {"kind": "random_permutation", "seed": 19, "counts": [4, 1, 1], "n_total": 6},
            "cached_input_path": "/tmp/unused-unit-input.pt", "input_sha256": "c" * 64,
            "train_steps": 10000, "batch_size": 2, "n_generated": 10, "actual_nfe": 4,
            "resolved_protocol": True, "blocking_issue_ids": [],
            "model_seeds": [8925, 77395, 65457], "source_artifacts": []}
    cfg = prepare_configs.make_config(manifest, case)
    run.validate_config(cfg)
    assert cfg["data"]["split"]["seed"] == 19
    assert cfg["training"]["batch_rule"] == "global_torch"
    case["split"]["kind"] = "unrecognized"
    with pytest.raises(ValueError, match="Unrecognized vector split"):
        prepare_configs.make_config(manifest, case)
