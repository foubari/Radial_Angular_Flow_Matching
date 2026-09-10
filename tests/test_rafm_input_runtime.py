"""RAFM-Ang runner contracts against original path, batching and RK4 code.

Small synthetic fixtures only; the parent runs these on cluster compute.
There are no optimizer steps, benchmark data loads, or tuning decisions.
"""
import copy

import pytest
import torch
from torch import nn

from experiments.rafm_inputs import run
from experiments.tflow.data import CachedData
from rafm.flow_matching.loss import cfm_loss
from rafm.flow_matching.sampler import Sampler
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.utils.seeds import set_all_seeds


def values_fixture(n=12, dim=3):
    generator = torch.Generator().manual_seed(285)
    values = torch.randn(n, dim, generator=generator)
    return values / values.norm(dim=1, keepdim=True) * torch.linspace(0.2, 3.0, n)[:, None]


class CapturePath(SphericalGeodesicPath):
    def sample_path(self, initial, target, times):
        self.initial = initial.clone()
        self.times = times.clone()
        self.state = super().sample_path(initial, target, times)
        return self.state

    def conditional_vector_field(self, initial, target, times):
        self.velocity = super().conditional_vector_field(initial, target, times)
        return self.velocity


class CaptureZero(nn.Module):
    def forward(self, x, t):
        self.x, self.t = x.clone(), t.clone()
        return torch.zeros_like(x)


def test_vector_training_pair_matches_original_cfm_target_and_rng_exactly():
    values = values_fixture()
    source = RadialEmpiricalSource(mode="ecdf").fit(values)
    original_path, original_model = CapturePath(), CaptureZero()
    torch.manual_seed(18731)
    original_loss = cfm_loss(original_model, original_path, source, values, angular=True)
    original_rng = torch.get_rng_state()
    expected_target = original_path.velocity / original_path.state.norm(dim=1, keepdim=True).clamp_min(1e-8)
    torch.manual_seed(18731)
    candidate_path = CapturePath()
    state, times, target = run.training_pair(values, "vector", candidate_path)
    assert torch.equal(state, original_model.x)
    assert torch.equal(times, original_model.t)
    assert torch.equal(candidate_path.initial, original_path.initial)
    assert torch.equal(target, expected_target)
    assert torch.equal(target.square().sum(dim=1).mean(), original_loss)
    assert torch.equal(torch.get_rng_state(), original_rng)


@pytest.mark.parametrize("kind", ["audio", "image"])
def test_downstream_training_pair_matches_original_source_before_time_rng(kind):
    values = values_fixture()
    original_path = SphericalGeodesicPath()
    torch.manual_seed(63107)
    # Literal order and arithmetic in audio_flow.py / dit_train_sit.py.
    radius = values.norm(dim=1, keepdim=True)
    normal = torch.randn(len(values), values.shape[1], device=values.device)
    initial = radius * normal / normal.norm(dim=1, keepdim=True)
    times = torch.rand(len(values), device=values.device)
    state = original_path.sample_path(initial, values, times)
    velocity = original_path.conditional_vector_field(initial, values, times)
    target = velocity / state.norm(dim=1, keepdim=True).clamp_min(1e-8)
    original_rng = torch.get_rng_state()
    torch.manual_seed(63107)
    recorded = CapturePath()
    actual_state, actual_times, actual_target = run.training_pair(values, kind, recorded)
    assert torch.equal(recorded.initial, initial)
    assert torch.equal(actual_times, times)
    assert torch.equal(actual_state, state)
    assert torch.equal(actual_target, target)
    assert torch.equal(torch.get_rng_state(), original_rng)


def test_nonfinite_training_path_is_an_explicit_failure():
    values = values_fixture()

    class BrokenPath(SphericalGeodesicPath):
        def conditional_vector_field(self, initial, target, times):
            return torch.full_like(target, float("nan"))

    with pytest.raises(FloatingPointError, match="Nonfinite spherical path"):
        run.training_pair(values, "vector", BrokenPath())


def test_vector_batch_uses_original_global_torch_rng_without_extra_draws():
    values = values_fixture(25)
    settings = {"batch_rule": "global_torch", "batch_size": 9}
    torch.manual_seed(2871)
    index = torch.randint(len(values), (9,), device=values.device)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(2871)
    actual, labels = run.batch(values, None, settings, "vector", 8925, 21)
    assert torch.equal(actual, values[index])
    assert labels is None
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("kind", ["audio", "image"])
def test_downstream_batch_and_audio_class_dropout_match_original_step_seeds(kind):
    values = values_fixture(25)
    labels = torch.arange(25) % 10
    preserved_labels = labels.clone()
    settings = {"batch_rule": "step_seeded", "batch_size": 12, "class_dropout": 0.4}
    seed, step = 8925, 41
    generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
    index = torch.randint(len(values), (12,), generator=generator)
    expected_labels = labels[index].clone()
    if kind == "audio":
        dropout = torch.rand(12, generator=torch.Generator().manual_seed(seed * 7 + step))
        expected_labels[dropout < settings["class_dropout"]] = 10
    torch.manual_seed(932)
    original_rng = torch.get_rng_state()
    actual, actual_labels = run.batch(values, labels, settings, kind, seed, step)
    assert torch.equal(actual, values[index])
    assert torch.equal(actual_labels, expected_labels)
    assert torch.equal(labels, preserved_labels)
    assert torch.equal(torch.get_rng_state(), original_rng)
    # Explicit per-step generators preserve batching despite unrelated global RNG.
    torch.randn(31)
    repeated, repeated_labels = run.batch(values, labels, settings, kind, seed, step)
    assert torch.equal(repeated, actual)
    assert torch.equal(repeated_labels, actual_labels)


class FixtureAngularModel(nn.Module):
    """Tiny smooth angular predictor for sampler arithmetic, no fitted weights."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.03))
        self.calls = 0

    def forward(self, x, t, labels=None):
        self.calls += 1
        result = self.scale * (torch.sin(x) + t[:, None])
        if labels is not None:
            result = result + 0.001 * labels[:, None]
        return result


class BoundLabels:
    def __init__(self, model, labels):
        self.model, self.labels = model, labels

    def __call__(self, x, t):
        return self.model(x, t, self.labels)


class OriginalAudioInterface(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, labels):
        return self.model(x.reshape(len(x), -1), t, labels).reshape_as(x)


def cached_fixture(dim):
    values = values_fixture(20, dim)
    return CachedData(values, None,
        {"train": torch.arange(12), "val": torch.arange(12, 16), "test": torch.arange(16, 20)},
        torch.zeros(dim))


@pytest.mark.parametrize("kind,dim", [("vector", 3), ("audio", 16254), ("image", 2048)])
def test_runner_sampler_matches_original_ambient_rk4_outputs_rng_and_actual_calls(kind, dim):
    data = cached_fixture(dim)
    cfg = {"kind": kind, "evaluation": {"model_evaluations": 8, "sample_batch_size": 7}}
    candidate = FixtureAngularModel().eval()
    actual = run.sample(cfg, candidate, data, n=20, seed=407)
    actual_rng = torch.get_rng_state()
    assert candidate.calls == 24
    assert actual["nfe"] == 8
    assert actual["n_batches"] == 3
    assert actual["model_calls_total"] == 24
    assert actual["state_renormalization"] is False
    assert actual["projection_r_min"] == 1e-3
    assert actual["radius_multiplier_floor"] == (0.0 if kind == "audio" else 1e-8)
    assert torch.isfinite(actual["samples"]).all()

    set_all_seeds(407)
    source = RadialEmpiricalSource(mode="ecdf").fit(data.split("train"))
    original = FixtureAngularModel().eval()
    if kind == "vector":
        initial = source.sample(20, dim, device="cpu")
        labels = None
    else:
        labels = torch.arange(10).repeat_interleave(2)
        radii = source.sample(20, dim).norm(dim=1, keepdim=True)
        normal = torch.randn(20, dim)
        initial = radii * normal / normal.norm(dim=1, keepdim=True)
    expected = []
    with torch.no_grad():
        for start in range(0, 20, 7):
            part = initial[start:start + 7]
            if kind == "audio":
                from experiments.poc_audio.audio_eval import rk4
                output = rk4(OriginalAudioInterface(original), part, labels[start:start + 7],
                             spherical=True, nfe=2, angular=True)
            else:
                bound = original if labels is None else BoundLabels(original, labels[start:start + 7])
                # Image and vector use precisely this angular multiplier,
                # projection guard, RK4 stage arithmetic and time construction.
                sampler = Sampler(bound, source, {"device": "cpu", "angular": True,
                                                  "path": "spherical_geodesic"})
                output = sampler._rk4(part, 2)
            expected.append(output)
    assert original.calls == 24
    assert torch.equal(actual["samples"], torch.cat(expected))
    assert torch.equal(actual["initial_radii"], initial.norm(dim=1))
    assert torch.equal(torch.get_rng_state(), actual_rng)
    if labels is None:
        assert actual["labels"] is None
    else:
        assert torch.equal(actual["labels"], labels)


def test_conditional_sampling_rejects_unbalanced_count_and_preserves_nonfinite_failure():
    data = cached_fixture(3)
    cfg = {"kind": "audio", "evaluation": {"model_evaluations": 8, "sample_batch_size": 7}}
    with pytest.raises(ValueError, match="class balanced"):
        run.sample(cfg, FixtureAngularModel(), data, n=21, seed=17)

    class NonfiniteModel(FixtureAngularModel):
        def forward(self, x, t, labels=None):
            return torch.full_like(x, float("nan"))

    cfg["kind"] = "vector"
    with pytest.raises(FloatingPointError, match="Nonfinite samples"):
        run.sample(cfg, NonfiniteModel(), data, n=20, seed=17)


def test_statistics_hashes_exact_selected_training_values_and_indices():
    data = cached_fixture(3)
    fitted = run.statistics(data)
    assert fitted.n_training_samples == 12
    assert fitted.training_data_sha256 == run.common.tensor_fingerprint(data.split("train"))["sha256"]
    assert fitted.training_indices_sha256 == run.common.tensor_fingerprint(data.indices["train"])["sha256"]
    original = fitted.to_dict()
    data.values[data.indices["val"]] *= 1000
    data.values[data.indices["test"]] *= 10000
    assert run.statistics(data).to_dict() == original
