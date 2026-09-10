"""Small correctness fixtures; run on an allocated cluster compute node.

No optimizer steps, benchmark data, tuning, or historical result replacement.
The downstream tests use the real backbone classes at small fixture sizes.
Full matched-size interface checks belong in the launcher's preflight.
"""
import copy
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from baselines.rafm_input_parameterization import (
    RadiusStatistics,
    ambient_velocity_callback,
    angular_target,
    build_input_model,
    fit_radius_statistics,
    radius_drift,
    unit_direction,
)
from rafm.flow_matching.loss import cfm_loss
from rafm.flow_matching.sampler import Sampler, _project_tangent
from rafm.models.mlp import MLP
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.sources.radial_empirical import RadialEmpiricalSource


MLP_CONFIG = {"kind": "mlp", "hidden_dim": 8, "n_layers": 3}


def stats(dim=3):
    train = torch.zeros(4, dim)
    train[:, 0] = torch.tensor([0.25, 1.0, 2.0, 4.0])
    return fit_radius_statistics(train, training_data_sha256="a" * 64,
                                 training_indices_sha256="b" * 64)


def model(arm, dim=3, cfg=None, seed=11):
    torch.manual_seed(seed)
    return build_input_model(MLP_CONFIG if cfg is None else cfg, dim, arm, stats(dim))


def test_statistics_fit_only_supplied_training_population_and_roundtrip():
    train = torch.tensor([[1.0, 0.0], [0.0, 2.0], [4.0, 0.0]])
    fitted = fit_radius_statistics(train, training_data_sha256="c" * 64,
                                   training_indices_sha256="d" * 64)
    expected = train.double().norm(dim=1).log()
    assert fitted.mean_log_radius == float(expected.mean())
    assert fitted.population_std_log_radius == float(expected.std(unbiased=False))
    assert fitted.n_training_samples == 3
    assert fitted.fit_split == "generator_training"
    assert RadiusStatistics.from_dict(fitted.to_dict()) == fitted
    # Extreme validation/test radii cannot affect the fit because there is no
    # API argument or implicit data loader through which those could enter.
    assert fitted.training_data_sha256 == "c" * 64
    assert fitted.training_indices_sha256 == "d" * 64


def test_zero_constant_and_subnormal_statistics_are_finite_and_explicit():
    fitted = fit_radius_statistics(torch.zeros(3, 2), training_data_sha256="a" * 64,
                                   training_indices_sha256="b" * 64)
    assert fitted.zero_radius_count == 3
    assert fitted.below_radius_floor_count == 3
    assert fitted.population_std_log_radius == 0
    assert fitted.standardization_scale == fitted.std_floor
    predictor = build_input_model(MLP_CONFIG, 2, "B", fitted)
    assert torch.isfinite(predictor(torch.zeros(2, 2), torch.zeros(2))).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_fit_rejects_nonfinite_training_data(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        fit_radius_statistics(torch.tensor([[bad, 1.0]]), training_data_sha256="a" * 64,
                              training_indices_sha256="b" * 64)


def test_statistics_reject_test_split_and_untracked_identity():
    bad = stats().to_dict()
    bad["fit_split"] = "test"
    with pytest.raises(ValueError, match="generator training"):
        RadiusStatistics.from_dict(bad)
    with pytest.raises(ValueError, match="SHA-256"):
        fit_radius_statistics(torch.ones(2, 3), training_data_sha256="unknown",
                              training_indices_sha256="b" * 64)


def test_unit_direction_has_no_sub_epsilon_radius_channel():
    direction = torch.tensor([[1.0, -2.0, 3.0]])
    scaled = torch.cat([direction * scale for scale in (1e-35, 1e-12, 1.0, 1e30)])
    unit = unit_direction(scaled)
    assert torch.allclose(unit, unit[0].expand_as(unit), atol=1e-7, rtol=1e-7)
    assert torch.allclose(unit.norm(dim=1), torch.ones(4), atol=1e-7, rtol=1e-7)
    assert torch.equal(unit_direction(torch.zeros(1, 3)), torch.zeros(1, 3))


def test_a_exactly_preserves_original_backbone_output_initialization_and_rng():
    torch.manual_seed(17)
    original = MLP(3, hidden_dim=8, n_layers=3)
    original_rng = torch.get_rng_state()
    candidate = model("A", seed=17)
    assert torch.equal(torch.get_rng_state(), original_rng)
    for key, value in original.state_dict().items():
        assert torch.equal(candidate.backbone.state_dict()[key], value)
    x, t = torch.randn(4, 3), torch.rand(4)
    assert torch.equal(candidate(x, t), original(x, t))
    assert candidate.parameter_report()["conditioning_overhead_parameters"] == 0


def test_b_c_identical_parameters_and_original_common_weights_rng():
    original, conditioned, unconditioned = model("A"), model("B"), model("C")
    for key, value in conditioned.state_dict().items():
        assert torch.equal(value, unconditioned.state_dict()[key]), key
    a_first = original.backbone.net[0]
    b_first = conditioned.backbone.net[0]
    assert torch.equal(a_first.weight[:, :3], b_first.weight[:, :3])
    assert torch.equal(a_first.weight[:, -1], b_first.weight[:, -1])
    assert torch.equal(a_first.bias, b_first.bias)
    for key, value in original.backbone.state_dict().items():
        if key != "net.0.weight":
            assert torch.equal(value, conditioned.backbone.state_dict()[key]), key
    assert conditioned.parameter_report()["conditioning_overhead_parameters"] == 8
    assert conditioned.parameter_report()["total_parameters"] == unconditioned.parameter_report()["total_parameters"]
    model("A", seed=5)
    a_rng = torch.get_rng_state()
    model("B", seed=5)
    assert torch.equal(torch.get_rng_state(), a_rng)
    model("C", seed=5)
    assert torch.equal(torch.get_rng_state(), a_rng)


def test_c_predictions_are_radius_invariant_including_tiny_positive_radii():
    candidate = model("C").eval()
    x = torch.tensor([[1.0, -2.0, 0.5]])
    t = torch.tensor([0.4])
    output = candidate(x, t)
    for scale in (1e-30, 1e-9, 0.25, 5.0, 1e20):
        assert torch.allclose(candidate(scale * x, t), output, atol=1e-7, rtol=1e-6)
        assert torch.equal(candidate.condition(scale * x), torch.zeros(1, 1))


def test_b_radius_condition_changes_and_gradient_flows_to_scalar_weights():
    candidate = model("B")
    x = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.5, -0.2]])
    t = torch.tensor([0.4, 0.8])
    assert not torch.allclose(candidate.condition(x), candidate.condition(5 * x))
    assert not torch.allclose(candidate(x, t), candidate(5 * x, t))
    candidate(x, t).square().sum().backward()
    gradient = candidate.backbone.net[0].weight.grad[:, 3]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_compiled_vector_training_does_not_specialize_on_network_counter(arm):
    candidate = model(arm)
    graphs = []

    def record_graph(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    # A tracing backend exercises real Dynamo guards without invoking a slow
    # optimizing compiler. Repeated calls must reuse one full training graph.
    compiled = torch.compile(candidate, backend=record_graph, fullgraph=True)
    x, t = torch.randn(4, 3), torch.rand(4)
    for _ in range(4):
        assert torch.isfinite(compiled(x, t)).all()
    assert len(graphs) == 1
    assert candidate.network_calls == 0
    candidate(x, t)
    assert candidate.network_calls == 1


def test_original_cfm_target_loss_and_rng_are_exactly_retained_for_a():
    candidate = model("A")
    original = copy.deepcopy(candidate.backbone)
    x1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]])
    source = RadialEmpiricalSource(mode="ecdf").fit(x1)
    path = SphericalGeodesicPath()
    torch.manual_seed(29)
    expected = cfm_loss(original, path, source, x1, angular=True)
    rng = torch.get_rng_state()
    torch.manual_seed(29)
    actual = cfm_loss(candidate, path, source, x1, angular=True)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)


def test_angular_target_is_unchanged_with_matched_radius_spherical_path():
    x1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    x0 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    t = torch.tensor([0.25, 0.75])
    path = SphericalGeodesicPath()
    xt = path.sample_path(x0, x1, t)
    ut = path.conditional_vector_field(x0, x1, t)
    expected = ut / xt.norm(dim=1, keepdim=True).clamp_min(1e-8)
    assert torch.equal(angular_target(xt, ut), expected)
    assert torch.allclose(xt.norm(dim=1), x1.norm(dim=1), atol=1e-6)
    assert torch.allclose((xt * expected).sum(dim=1), torch.zeros(2), atol=1e-6)
    # No input-arm argument exists: all three arms consume exactly this target.


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_original_ambient_projection_order_and_guard_are_preserved(arm):
    candidate = model(arm)
    x = torch.tensor([[1.0, -1.0, 0.5], [1e-5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    t = torch.tensor([0.1, 0.2, 0.3])
    expected = _project_tangent(x.norm(dim=1, keepdim=True).clamp_min(1e-8) * candidate(x, t), x)
    assert torch.equal(ambient_velocity_callback(candidate)(x, t), expected)
    expected_audio = _project_tangent(x.norm(dim=1, keepdim=True) * candidate(x, t), x)
    assert torch.equal(ambient_velocity_callback(candidate, radius_floor=0)(x, t), expected_audio)


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_existing_rk4_finite_sampling_counts_actual_calls_and_reports_drift(arm):
    candidate = model(arm).eval()
    source = RadialEmpiricalSource(mode="ecdf").fit(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    sampler = Sampler(candidate, source, {"solver": "rk4", "nfe": 3, "angular": True,
                                         "path": "spherical_geodesic", "device": "cpu"})
    x0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    start_calls = candidate.network_calls
    samples = sampler._rk4(x0.clone(), 3)
    assert candidate.network_calls - start_calls == 12
    assert torch.isfinite(samples).all()
    drift = radius_drift(x0, samples)
    assert drift["mean_absolute"] >= 0
    result = sampler.sample(2, 3)
    assert result["nfe"] == 12
    assert candidate.network_calls - start_calls == 24


def test_checkpoint_roundtrip_rejects_arm_statistics_tensor_and_source_mismatch():
    candidate = model("B")
    payload = copy.deepcopy(candidate.checkpoint_payload())
    restored = model("B", seed=27)
    restored.load_checkpoint_payload(payload)
    x, t = torch.randn(3, 3), torch.rand(3)
    assert torch.equal(candidate(x, t), restored(x, t))
    with pytest.raises(ValueError, match="identity mismatch"):
        model("C").load_checkpoint_payload(payload)
    bad = copy.deepcopy(payload)
    bad["metadata"]["radius_statistics"]["training_indices_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        restored.load_checkpoint_payload(bad)
    bad = copy.deepcopy(payload)
    bad["state_dict"]["radius_mean_log"] += 1
    with pytest.raises(ValueError, match="statistics buffer"):
        restored.load_checkpoint_payload(bad)
    bad = copy.deepcopy(payload)
    bad["state_dict"]["backbone.net.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        restored.load_checkpoint_payload(bad)


def test_only_a_can_strictly_load_original_backbone_checkpoint():
    original = model("A", seed=13)
    other = model("A", seed=23)
    other.load_original_backbone_state(original.backbone.state_dict())
    for key, value in original.backbone.state_dict().items():
        assert torch.equal(value, other.backbone.state_dict()[key])
    with pytest.raises(ValueError, match="only load into arm A"):
        model("B").load_original_backbone_state(original.backbone.state_dict())


def downstream_configs():
    return {
        "audio_unet": {"kind": "audio_unet", "event_shape": [2, 9, 7], "ch": 8,
                       "mult": [1, 2], "cond": 16, "attn_from": 1, "num_classes": 10},
        "image_sit": {"kind": "image_sit", "event_shape": [32, 8, 8], "hidden": 24,
                      "depth": 1, "heads": 3, "num_classes": 10, "class_dropout": 0.1,
                      "source_dir": os.environ.get("RAFM_SIT_SOURCE", "/mnt/vast01/users/fouad.oubari/references/SiT-tflow")},
    }


@pytest.mark.parametrize("kind", ["audio_unet", "image_sit"])
def test_real_downstream_interfaces_b_c_equal_modules_conditioning_and_gradient(kind):
    cfg = downstream_configs()[kind]
    if kind == "image_sit" and not Path(cfg["source_dir"], "models.py").exists():
        pytest.skip("pinned SiT source absent; full preflight must report this missing interface")
    dim = int(torch.tensor(cfg["event_shape"]).prod())
    b, c = model("B", dim, cfg), model("C", dim, cfg)
    b.eval()
    c.eval()
    for key, value in b.state_dict().items():
        assert torch.equal(value, c.state_dict()[key]), key
    assert b.parameter_report()["total_parameters"] == c.parameter_report()["total_parameters"]
    condition_dim = 16 if kind == "audio_unet" else 24
    assert b.parameter_report()["conditioning_overhead_parameters"] == 32 + 17 * condition_dim
    # Original final layers are zero initialized. Reveal the existing interior
    # computational graph with a fixture-only final weight, without training.
    b_final = b.backbone.out if kind == "audio_unet" else b.backbone.final_layer.linear
    c_final = c.backbone.out if kind == "audio_unet" else c.backbone.final_layer.linear
    torch.manual_seed(41)
    nn.init.normal_(b_final.weight, std=0.01)
    c_final.weight.data.copy_(b_final.weight.data)
    if kind == "image_sit":
        # SiT also starts with zero conditioning gates. Initialize only the
        # final modulation in this test so a radius derivative is observable.
        last_b = b.backbone.final_layer.adaLN_modulation[-1]
        last_c = c.backbone.final_layer.adaLN_modulation[-1]
        nn.init.normal_(last_b.weight, std=0.01)
        last_c.weight.data.copy_(last_b.weight.data)
    x, t, labels = torch.randn(2, dim), torch.tensor([0.2, 0.6]), torch.tensor([1, 7])
    hooks_before = len(b._time_embedding_module()._forward_hooks)
    prediction = b(x, t, labels)
    assert prediction.shape == x.shape and torch.isfinite(prediction).all()
    assert len(b._time_embedding_module()._forward_hooks) == hooks_before
    prediction.square().sum().backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in b.radius_embedding.parameters())
    with torch.no_grad():
        assert torch.allclose(c(x, t, labels), c(3 * x, t, labels), atol=2e-6, rtol=2e-5)
        assert not torch.allclose(b(x, t, labels), b(3 * x, t, labels), atol=1e-7, rtol=1e-7)
    assert len(b._time_embedding_module()._forward_hooks) == hooks_before


def test_downstream_temporary_condition_hook_is_removed_on_failure(monkeypatch):
    cfg = downstream_configs()["audio_unet"]
    candidate = model("B", 126, cfg)
    module = candidate._time_embedding_module()
    before = len(module._forward_hooks)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional fixture failure")

    monkeypatch.setattr(candidate.backbone, "forward", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        candidate(torch.ones(2, 126), torch.zeros(2), torch.tensor([0, 1]))
    assert len(module._forward_hooks) == before
