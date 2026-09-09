"""Equation-level checks for t-Flow; no optimizer or benchmark is involved."""
import math

import pytest
import torch

from baselines.tflow_core import (
    TFlowSourceConfig,
    heun_sample,
    sample_student_t,
    tflow_loss,
    tflow_time_grid,
    tflow_vector_field,
)


@pytest.mark.parametrize("nu", [0, 1, 2, float("inf"), float("nan")])
def test_non_l2_source_is_rejected(nu):
    with pytest.raises(ValueError):
        TFlowSourceConfig(nu=nu)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        TFlowSourceConfig(scale=scale)


def test_chi_square_is_shared_over_the_entire_example(monkeypatch):
    gaussian = torch.arange(1, 13, dtype=torch.float64).reshape(2, 2, 3)

    class FixedChiSquare:
        def __init__(self, df):
            assert df.item() == 3

        def sample(self, sample_shape):
            # This fails if the implementation draws per coordinate/channel.
            assert sample_shape == (2,)
            return torch.tensor([3.0, 12.0], dtype=torch.float64)

    def fixed_gaussian(shape, *, device, dtype):
        assert shape == (2, 2, 3)
        return gaussian.to(device=device, dtype=dtype)

    monkeypatch.setattr(torch.distributions, "Chi2", FixedChiSquare)
    monkeypatch.setattr(torch, "randn", fixed_gaussian)
    actual = sample_student_t((2, 2, 3), TFlowSourceConfig(nu=3, scale=2))
    expected = gaussian * torch.tensor([2.0, 1.0]).reshape(2, 1, 1)
    torch.testing.assert_close(actual, expected.float())


def test_global_rng_state_replays_source_draws():
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(1701)
        saved = torch.random.get_rng_state()
        first = sample_student_t((7, 3), TFlowSourceConfig(nu=5, scale=0.7))
        torch.random.set_rng_state(saved)
        second = sample_student_t((7, 3), TFlowSourceConfig(nu=5, scale=0.7))
        assert torch.equal(first, second)
    finally:
        torch.random.set_rng_state(state)


def test_loss_is_noise_regression_with_the_published_interpolant():
    data = torch.tensor([[4.0, -8.0], [-2.0, 6.0]], dtype=torch.float64)
    noise = torch.tensor([[2.0, 0.0], [-4.0, 2.0]], dtype=torch.float64)
    times = torch.tensor([0.25, 0.75], dtype=torch.float64)
    captured = {}

    def predictor(x, t):
        captured["x"], captured["t"] = x, t
        return torch.zeros_like(x)

    per_example = tflow_loss(
        predictor, data, TFlowSourceConfig(), times=times, noise=noise,
        reduction="none",
    )
    torch.testing.assert_close(captured["x"], torch.tensor(
        [[2.5, -2.0], [-2.5, 5.0]], dtype=torch.float64
    ))
    assert torch.equal(captured["t"], times)
    torch.testing.assert_close(per_example, torch.tensor([4.0, 20.0], dtype=torch.float64))
    default_loss = tflow_loss(
        predictor, data, TFlowSourceConfig(), times=times, noise=noise
    )
    torch.testing.assert_close(default_loss, torch.tensor(12.0, dtype=torch.float64))


def test_loss_gradient_is_not_a_velocity_mse_gradient():
    data = torch.tensor([[1e30, -1e30], [-1e30, 1e30]])
    noise = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    offset = torch.nn.Parameter(torch.tensor(2.0))

    def predictor(x, t):
        return noise + offset

    loss = tflow_loss(
        predictor, data, TFlowSourceConfig(), times=torch.tensor([0.25, 0.75]),
        noise=noise,
    )
    loss.backward()
    # Huge data magnitudes do not become squared velocity labels.
    torch.testing.assert_close(loss, torch.tensor(8.0))
    torch.testing.assert_close(offset.grad, torch.tensor(8.0))


def test_noise_loss_equals_time_squared_weighted_velocity_loss():
    data = torch.tensor([[4.0, 6.0], [-2.0, 3.0]], dtype=torch.float64)
    noise = torch.tensor([[1.0, -2.0], [0.0, 2.0]], dtype=torch.float64)
    times = torch.tensor([0.25, 0.75], dtype=torch.float64)
    velocity = torch.tensor([[0.5, 1.0], [2.0, -1.0]], dtype=torch.float64)

    def predictor(x, t):
        return x - t[:, None] * velocity

    loss = tflow_loss(predictor, data, TFlowSourceConfig(), times=times, noise=noise)
    velocity_residual = (velocity - (data - noise)).square().sum(dim=1)
    torch.testing.assert_close(loss, (times.square() * velocity_residual).mean())
    assert not torch.isclose(loss, velocity_residual.mean())


def test_loss_accepts_exact_training_endpoints():
    data = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    noise = -data
    seen = []

    def predictor(x, t):
        seen.append(x.clone())
        return noise

    loss = tflow_loss(
        predictor, data, TFlowSourceConfig(), times=torch.tensor([0.0, 1.0]), noise=noise
    )
    assert loss.item() == 0
    assert torch.equal(seen[0][0], noise[0])
    assert torch.equal(seen[0][1], data[1])


@pytest.mark.parametrize("grid", ["linear", "power_time", "power_sigma"])
def test_grid_has_exact_endpoints_and_monotonic_model_times(grid):
    times = tflow_time_grid(t_min=0.01, n_steps=64, grid=grid, dtype=torch.float32)
    assert times.shape == (65,)
    assert times[0] == torch.tensor(0.01)
    assert times[-1].item() == 1.0
    assert bool((times[1:] > times[:-1]).all())
    if grid == "power_sigma":
        torch.testing.assert_close(times[-2], torch.tensor(0.99))


@pytest.mark.parametrize("grid", ["linear", "power_time", "power_sigma"])
def test_single_interval_has_two_endpoints(grid):
    times = tflow_time_grid(t_min=0.1, n_steps=1, grid=grid)
    torch.testing.assert_close(times, torch.tensor([0.1, 1.0], dtype=torch.float64))


def test_heun_matches_linear_ode_polynomial_and_counts_every_call():
    initial = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float64)
    original = initial.clone()
    calls = []

    def noise_predictor(x, t):
        calls.append((x.clone(), t.clone()))
        # Corresponds to dx/dt = x, permitting an independent Heun solution.
        return (1 - t[:, None]) * x

    result = heun_sample(noise_predictor, initial, t_min=0.25, n_steps=2, grid="linear")
    h = (1 - 0.25) / 2
    factor = (1 + h + h * h / 2) ** 2
    torch.testing.assert_close(result["samples"], initial * factor)
    assert result["nfe"] == len(calls) == 4
    assert result["n_steps"] == 2
    assert calls[0][1][0].item() == 0.25
    assert calls[-1][1][0].item() == 1.0
    assert torch.equal(initial, original)
    assert bool((result["samples"].norm(dim=1) > initial.norm(dim=1)).all())


def test_heun_constant_drift_reaches_the_data_endpoint():
    initial = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    drift = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    result = heun_sample(
        lambda x, t: x - t[:, None] * drift,
        initial, t_min=0.01, n_steps=8, grid="power_sigma",
    )
    torch.testing.assert_close(result["samples"], initial + 0.99 * drift)
    assert result["nfe"] == 16
    assert result["sigma_min"] == 0.01


@pytest.mark.parametrize("time", [0.0, -0.1, 1.1, float("nan")])
def test_vector_field_rejects_invalid_times(time):
    with pytest.raises((ValueError, FloatingPointError)):
        tflow_vector_field(lambda x, t: x, torch.ones(2, 3), time)


def test_nonfinite_predictions_fail_instead_of_being_repaired():
    with pytest.raises(FloatingPointError, match="noise prediction"):
        tflow_vector_field(
            lambda x, t: torch.full_like(x, math.nan), torch.ones(2, 3), 0.5
        )


def test_overflowed_field_fails_even_if_inputs_are_finite():
    x = torch.full((2, 3), torch.finfo(torch.float32).max)
    with pytest.raises(FloatingPointError, match="vector field"):
        tflow_vector_field(lambda x, t: -x, x, 0.5)


def test_sampler_rejects_nonfinite_initial_values_before_calling_model():
    def should_not_be_called(x, t):
        pytest.fail("model was called with an invalid initial state")

    with pytest.raises(FloatingPointError, match="initial state"):
        heun_sample(
            should_not_be_called, torch.tensor([[math.inf]]), t_min=0.01, n_steps=2
        )


@pytest.mark.parametrize("kwargs", [
    {"t_min": 0, "n_steps": 2},
    {"t_min": 1, "n_steps": 2},
    {"t_min": 0.01, "n_steps": 0},
    {"t_min": 0.01, "n_steps": 1.5},
    {"t_min": 0.01, "n_steps": 2, "rho": 0},
    {"t_min": 0.01, "n_steps": 2, "sigma_min": 0.999},
])
def test_invalid_sampler_grid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        tflow_time_grid(**kwargs)


def test_sampler_rejects_a_start_time_that_underflows_at_model_precision():
    with pytest.raises(ValueError, match="model precision"):
        heun_sample(
            lambda x, t: x, torch.ones(2, 3), t_min=1e-60, n_steps=2, grid="linear"
        )
