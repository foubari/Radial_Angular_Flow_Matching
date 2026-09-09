"""Lightweight distribution/analytic tests; no optimizer or benchmark model."""
import math

import numpy as np
import pytest
import torch
from scipy.stats import f as fisher_f, kstest

from baselines.tflow_core import TFlowSourceConfig, sample_student_t, heun_sample
from experiments.tflow.validation import source_candidates, selection_score


def test_joint_student_t_radius_has_f_law():
    torch.manual_seed(319)
    dim, nu, scale = 8, 5.0, 1.7
    sample = sample_student_t((12000, dim), TFlowSourceConfig(nu, scale), dtype=torch.float64)
    squared_radius = (sample.square().sum(1) / (dim * scale**2)).numpy()
    # This joint radial law detects independent per-coordinate chi-square scales.
    discrepancy = kstest(squared_radius, fisher_f(dim, nu).cdf).statistic
    assert discrepancy < 0.025


def test_source_grid_uses_train_median_not_target_df():
    train = torch.tensor([[3.,4.], [6.,8.], [9.,12.]], dtype=torch.float64)
    choices = source_candidates(train)
    assert len(choices) == 9
    assert {row["nu"] for row in choices} == {3.,5.,7.}
    for row in choices:
        median = row["scale"] * math.sqrt(2 * fisher_f.ppf(0.5, 2, row["nu"]))
        assert median == pytest.approx(10 * row["scale_multiplier"])


def test_selection_is_identical_for_identical_sets_and_replays_projection_rng():
    torch.manual_seed(541)
    data = torch.randn(30,4)
    state = torch.get_rng_state().clone()
    score = selection_score(data, data)
    assert score["selection_score"] == 0
    assert torch.equal(state, torch.get_rng_state())
    assert selection_score(data,data) == score


@pytest.mark.parametrize("grid", ["linear", "power_time", "power_sigma"])
@pytest.mark.parametrize("t_min", [0.001, 0.01])
def test_endpoint_truncation_against_gaussian_conditional_noise_oracle(grid, t_min):
    # Independent Gaussian endpoints provide an exact conditional noise predictor.
    # This tests the ODE numerics, not a t-Flow-trained network or empirical score.
    source_std, target_std = 1.3, 2.1
    initial = torch.tensor([[-1.,0.3], [0.5,1.2]], dtype=torch.float64)
    def oracle(x,t):
        variance = t.square()*target_std**2 + (1-t).square()*source_std**2
        return x * (((1-t)*source_std**2)/variance)[:,None]
    result = heun_sample(oracle, initial, t_min=t_min, n_steps=256, grid=grid)
    start_variance = t_min**2*target_std**2 + (1-t_min)**2*source_std**2
    exact = initial * target_std/math.sqrt(start_variance)
    assert result["nfe"] == 512
    assert torch.allclose(result["samples"], exact, rtol=0.001, atol=1e-5)
    # Explicit initialization truncation bias: source at t_min is an approximation.
    bias = abs(source_std/math.sqrt(start_variance)-1)
    assert bias < 1.02*t_min


def test_nonfinite_validation_is_a_failure():
    with pytest.raises(FloatingPointError):
        selection_score(torch.tensor([[float('inf'),0.],[0.,1.]]), torch.ones(2,2))
