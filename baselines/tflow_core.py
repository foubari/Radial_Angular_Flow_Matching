"""Direct noise-prediction t-Flow, with explicit numerical adaptations.

Reference: Pandey et al., arXiv:2410.14171v2, Appendix B, Eqs. 164--166.
See docs/tflow_method.md for conventions, provenance, and endpoint limitations.
This module does not launch training or change a model's train/eval mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor


NoiseModel = Callable[[Tensor, Tensor], Tensor]
TimeGrid = Literal["linear", "power_time", "power_sigma"]


def _finite_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class TFlowSourceConfig:
    """Student-t scale matrix is scale**2 * I; scale is not its stddev.

    nu > 2 ensures the noise label has a finite second moment. The target
    distribution need not have finite variance or the same degrees of freedom.
    """

    nu: float = 3.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        nu = _finite_real("nu", self.nu)
        scale = _finite_real("scale", self.scale)
        if nu <= 2:
            raise ValueError("nu must exceed 2 for the noise-prediction L2 objective")
        if scale <= 0:
            raise ValueError("scale must be positive")
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "scale", scale)


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains a non-finite value")


def _batch(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim < 2 or any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must have nonempty shape [batch, ...]")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    _finite(name, value)


def _same_tensor(name: str, value: Tensor, reference: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} shape must equal {tuple(reference.shape)}")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must match the input device and dtype")
    _finite(name, value)


def _broadcast_time(times: Tensor, x: Tensor) -> Tensor:
    return times.reshape(x.shape[0], *([1] * (x.ndim - 1)))


def _times(times: Tensor | float, x: Tensor, *, positive: bool) -> Tensor:
    result = torch.as_tensor(times, dtype=x.dtype, device=x.device)
    if result.ndim == 0:
        result = result.expand(x.shape[0])
    elif result.shape == (x.shape[0], 1):
        result = result[:, 0]
    elif result.shape != (x.shape[0],):
        raise ValueError("times must be scalar, [batch], or [batch, 1]")
    _finite("times", result)
    invalid_lower = result <= 0 if positive else result < 0
    if bool((invalid_lower | (result > 1)).any().item()):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"times must lie in {interval}")
    return result


def sample_student_t(
    shape: Sequence[int],
    source: TFlowSourceConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample one multivariate Student-t vector per example.

    All non-batch coordinates share ONE chi-square variable per example.
    Uses the caller's global PyTorch RNG (including its device RNG); seed and
    save/restore that RNG in the runner. No private gamma primitive is used.
    The mixture is computed in float64 and checked after casting to dtype.
    """
    shape = tuple(shape)
    if len(shape) < 2 or any(
        isinstance(s, bool) or not isinstance(s, Integral) or s <= 0 for s in shape
    ):
        raise ValueError("shape must be positive integers [batch, ...]")
    if not isinstance(source, TFlowSourceConfig):
        raise TypeError("source must be a TFlowSourceConfig")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be float32 or float64")
    degrees = torch.tensor(source.nu, device=device, dtype=torch.float64)
    chi_square = torch.distributions.Chi2(degrees).sample((shape[0],))
    _finite("chi-square sample", chi_square)
    if bool((chi_square <= 0).any().item()):
        raise FloatingPointError("chi-square sample must be strictly positive")
    kappa = (chi_square / source.nu).reshape(shape[0], *([1] * (len(shape) - 1)))
    _finite("chi-square / nu", kappa)
    if bool((kappa <= 0).any().item()):
        raise FloatingPointError("chi-square / nu underflowed to zero")
    gaussian = torch.randn(shape, device=device, dtype=torch.float64)
    noise = (source.scale * gaussian / torch.sqrt(kappa)).to(dtype=dtype)
    _finite("Student-t noise", noise)
    return noise


def tflow_loss(
    model: NoiseModel,
    x_data: Tensor,
    source: TFlowSourceConfig,
    *,
    times: Tensor | float | None = None,
    noise: Tensor | None = None,
    reduction: Literal["batch_mean", "mean", "sum", "none"] = "batch_mean",
) -> Tensor:
    """Regress scaled Student-t noise directly along a straight interpolant.

    model(x_t, t) receives times of shape [batch]. There are no t-dependent
    weights or velocity targets. Default 'batch_mean' averages squared Euclidean
    norms over examples, as in the paper. 'none' returns those per-example norms;
    'sum' sums all squared residuals. Explicit 'mean' also averages coordinates
    and changes the objective scale by 1/d (including its optimizer interaction).
    Explicit times/noise support deterministic checks and paired comparisons.
    """
    _batch("x_data", x_data)
    if not isinstance(source, TFlowSourceConfig):
        raise TypeError("source must be a TFlowSourceConfig")
    if reduction not in ("batch_mean", "mean", "sum", "none"):
        raise ValueError("reduction must be batch_mean, mean, sum, or none")
    if times is None:
        times = torch.rand(x_data.shape[0], device=x_data.device, dtype=x_data.dtype)
    times = _times(times, x_data, positive=False)
    if noise is None:
        noise = sample_student_t(
            x_data.shape, source, device=x_data.device, dtype=x_data.dtype
        )
    _same_tensor("noise", noise, x_data)
    t = _broadcast_time(times, x_data)
    x_t = t * x_data + (1 - t) * noise
    _finite("interpolant", x_t)
    prediction = model(x_t, times)
    _same_tensor("noise prediction", prediction, x_data)
    squared_residual = (prediction - noise).square()
    _finite("squared noise residual", squared_residual)
    loss = squared_residual.flatten(start_dim=1).sum(dim=1)
    if reduction == "batch_mean":
        loss = loss.mean()
    elif reduction == "mean":
        loss = squared_residual.mean()
    elif reduction == "sum":
        loss = loss.sum()
    _finite("noise-prediction loss", loss)
    return loss


def tflow_vector_field(model: NoiseModel, x: Tensor, times: Tensor | float) -> Tensor:
    """Convert a noise prediction to dx/dt = (x - predicted_noise) / t.

    Zero time is rejected, not clamped. No radius or tangent projection occurs.
    """
    _batch("state", x)
    times = _times(times, x, positive=True)
    predicted_noise = model(x, times)
    _same_tensor("noise prediction", predicted_noise, x)
    field = (x - predicted_noise) / _broadcast_time(times, x)
    _finite("t-Flow vector field", field)
    return field


def tflow_time_grid(
    *,
    t_min: float,
    n_steps: int,
    rho: float = 7.0,
    grid: TimeGrid = "power_sigma",
    sigma_min: float = 0.01,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Build an increasing endpoint-preserving grid from t_min to exactly 1.

    power_time: t(u) = [t_min**(1/rho) + u*(1-t_min**(1/rho))]**rho.
    power_sigma: interpolate sigma**(1/rho) from (1-t_min)**(1/rho)
    to sigma_min**(1/rho), then append t=1 as an explicit final interval.
    Both grid orientations and the positive start are benchmark adaptations.
    An unrepresentable or non-increasing grid is rejected without deduplication.
    """
    t_min = _finite_real("t_min", t_min)
    rho = _finite_real("rho", rho)
    sigma_min = _finite_real("sigma_min", sigma_min)
    if not 0 < t_min < 1:
        raise ValueError("t_min must lie strictly between zero and one")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if grid not in ("linear", "power_time", "power_sigma"):
        raise ValueError("grid must be linear, power_time, or power_sigma")
    if not 0 < sigma_min < 1:
        raise ValueError("sigma_min must lie strictly between zero and one")
    if grid == "power_sigma" and sigma_min >= 1 - t_min:
        raise ValueError("sigma_min must be smaller than 1 - t_min")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("grid dtype must be float32 or float64")
    u = torch.linspace(0, 1, n_steps + 1, device=device, dtype=dtype)
    if grid == "linear":
        times = t_min + (1 - t_min) * u
    elif grid == "power_time":
        root = t_min ** (1 / rho)
        times = (root + (1 - root) * u).pow(rho)
    elif n_steps == 1:
        # A single interval has no interior sigma knot.
        times = torch.tensor([t_min, 1.0], device=device, dtype=dtype)
    else:
        u_sigma = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
        high = (1 - t_min) ** (1 / rho)
        low = sigma_min ** (1 / rho)
        sigma = (high + (low - high) * u_sigma).pow(rho)
        sigma[-1] = sigma_min
        times = torch.cat((1 - sigma, torch.ones(1, device=device, dtype=dtype)))
    times[0] = t_min
    times[-1] = 1
    _finite("time grid", times)
    if bool(((times <= 0) | (times > 1)).any().item()):
        raise ValueError("time-grid endpoints cannot be represented in this dtype")
    if not bool((times[1:] > times[:-1]).all().item()):
        raise ValueError("time grid is not strictly increasing at this precision")
    return times


@torch.no_grad()
def heun_sample(
    model: NoiseModel,
    x_initial: Tensor,
    *,
    t_min: float,
    n_steps: int,
    rho: float = 7.0,
    grid: TimeGrid = "power_sigma",
    sigma_min: float = 0.01,
) -> dict:
    """Integrate using two actual model calls per interval, including the last.

    The caller normally supplies a fresh sample_student_t draw and calls
    model.eval() beforehand. Treating that draw as a state at t_min approximates
    the omitted [0, t_min] trajectory; this routine cannot remove that bias.
    Input storage and model mode are left intact. Samples retain device/dtype.
    No clipping, resampling, projection, or final denoising step is performed.
    """
    _batch("initial state", x_initial)
    # Float64 grid arithmetic avoids needless cancellation close to endpoints.
    times = tflow_time_grid(
        t_min=t_min, n_steps=n_steps, rho=rho, grid=grid, sigma_min=sigma_min,
        device=x_initial.device, dtype=torch.float64,
    )
    # The model sees input-precision times: validate that this grid also remains
    # strictly increasing there, rather than silently repeating rounded times.
    model_times = times.to(dtype=x_initial.dtype)
    if bool((model_times <= 0).any().item()) or not bool(
        (model_times[1:] > model_times[:-1]).all().item()
    ):
        raise ValueError("time grid is not positive and increasing at model precision")
    x = x_initial.clone()
    nfe = 0
    for i in range(n_steps):
        dt = (times[i + 1] - times[i]).to(dtype=x.dtype)
        d_first = tflow_vector_field(model, x, model_times[i])
        nfe += 1
        predictor = x + dt * d_first
        _finite("Heun predictor", predictor)
        d_second = tflow_vector_field(model, predictor, model_times[i + 1])
        nfe += 1
        x = x + dt * (0.5 * d_first + 0.5 * d_second)
        _finite("Heun corrected state", x)
    return {
        "samples": x,
        "nfe": nfe,
        "n_steps": int(n_steps),
        "t_min": float(t_min),
        "t_max": 1.0,
        "grid": grid,
        "rho": float(rho),
        "sigma_min": float(sigma_min) if grid == "power_sigma" else None,
        "time_grid": times,
    }
