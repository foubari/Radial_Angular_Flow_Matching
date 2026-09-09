"""Bounded training/validation-only source selection, separate from test metrics."""
from __future__ import annotations

import math
from numbers import Integral
import numpy as np
import torch
from scipy.stats import f as fisher_f, ks_2samp

TUNING_SEED = 46021
DEGREES_OF_FREEDOM = (3.0, 5.0, 7.0)
SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)


def _vectors(values, name):
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError(f"{name} must contain finite floating-point values")
    # Finite heavy-tailed float32 rows can have overflowing float32 squared
    # norms/projections. Float64 arithmetic leaves their coordinates unchanged.
    return values.detach().to(device="cpu", dtype=torch.float64)


def _radii(values):
    scale = values.abs().amax(dim=1)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    result = (values / divisor[:, None]).norm(dim=1) * scale
    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("Validation radius is not representable in float64")
    return result


def source_candidates(train):
    """Median-radius calibration uses only training rows, not target df/moments.

    For unit-scale multivariate t, ||N||^2/d ~ F(d,nu). This keeps scale
    selection meaningful even when the target has infinite second moments.
    No input normalization or coordinate transformation is performed.
    """
    train = _vectors(train, "training data")
    radius = float(torch.quantile(_radii(train), 0.5))
    if radius <= 0:
        raise ValueError("Median training radius must be positive")
    dim = train.shape[1]
    choices = []
    for nu in DEGREES_OF_FREEDOM:
        reference_scale = radius / math.sqrt(dim * float(fisher_f.ppf(0.5, dim, nu)))
        for multiplier in SCALE_MULTIPLIERS:
            if not math.isfinite(reference_scale * multiplier) or reference_scale * multiplier <= 0:
                raise FloatingPointError("Calibrated source scale is not finite and positive")
            choices.append({"nu": nu, "scale": reference_scale * multiplier,
                            "scale_multiplier": multiplier, "reference_scale": reference_scale,
                            "training_median_radius": radius})
    return choices


def selection_score(samples, validation, *, projection_seed=61719, n_projections=64):
    """Equal-weight radius KS and mean projected KS; bounded CDF criteria.

    Works with unequal sample counts without bootstrap resizing. This is a
    tuning criterion, not a replacement for any reported paper metric.
    Caller must supply validation rows, never the held-out test split.
    """
    x, y = _vectors(samples, "generated samples"), _vectors(validation, "validation data")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Selection tensors must share their vector dimension")
    if min(len(samples), len(validation)) < 2:
        raise ValueError("At least two samples per selection tensor are required")
    if isinstance(n_projections, bool) or not isinstance(n_projections, Integral) or n_projections < 1:
        raise ValueError("n_projections must be a positive integer")
    directions = torch.randn(n_projections, x.shape[1], dtype=torch.float64,
                             generator=torch.Generator().manual_seed(projection_seed))
    directions /= directions.norm(dim=1, keepdim=True)
    px, py = (x @ directions.T).numpy(), (y @ directions.T).numpy()
    if not np.isfinite(px).all() or not np.isfinite(py).all():
        raise FloatingPointError("Validation projections overflowed")
    radial = float(ks_2samp(_radii(x).numpy(), _radii(y).numpy()).statistic)
    projected = float(np.mean([ks_2samp(px[:,i], py[:,i]).statistic for i in range(n_projections)]))
    return {"selection_score": 0.5 * (radial + projected), "radial_ks": radial,
            "projected_ks_mean": projected, "projection_seed": projection_seed,
            "n_projections": n_projections, "n_generated": len(samples), "n_validation": len(validation)}
