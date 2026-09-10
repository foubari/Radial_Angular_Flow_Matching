"""RAFM-Ang input ablation, with the original population loss and sampler.

Arm A passes the ambient state to the original backbone. Arms B/C pass only
unit direction, plus a standardized log-radius or a constant zero scalar.
The B/C architectures and parameter initializations are identical. Nothing in
this module draws paths, changes their coupling, or introduces an integrator.

Construct on CPU with the same training seed in each arm, then move to the
benchmark device. The caller retains the benchmark's class dropout, optimizer,
EMA, precision, angular-MSE reduction, source draws, and ODE implementation.
For sampling, an existing sampler can call the angular model directly and do
its existing velocity reconstruction/projection. Do not project it twice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from rafm.flow_matching.sampler import _project_tangent
from rafm.models.mlp import MLP


SCHEMA_VERSION = 1
ARMS = ("A", "B", "C")
RADIUS_EMBEDDING_WIDTH = 16


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _sha_value(value: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    return value.lower()


@dataclass(frozen=True)
class RadiusStatistics:
    """Training-only population mean/std of log(max(||X||, radius_floor))."""

    mean_log_radius: float
    population_std_log_radius: float
    standardization_scale: float
    radius_floor: float
    std_floor: float
    n_training_samples: int
    dimension: int
    training_data_sha256: str
    training_indices_sha256: str
    zero_radius_count: int
    below_radius_floor_count: int
    fit_split: str = "generator_training"
    accumulation_dtype: str = "float64"

    def __post_init__(self):
        numeric = (self.mean_log_radius, self.population_std_log_radius,
                   self.standardization_scale, self.radius_floor, self.std_floor)
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("radius statistics must be finite")
        if self.population_std_log_radius < 0 or self.radius_floor <= 0 or self.std_floor <= 0:
            raise ValueError("invalid radius-statistics scales")
        if self.standardization_scale != max(self.population_std_log_radius, self.std_floor):
            raise ValueError("standardization scale does not match population std/floor")
        if self.n_training_samples < 1 or self.dimension < 1:
            raise ValueError("radius statistics require a nonempty training split")
        if not 0 <= self.zero_radius_count <= self.below_radius_floor_count <= self.n_training_samples:
            raise ValueError("invalid radius safeguard counts")
        if self.fit_split != "generator_training" or self.accumulation_dtype != "float64":
            raise ValueError("radius statistics must be fitted to generator training data in float64")
        _sha_value(self.training_data_sha256, "training_data_sha256")
        _sha_value(self.training_indices_sha256, "training_indices_sha256")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> "RadiusStatistics":
        return cls(**dict(value))


def fit_radius_statistics(
    training_flat: Tensor,
    *,
    training_data_sha256: str,
    training_indices_sha256: str,
    radius_floor: float = 1e-8,
    std_floor: float = 1e-8,
) -> RadiusStatistics:
    """Fit once on the already selected, already preprocessed training tensor.

    No splitting, centering, whitening, data generation, or test-set access is
    performed. The caller supplies the exact training tensor and its cache /
    split-index identities. The scalar statistics use float64 population std
    (correction=0); a degenerate distribution uses std_floor as its divisor.
    """
    if training_flat.ndim != 2 or min(training_flat.shape) < 1:
        raise ValueError("training_flat must be a nonempty (samples, dimension) tensor")
    if not training_flat.is_floating_point():
        raise ValueError("training_flat must have a floating dtype")
    if not math.isfinite(radius_floor) or radius_floor <= 0:
        raise ValueError("radius_floor must be positive and finite")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be positive and finite")
    data_sha = _sha_value(training_data_sha256, "training_data_sha256")
    index_sha = _sha_value(training_indices_sha256, "training_indices_sha256")
    with torch.no_grad():
        values = training_flat.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("training radius statistics cannot use nonfinite data")
        radii = values.norm(dim=1)
        if not bool(torch.isfinite(radii).all()):
            raise ValueError("training radii are nonfinite")
        log_radii = radii.clamp_min(radius_floor).log()
        mean = float(log_radii.mean())
        std = float(log_radii.std(unbiased=False))
        return RadiusStatistics(
            mean, std, max(std, std_floor), radius_floor, std_floor,
            len(values), values.shape[1], data_sha, index_sha,
            int((radii == 0).sum()), int((radii < radius_floor).sum()),
        )


def unit_direction(x: Tensor) -> Tensor:
    """Scale-stable unit direction; exact zero maps to an all-zero direction.

    Scaling by the largest absolute component before taking the L2 norm avoids
    underflow for tiny nonzero states. Unlike x / norm.clamp_min(eps), this does
    not feed residual radius information to C for radii smaller than eps.
    The division guards replace only an exact zero denominator with one.
    """
    maximum = x.abs().amax(dim=1, keepdim=True)
    scaled = x / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    norm = scaled.norm(dim=1, keepdim=True)
    return scaled / torch.where(norm > 0, norm, torch.ones_like(norm))


def angular_target(x_t: Tensor, conditional_velocity: Tensor, radius_floor: float = 1e-8) -> Tensor:
    """Original angular target u_t / ||x_t||, with the original 1e-8 guard."""
    if x_t.ndim != 2 or conditional_velocity.shape != x_t.shape:
        raise ValueError("angular target requires matching flat state and velocity tensors")
    if not math.isfinite(radius_floor) or radius_floor <= 0:
        raise ValueError("target radius floor must be positive and finite")
    return conditional_velocity / x_t.norm(dim=1, keepdim=True).clamp_min(radius_floor)


class RAFMAngInputModel(nn.Module):
    """Flat-state angular predictor around a raw, unchanged backbone class.

    For downstream backbones a scoped forward hook adds the radius embedding
    to the existing time embedding. It is removed in a finally block and does
    not replace or reproduce the original backbone forward implementation.
    The hook output participates normally in autograd. A model instance must
    not be called concurrently from multiple threads (ordinary single-GPU
    training and sampling are sequential). No mutable radius state is retained.
    """

    def __init__(self, backbone: nn.Module, model_cfg: Mapping, dim: int, arm: str,
                 radius_statistics: RadiusStatistics | None = None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if dim < 1:
            raise ValueError("dimension must be positive")
        self.arm = arm
        self.dim = dim
        self.model_cfg = copy.deepcopy(dict(model_cfg))
        self.kind = self.model_cfg["kind"]
        if self.kind not in ("mlp", "audio_unet", "image_sit"):
            raise ValueError("unknown RAFM-Ang backbone kind")
        self.backbone = backbone
        self.radius_statistics = radius_statistics
        self.network_calls = 0
        self.original_parameter_count = sum(p.numel() for p in backbone.parameters())
        self.event_shape = None if self.kind == "mlp" else tuple(
            self.model_cfg.get("event_shape", (2, 129, 63) if self.kind == "audio_unet" else (32, 8, 8)))
        if self.event_shape is not None and math.prod(self.event_shape) != dim:
            raise ValueError("event shape does not match flat dimension")
        if radius_statistics is not None and radius_statistics.dimension != dim:
            raise ValueError("radius-statistics dimension does not match the backbone")
        if arm == "A":
            self.radius_embedding = None
            return
        if radius_statistics is None:
            raise ValueError("B/C require generator-training radius statistics")
        self.register_buffer("radius_mean_log", torch.tensor(radius_statistics.mean_log_radius, dtype=torch.float64))
        self.register_buffer("radius_std_scale", torch.tensor(radius_statistics.standardization_scale, dtype=torch.float64))
        if self.kind == "mlp":
            if not isinstance(backbone, MLP) or backbone.pre is not None:
                raise ValueError("B/C vector backbone must be the original MLP with no preprocessing")
            old = backbone.net[0]
            if not isinstance(old, nn.Linear) or old.in_features != dim + 1:
                raise ValueError("unexpected original MLP input layout")
            # Input layout becomes [unit direction, scalar radius condition, time].
            # Preserve every original coordinate/time weight and bias. Only the
            # new radius column has newly initialized weights.
            widened = nn.Linear(old.in_features + 1, old.out_features,
                                bias=old.bias is not None, device=old.weight.device,
                                dtype=old.weight.dtype)
            with torch.no_grad():
                widened.weight[:, :dim].copy_(old.weight[:, :dim])
                widened.weight[:, -1].copy_(old.weight[:, -1])
                if old.bias is not None:
                    widened.bias.copy_(old.bias)
            backbone.net[0] = widened
            self.radius_embedding = None
        else:
            time_module = self._time_embedding_module()
            if self.kind == "audio_unet":
                condition_dim = int(backbone.yemb.embedding_dim)
            else:
                condition_dim = int(backbone.y_embedder.embedding_table.embedding_dim)
            if not isinstance(time_module, nn.Module):
                raise ValueError("backbone has no recognized time-conditioning pathway")
            self.radius_embedding = nn.Sequential(
                nn.Linear(1, RADIUS_EMBEDDING_WIDTH), nn.SiLU(),
                nn.Linear(RADIUS_EMBEDDING_WIDTH, condition_dim),
            )

    def _time_embedding_module(self) -> nn.Module:
        return self.backbone.temb if self.kind == "audio_unet" else self.backbone.t_embedder

    def condition(self, flat_x: Tensor) -> Tensor:
        """Return B's h(R) or C's constant zero; A has no scalar condition."""
        if self.arm == "A":
            raise ValueError("original arm A has no separate radius condition")
        if self.arm == "C":
            return flat_x.new_zeros((len(flat_x), 1))
        radius = flat_x.norm(dim=1, keepdim=True).clamp_min(self.radius_statistics.radius_floor)
        mean = self.radius_mean_log.to(dtype=flat_x.dtype)
        scale = self.radius_std_scale.to(dtype=flat_x.dtype)
        return (radius.log() - mean) / scale

    def forward(self, flat_x: Tensor, t: Tensor, labels: Tensor | None = None) -> Tensor:
        if flat_x.ndim != 2 or flat_x.shape[1] != self.dim:
            raise ValueError("RAFM-Ang input must be a flat (batch, dimension) tensor")
        if t.numel() not in (1, len(flat_x)):
            raise ValueError("time must be scalar or one scalar per example")
        times = t.reshape(-1).expand(len(flat_x))
        if self.kind != "mlp":
            if labels is None or labels.shape != (len(flat_x),) or labels.dtype != torch.long:
                raise ValueError("conditional backbone requires one int64 class label per example")
            if labels.device != flat_x.device:
                raise ValueError("labels and state must be on the same device")
        elif labels is not None:
            raise ValueError("the original vector MLP has no class-conditioning input")
        model_x = flat_x if self.arm == "A" else unit_direction(flat_x)
        # A changing Python counter becomes a guard in a compiled training
        # graph and would trigger recompilation every step. Count only eager
        # calls; the original uncompiled sampling path is counted exactly.
        if not torch.compiler.is_compiling():
            self.network_calls += 1
        if self.kind == "mlp":
            if self.arm == "A":
                result = self.backbone(model_x, times)
            else:
                features = torch.cat((model_x, self.condition(flat_x), times[:, None].float()), dim=1)
                result = self.backbone.net(features)
        else:
            shaped = model_x.reshape(len(flat_x), *self.event_shape)
            if self.arm == "A":
                result = self.backbone(shaped, times, labels)
            else:
                embedded_radius = self.radius_embedding(self.condition(flat_x))

                def add_radius(_module, _inputs, output):
                    if output.shape != embedded_radius.shape:
                        raise ValueError("radius embedding shape does not match original time conditioning")
                    return output + embedded_radius.to(dtype=output.dtype)

                handle = self._time_embedding_module().register_forward_hook(add_radius)
                try:
                    result = self.backbone(shaped, times, labels)
                finally:
                    handle.remove()
            result = result.reshape(len(flat_x), -1)
        if result.shape != flat_x.shape:
            raise ValueError("angular prediction shape differs from the ambient state")
        return result

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        overhead = total - self.original_parameter_count
        return {
            "arm": self.arm, "backbone_kind": self.kind,
            "original_backbone_parameters": self.original_parameter_count,
            "total_parameters": total, "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "conditioning_overhead_parameters": overhead,
            "conditioning_overhead_fraction": overhead / self.original_parameter_count,
            "radius_embedding_hidden_width": RADIUS_EMBEDDING_WIDTH if self.kind != "mlp" and self.arm != "A" else None,
        }

    def checkpoint_metadata(self) -> dict:
        root = Path(__file__).resolve().parents[1]
        files = {"baselines/rafm_input_parameterization.py": _sha256(Path(__file__)),
                 "rafm/models/mlp.py": _sha256(root / "rafm/models/mlp.py"),
                 "rafm/flow_matching/sampler.py": _sha256(root / "rafm/flow_matching/sampler.py")}
        return {
            "schema_version": SCHEMA_VERSION, "method": "rafm_angular_input_ablation",
            "arm": self.arm, "dimension": self.dim, "model_config": copy.deepcopy(self.model_cfg),
            "input": "ambient_state" if self.arm == "A" else "unit_direction",
            "radius_condition": {"A": None, "B": "standardized_log_radius", "C": "constant_zero"}[self.arm],
            "radius_statistics": None if self.radius_statistics is None else self.radius_statistics.to_dict(),
            "direction_safeguard": "scale by maximum absolute coordinate, normalize L2; exact zero maps to zero",
            "conditioning_injection": None if self.arm == "A" else (
                "extra MLP input before time" if self.kind == "mlp" else "add scalar MLP embedding to original time embedding"),
            "original_backbone_provenance": copy.deepcopy(getattr(self.backbone, "tflow_provenance", {
                "backbone": "MLP", "source": "rafm/models/mlp.py", "source_sha256": files["rafm/models/mlp.py"]})),
            "parameters": self.parameter_report(), "implementation_sha256": files,
            "population_objective": "unchanged original matched-radius spherical path and angular MSE",
            "sampler": "unchanged benchmark ambient solver, radius multiplier, tangent projection and guards",
        }

    def checkpoint_payload(self) -> dict:
        return {"metadata": self.checkpoint_metadata(), "state_dict": self.state_dict()}

    def load_checkpoint_payload(self, payload: Mapping) -> None:
        if set(payload) != {"metadata", "state_dict"}:
            raise ValueError("unexpected input-ablation checkpoint structure")
        expected = json.dumps(self.checkpoint_metadata(), sort_keys=True, allow_nan=False)
        recorded = json.dumps(payload["metadata"], sort_keys=True, allow_nan=False)
        if expected != recorded:
            raise ValueError("checkpoint configuration, radius statistics, arm or source identity mismatch")
        current = self.state_dict()
        if set(payload["state_dict"]) != set(current):
            raise ValueError("checkpoint state keys do not match the configured architecture")
        for key, tensor in payload["state_dict"].items():
            if not isinstance(tensor, Tensor) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"checkpoint has a nonfinite or invalid tensor: {key}")
            if tensor.shape != current[key].shape or tensor.dtype != current[key].dtype:
                raise ValueError(f"checkpoint shape or dtype mismatch: {key}")
            if key in ("radius_mean_log", "radius_std_scale") and not torch.equal(tensor.cpu(), current[key].cpu()):
                raise ValueError(f"checkpoint radius-statistics buffer disagrees with recorded fit: {key}")
        self.load_state_dict(payload["state_dict"], strict=True)

    def load_original_backbone_state(self, state_dict: Mapping) -> None:
        """Strict load of an original unwrapped A checkpoint; never reinterpret B/C."""
        if self.arm != "A":
            raise ValueError("historical original checkpoints can only load into arm A")
        for key, value in state_dict.items():
            if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"original checkpoint has a nonfinite or invalid tensor: {key}")
        self.backbone.load_state_dict(state_dict, strict=True)


def build_input_model(model_cfg: Mapping, dim: int, arm: str,
                      radius_statistics: RadiusStatistics | None = None) -> RAFMAngInputModel:
    """Factory using the exact existing MLP, UNetVel, or pinned SiT constructor.

    The original backbone is always initialized first, preserving its parameter
    initialization for matched seeds. New B/C parameters use standard Linear
    initialization afterwards inside an isolated CPU RNG context. Thus A/B/C
    retain the same post-constructor CPU RNG state as the raw backbone. Reset
    the training RNG consistently across arms after construction if the
    benchmark historically resets it before training.
    """
    if model_cfg["kind"] == "mlp":
        if model_cfg.get("premodule") is not None:
            raise ValueError("input ablation does not permit another MLP preprocessing module")
        backbone = MLP(dim, int(model_cfg.get("hidden_dim", 128)), int(model_cfg.get("n_layers", 3)))
    else:
        from baselines.tflow_downstream import build_backbone

        backbone = build_backbone(model_cfg, dim)
    with torch.random.fork_rng(devices=[]):
        return RAFMAngInputModel(backbone, model_cfg, dim, arm, radius_statistics)


def angular_callback(model: RAFMAngInputModel, labels: Tensor | None = None) -> Callable[[Tensor, Tensor], Tensor]:
    """Bind labels for use in an existing angular-MSE trainer / ambient sampler."""
    return lambda x, t: model(x, t, labels)


def ambient_velocity_callback(
    model: RAFMAngInputModel,
    labels: Tensor | None = None,
    *,
    radius_floor: float = 1e-8,
    project_tangent: bool = True,
    projection_r_min: float = 1e-3,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Existing ambient v=R*f followed by the original tangent projection.

    Vector and image code use radius_floor=1e-8. Original AudioMNIST evaluation
    uses radius_floor=0 (unclamped norm). All use projection_r_min=1e-3 and its
    existing small-radius fallback. Applying projection after multiplication is
    the exact original operation order, algebraically R*project(f) away from
    the same guarded region. This function never advances or renormalizes x.
    """
    if not math.isfinite(radius_floor) or radius_floor < 0:
        raise ValueError("sampling radius floor must be nonnegative and finite")
    if not math.isfinite(projection_r_min) or projection_r_min <= 0:
        raise ValueError("projection minimum radius must be positive and finite")

    def velocity(x: Tensor, t: Tensor) -> Tensor:
        angular = model(x, t, labels)
        radius = x.norm(dim=1, keepdim=True)
        if radius_floor:
            radius = radius.clamp_min(radius_floor)
        result = radius * angular
        return _project_tangent(result, x, r_min=projection_r_min) if project_tangent else result

    return velocity


def radius_drift(initial: Tensor, final: Tensor, radius_floor: float = 1e-8) -> dict:
    """Report actual ambient-solver drift; never correct or conceal it."""
    if initial.ndim != 2 or final.shape != initial.shape:
        raise ValueError("radius drift requires matching initial and final flat states")
    if not math.isfinite(radius_floor) or radius_floor <= 0:
        raise ValueError("drift denominator floor must be positive and finite")
    with torch.no_grad():
        start = initial.norm(dim=1)
        end = final.norm(dim=1)
        delta = (end - start).abs()
        relative = delta / start.clamp_min(radius_floor)
        values = {
            "mean_absolute": float(delta.mean()), "max_absolute": float(delta.max()),
            "mean_relative": float(relative.mean()), "max_relative": float(relative.max()),
            "initial_below_radius_floor": int((start < radius_floor).sum()),
            "radius_floor": radius_floor,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("nonfinite radius drift")
        return values
