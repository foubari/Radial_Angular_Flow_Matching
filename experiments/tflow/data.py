"""Load hashed, already-preprocessed inputs; never synthesize replacements."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(spec):
    path = Path(spec["path"])
    expected = spec.get("sha256")
    if not path.is_absolute():
        raise ValueError(f"Input paths must be absolute: {path}")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"An exact SHA-256 is required for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {path}; expected {expected}, got {actual}")
    return torch.load(path, map_location="cpu", weights_only=True)


@dataclass
class CachedData:
    values: torch.Tensor
    labels: torch.Tensor | None
    indices: dict[str, torch.Tensor]
    mean: torch.Tensor
    external_test: torch.Tensor | None = None
    external_gains: torch.Tensor | None = None
    external_labels: torch.Tensor | None = None

    def split(self, name):
        if name == "test" and self.external_test is not None:
            return self.external_test
        return self.values[self.indices[name]]

    def split_labels(self, name):
        if name == "test" and self.external_test is not None:
            return self.external_labels
        return None if self.labels is None else self.labels[self.indices[name]]


def split_indices(n, spec):
    counts = [spec[k] for k in ("n_train", "n_val", "n_test")]
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in counts):
        raise ValueError("Split counts must be integers")
    if min(counts) < 0 or counts[0] == 0 or counts[1] == 0 or sum(counts) != n:
        raise ValueError("Explicit train/val/test counts must partition the cached rows")
    if spec["kind"] == "indices":
        blob = load_pinned(spec["file"])
        indices = {k: _integer_vector(blob[k], f"{k} indices") for k in ("train", "val", "test")}
        if any(v.ndim != 1 for v in indices.values()) or [len(v) for v in indices.values()] != counts:
            raise ValueError("Split-index shapes/counts do not match the protocol")
        joined = torch.cat(list(indices.values()))
        if not torch.equal(joined.sort().values, torch.arange(n)):
            raise ValueError("Split indices overlap, omit rows or contain invalid indices")
        return indices
    if spec["kind"] == "random":
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(spec["seed"]))
    elif spec["kind"] == "chrono":
        perm = torch.arange(n)
    else:
        raise ValueError("Split kind must be random, chrono or pinned indices")
    a, b, _ = counts
    return {"train": perm[:a], "val": perm[a:a+b], "test": perm[a+b:]}


def _integer_vector(value, name):
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must be an integer vector; fractional labels/indices are not cast")
    return tensor.long()


def _finite(name, values):
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError(f"{name} contains nonfinite values")


def load_data(cfg, *, include_external_test=True):
    """Read exact cached inputs; tuning can omit the separate audio test entirely."""
    spec = cfg["data"]
    blob = load_pinned(spec["input"])
    kind = cfg["kind"]
    labels = None
    external_test = external_gains = external_labels = None
    if kind == "audio":
        if not isinstance(blob, dict) or not {"x", "digit"}.issubset(blob):
            raise ValueError("Expected the original AudioMNIST x/digit cache")
        values = blob["x"].reshape(len(blob["x"]), -1).float()
        labels = _integer_vector(blob["digit"], "audio labels")
        if include_external_test:
            if Path(spec["input"]["path"]).resolve() == Path(spec["external_test"]["path"]).resolve():
                raise ValueError("Audio training and external test resources must be distinct")
            test_blob = load_pinned(spec["external_test"])
            if not isinstance(test_blob, dict) or not {"x", "g"}.issubset(test_blob):
                raise ValueError("Expected original external audio x/g cache")
            external_test = test_blob["x"].reshape(len(test_blob["x"]), -1).float()
            external_gains = test_blob["g"].float()
            if external_test.shape != (3000, values.shape[1]) or external_gains.shape != (3000,):
                raise ValueError("Audio external test cache must contain 3000 matching vectors/gains")
            _finite("External audio test", external_test)
            _finite("External audio gains", external_gains)
            if bool((external_gains <= 0).any().item()):
                raise ValueError("Audio test gains must be strictly positive")
            if "digit" in test_blob:
                external_labels = _integer_vector(test_blob["digit"], "external audio labels")
                if external_labels.shape != (3000,) or bool(((external_labels < 0) | (external_labels >= 10)).any().item()):
                    raise ValueError("External audio labels must be 3000 digit indices")
    else:
        if not isinstance(blob, torch.Tensor):
            raise TypeError("Expected a preprocessed tensor, not an automatically converted payload")
        values = blob.float()
        if "labels" in spec:
            labels = _integer_vector(load_pinned(spec["labels"]), "image labels")
    if values.ndim != 2 or list(values.shape) != spec["shape"]:
        raise ValueError(f"Cached input shape {list(values.shape)} != {spec['shape']}")
    _finite("Cached input", values)
    if labels is not None and (labels.shape != (len(values),) or labels.min() < 0 or labels.max() >= 10):
        raise ValueError("Expected one digit/image class label 0..9 per row")
    indices = split_indices(len(values), spec["split"])
    mean = torch.zeros(values.shape[1], dtype=values.dtype)
    transform = spec["transform"]
    if transform == "piv_recenter":
        mean = values.mean(0)
    elif transform == "train_center":
        mean = values[indices["train"]].mean(0)
    elif transform != "as_cached":
        raise ValueError("Only recorded cached-coordinate transformations are permitted")
    if kind == "audio" and transform != "as_cached":
        raise ValueError("The AudioMNIST gain benchmark must not be centered")
    values = values - mean
    _finite("Preprocessing mean", mean)
    _finite("Preprocessed input", values)
    if kind != "audio" and not len(indices["test"]):
        raise ValueError("Vector/image protocols require a nonempty held-out test split")
    return CachedData(values, labels, indices, mean, external_test, external_gains, external_labels)
