"""Existing downstream backbones and cached-sample image evaluation for t-Flow.

No training, source draws, ODE sampling, downloads, or evaluation occur at import.
Backbones remain raw modules so checkpoints keep their original state_dict keys.
All resource paths are explicit; evaluators never change existing real caches.
"""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn


SIT_COMMIT = "cbde832a40b153ccc79603412409da9c9b0c568c"
SIT_REPOSITORY = "https://github.com/willisma/SiT"
LATENT_SCALE = 0.41407
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _absolute_path(value: str | Path, *, exists: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"resource/output paths must be absolute: {path}")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def _event_shape(model_cfg: Mapping) -> tuple[int, int, int]:
    kind = model_cfg["kind"]
    default = (32, 8, 8) if kind == "image_sit" else (2, 129, 63)
    shape = tuple(model_cfg.get("event_shape", default))
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("event_shape must contain three positive integers")
    if kind == "image_sit" and shape != (32, 8, 8):
        raise ValueError("the existing DC-AE/SiT protocol requires event_shape [32, 8, 8]")
    if kind == "audio_unet" and shape[0] != 2:
        raise ValueError("the existing UNetVel requires two real/imaginary channels")
    return shape


def build_backbone(model_cfg: Mapping, dim: int) -> nn.Module:
    """Build a raw image SiT or audio UNetVel without checkpoint-key prefixes.

    Required kind: image_sit or audio_unet. For image_sit, source_dir must be an
    absolute checkout of SIT_COMMIT. Image defaults match dit_train_sit.py:
    hidden=384, depth=12, heads=6, num_classes=10, class_dropout=0.1.
    Audio defaults match UNetVel: ch=96, mult=[1,2,4], cond=256, attn_from=2.
    This constructor does not load weights or move the model to an accelerator.
    """
    kind = model_cfg["kind"]
    if kind not in ("image_sit", "audio_unet"):
        raise ValueError("kind must be image_sit or audio_unet")
    shape = _event_shape(model_cfg)
    if dim != math.prod(shape):
        raise ValueError(f"dim {dim} does not match event_shape {shape}")
    classes = int(model_cfg.get("num_classes", model_cfg.get("ncls", 10)))
    if classes < 1:
        raise ValueError("num_classes must be positive")
    if kind == "audio_unet":
        from experiments.poc_audio.audio_flow import UNetVel

        model = UNetVel(
            ch=int(model_cfg.get("ch", 96)),
            mult=tuple(model_cfg.get("mult", (1, 2, 4))), ncls=classes,
            cond=int(model_cfg.get("cond", 256)),
            attn_from=int(model_cfg.get("attn_from", 2)),
        )
        model.tflow_provenance = {
            "backbone": "UNetVel", "event_shape": list(shape),
            "source": "experiments/poc_audio/audio_flow.py",
            "source_sha256": _sha256(REPO_ROOT / "experiments/poc_audio/audio_flow.py"),
            "state_dict_format": "original_unwrapped",
        }
        return model
    if "source_dir" in model_cfg:
        repository = _absolute_path(model_cfg["source_dir"])
    else:
        repository = _absolute_path(model_cfg["sit_repo"])
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SIT_COMMIT:
        raise ValueError(f"SiT checkout must be pinned to {SIT_COMMIT}; got {commit}")
    subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet", SIT_COMMIT, "--", "models.py"],
        check=True,
    )
    source = repository / "models.py"
    name = f"_tflow_official_sit_{SIT_COMMIT}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import pinned SiT module at {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    model = sys.modules[name].SiT(
        input_size=8, patch_size=1, in_channels=32,
        hidden_size=int(model_cfg.get("hidden", 384)),
        depth=int(model_cfg.get("depth", 12)), num_heads=int(model_cfg.get("heads", 6)),
        num_classes=classes, class_dropout_prob=float(model_cfg.get("class_dropout", 0.1)),
        learn_sigma=False,
    )
    model.tflow_provenance = {
        "backbone": "SiT", "repository": SIT_REPOSITORY, "commit": commit,
        "source_sha256": _sha256(source), "license": "MIT",
        "license_sha256": _sha256(repository / "LICENSE.txt"),
        "event_shape": list(shape), "state_dict_format": "original_unwrapped",
    }
    return model


def flat_noise_callback(model: nn.Module, model_cfg: Mapping, labels: Tensor):
    """Bind class labels to a plain (flat_state, times)->flat_noise callback.

    Conditional models require integer labels. Audio class dropout, when
    enabled for training, is performed by the caller using the existing null
    label num_classes; SiT performs its own class dropout in model.train().
    Output casting to the input dtype matches the existing mixed-precision
    training scripts' explicit float conversion before residual computation.
    No CFG, amplitude scaling, tangential projection or network wrapper is used.
    """
    shape = _event_shape(model_cfg)
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("labels must be a vector of int64 class indices")
    classes = int(model_cfg.get("num_classes", model_cfg.get("ncls", 10)))
    if bool(((labels < 0) | (labels > classes)).any().item()):
        raise ValueError("labels must be class indices or the existing null-class index")

    def predict(x: Tensor, t: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape != (len(labels), math.prod(shape)):
            raise ValueError("flat batch shape does not match event_shape and bound labels")
        result = model(x.reshape(len(x), *shape), t, labels.to(device=x.device))
        if result.shape != (len(x), *shape):
            raise ValueError("backbone output shape does not match the noise target")
        return result.reshape_as(x).to(dtype=x.dtype)

    return predict


def image_reference_manifest(real_reference_dir: str | Path) -> dict:
    """Hash names and bytes of the complete flat PNG cache actually evaluated.

    Manifest digest: SHA256 over sorted relative filename, NUL, file SHA256,
    newline for each PNG. This never constructs or changes the reference cache.
    """
    root = _absolute_path(real_reference_dir)
    paths = sorted(root.glob("*.png"))
    if len(paths) < 6:
        raise ValueError("image reference requires at least six PNGs for PRDC k=5")
    if any(p.is_dir() for p in root.iterdir()):
        raise ValueError("real reference must be a flat PNG directory, without subdirectories")
    allowed = {".png", ".json", ".txt"}
    if any(p.suffix.lower() not in allowed for p in root.iterdir()):
        raise ValueError("reference contains files outside the recorded PNG/metadata set")
    files = [{"name": p.name, "sha256": _sha256(p)} for p in paths]
    digest = hashlib.sha256()
    for item in files:
        digest.update((item["name"] + "\0" + item["sha256"] + "\n").encode())
    return {"path": str(root), "count": len(files), "sha256": digest.hexdigest(), "files": files}


def _flat_cpu_finite(name: str, value: Tensor, dim: int = 2048) -> Tensor:
    if value.device.type != "cpu" or value.ndim != 2 or value.shape[1] != dim or len(value) < 1:
        raise ValueError(f"{name} must be a nonempty CPU tensor with shape [N, {dim}]")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    result = value.detach().float()
    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError(f"{name} cannot be represented in protocol float32")
    return result


def _inception_features(image_dir: Path, device: torch.device, weights: Path, batch: int):
    from PIL import Image
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torchvision.transforms import PILToTensor

    extractor = FeatureExtractorInceptionV3(
        "inception", ["2048"], feature_extractor_weights_path=str(weights)
    ).to(device).eval()
    paths = sorted(image_dir.glob("*.png"))
    transform = PILToTensor()
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            images = []
            for path in paths[start:start + batch]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            values = extractor(torch.stack(images).to(device))[0].float().cpu()
            if not bool(torch.isfinite(values).all().item()):
                raise FloatingPointError("Inception features contain non-finite values")
            features.append(values.numpy())
    return np.concatenate(features)


@torch.no_grad()
def evaluate_image(
    samples: Tensor,
    test_data: Tensor,
    *,
    train_mean: Tensor,
    cfg: Mapping,
    output_dir: str | Path,
    labels: Tensor | None = None,
) -> dict:
    """Evaluate already generated centered DC-AE latents under the existing protocol.

    Required cfg paths (absolute): decoder_path (local AutoencoderDC snapshot),
    inception_weights_path, real_reference_dir, generated_image_dir (new/empty;
    place generated datasets under the workspace data root). Also required:
    expected_reference_sha256, the digest from image_reference_manifest.
    Optional device='cpu', decode_batch=64, feature_batch=128, real_n, metric_seed=2020.
    output_dir receives JSON provenance/metrics only; real/model resources are read-only.
    No checkpoint loading, source draw, ODE, reference construction or downloads occur.
    """
    samples = _flat_cpu_finite("samples", samples)
    test_data = _flat_cpu_finite("test_data", test_data)
    if len(samples) < 6:
        raise ValueError("at least six generated samples are required for PRDC k=5")
    if train_mean.device.type != "cpu" or train_mean.shape not in ((2048,), (1, 2048)):
        raise ValueError("train_mean must be a CPU tensor with 2048 coordinates")
    mean = train_mean.detach().reshape(1, -1).float()
    if not bool(torch.isfinite(mean).all().item()):
        raise FloatingPointError("train_mean contains non-finite values")
    if labels is not None and (labels.shape != (len(samples),) or labels.dtype != torch.int64):
        raise ValueError("labels must contain one int64 class per generated sample")
    decoder_path = _absolute_path(cfg["decoder_path"])
    weights = _absolute_path(cfg["inception_weights_path"])
    for filename, expected in cfg.get("decoder_files_sha256", {}).items():
        if Path(filename).name != filename or _sha256(decoder_path / filename) != expected:
            raise ValueError(f"Frozen decoder file hash mismatch: {filename}")
    if cfg.get("inception_weights_sha256") and _sha256(weights) != cfg["inception_weights_sha256"]:
        raise ValueError("Frozen Inception weights hash mismatch")
    reference = image_reference_manifest(cfg["real_reference_dir"])
    if reference["sha256"] != cfg["expected_reference_sha256"]:
        raise ValueError("real-reference cache hash does not match the frozen protocol")
    if "real_n" in cfg and int(cfg["real_n"]) != reference["count"]:
        raise ValueError("real_n differs from the number of cached PNGs actually evaluated")
    generated = _absolute_path(cfg["generated_image_dir"], exists=False)
    output = _absolute_path(output_dir, exists=False)
    if generated == Path(reference["path"]) or generated == decoder_path:
        raise ValueError("generated_image_dir must not alias a resource directory")
    if generated.exists() and any(generated.iterdir()):
        raise FileExistsError(f"generated_image_dir must be new or empty: {generated}")
    if (output / "image_metrics.json").exists():
        raise FileExistsError(output / "image_metrics.json")
    decode_batch = int(cfg.get("decode_batch", 64))
    feature_batch = int(cfg.get("feature_batch", 128))
    if min(decode_batch, feature_batch) < 1:
        raise ValueError("decode_batch and feature_batch must be positive")
    device = torch.device(cfg.get("device", "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("image evaluation supports CPU or CUDA")

    # Import dependencies before creating any output. Every model resource is local.
    from diffusers import AutoencoderDC
    from prdc import compute_prdc
    import torch_fidelity
    from torchvision.utils import save_image
    from rafm.metrics.radial import radial_metrics
    from rafm.metrics.distributional import sliced_wasserstein

    radial = radial_metrics(samples, test_data)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(0).get_state())
        sw1 = float(sliced_wasserstein(samples, test_data, n_projections=200))
    ae = AutoencoderDC.from_pretrained(
        str(decoder_path), torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    generated.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(samples), decode_batch):
        latent = ((samples[start:start + decode_batch] + mean) / LATENT_SCALE).to(device)
        if not bool(torch.isfinite(latent).all().item()):
            raise FloatingPointError("inverse latent preprocessing produced non-finite values")
        decoded = ae.decode(latent.reshape(-1, 32, 8, 8)).sample
        if not bool(torch.isfinite(decoded).all().item()):
            raise FloatingPointError("DC-AE decoding produced a non-finite image")
        # Pixel clipping is the original decoder/image protocol, not ODE clipping.
        images = ((decoded.clamp(-1, 1) + 1) / 2).cpu()
        for offset, image in enumerate(images):
            save_image(image, generated / f"{start + offset:05d}.png")
    del ae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    kid_subset = min(1000, len(samples) // 2, reference["count"])
    metric_seed = int(cfg.get("metric_seed", 2020))
    device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with device_context:
        metrics = torch_fidelity.calculate_metrics(
            input1=str(generated), input2=reference["path"], cuda=device.type == "cuda",
            fid=True, kid=True, kid_subset_size=kid_subset, batch_size=feature_batch,
            feature_extractor_weights_path=str(weights), rng_seed=metric_seed,
            cache=False, verbose=False,
        )
    fake_features = _inception_features(generated, device, weights, feature_batch)
    real_features = _inception_features(Path(reference["path"]), device, weights, feature_batch)
    prdc = compute_prdc(real_features, fake_features, nearest_k=5)
    image_metrics = {
        "fid": float(metrics["frechet_inception_distance"]),
        "kid": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
        **{name: float(prdc[name]) for name in ("precision", "recall", "density", "coverage")},
    }
    latent_metrics = {"radial_w1": float(radial["radial_w1"]), "ks": float(radial["ks_stat"]), "sliced_w1": sw1}
    if not all(math.isfinite(value) for value in (*image_metrics.values(), *latent_metrics.values())):
        raise FloatingPointError("image/latent metrics contain non-finite values")
    protocol_file = REPO_ROOT / "experiments/image_latents/dit/dit_eval_sit.py"
    versions = {}
    for name in ("torch", "torchvision", "torch-fidelity", "prdc", "diffusers", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    result = {
        "method": "tflow", "n": len(samples), "real_n": reference["count"], "cfg": 1.0,
        "latent": latent_metrics, "image": image_metrics,
        "reference": {key: reference[key] for key in ("path", "count", "sha256")},
        "protocol": {
            "source": str(protocol_file), "source_sha256": _sha256(protocol_file),
            "decoder_path": str(decoder_path), "latent_scale": LATENT_SCALE,
            "inception_weights_path": str(weights), "inception_weights_sha256": _sha256(weights),
            "inception_layer": "2048", "kid_subset_size": kid_subset,
            "kid_rng_seed": metric_seed, "prdc_nearest_k": 5,
            "sliced_projections": 200, "sliced_seed": 0,
            "generated_image_dir": str(generated), "package_versions": versions,
            "precision": "float32", "device": str(device),
        },
    }
    if labels is not None:
        result["class_counts"] = {str(int(k)): int(v) for k, v in zip(*torch.unique(labels.cpu(), return_counts=True))}
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_manifest.json").write_text(json.dumps(reference, indent=2) + "\n")
    (output / "image_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
