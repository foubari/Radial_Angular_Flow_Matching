"""Explicit train/evaluate entry points for prepared, protocol-matched t-Flow.

Nothing is launched on import. Unresolved data/protocols and unfrozen source
selection are refused. CLI commands in docs/run_plan.md require user approval.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time
import traceback
from numbers import Integral, Real

import numpy as np
import torch

from baselines.tflow_core import TFlowSourceConfig, sample_student_t, tflow_loss, heun_sample
from experiments.tflow.data import load_data, sha256
from rafm.utils.seeds import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/vast01/users/fouad.oubari/data/tflow")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def implementation_manifest(cfg):
    """Bind resume/selection to relevant source bytes, independent of git status."""
    names = ["baselines/tflow_core.py", "baselines/tflow_downstream.py",
             "experiments/tflow/run.py", "experiments/tflow/data.py",
             "experiments/tflow/validation.py", "experiments/tflow/tune.py", "rafm/utils/seeds.py"]
    if cfg["kind"] == "vector":
        names += ["rafm/models/mlp.py", "rebuttal_experiments/lib/resmlp.py",
                  "rafm/metrics/radial.py", "rafm/metrics/distributional.py",
                  "rafm/metrics/angular.py", "rafm/metrics/stability.py"]
    elif cfg["kind"] == "audio":
        names += ["experiments/poc_audio/audio_flow.py", "experiments/poc_audio/audio_classifier.py",
                  "experiments/poc_audio/audio_empirical_gain.py"]
    else:
        names += ["experiments/image_latents/dit/dit_eval_sit.py", "rafm/metrics/radial.py",
                  "rafm/metrics/distributional.py"]
    files = {name: sha256(REPO_ROOT / name) for name in names}
    if cfg["kind"] == "image":
        source_dir = cfg["model"].get("source_dir", cfg["model"].get("sit_repo"))
        if source_dir is None:
            raise ValueError("Image model requires an explicit pinned source_dir")
        for name in ("models.py", "LICENSE.txt"):
            path = Path(source_dir) / name
            files[str(path.resolve())] = sha256(path)
    return {"files": files, "python": platform.python_version(),
            "torch": str(torch.__version__), "numpy": np.__version__}


def implementation_sha256(cfg):
    return json_hash(implementation_manifest(cfg))


def run_signature(cfg, seed, source, stage="final"):
    return {"config": cfg, "seed": seed, "source": vars(source), "stage": stage,
            "implementation_sha256": implementation_sha256(cfg)}


def sanitize_metrics(value, prefix=""):
    """Return JSON-compatible metrics plus every nested undefined/nonfinite path."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        clean, bad = {}, []
        for key, item in value.items():
            clean[key], fields = sanitize_metrics(item, f"{prefix}.{key}" if prefix else str(key))
            bad.extend(fields)
        return clean, bad
    if isinstance(value, (list, tuple)):
        clean, bad = [], []
        for index, item in enumerate(value):
            item, fields = sanitize_metrics(item, f"{prefix}[{index}]")
            clean.append(item)
            bad.extend(fields)
        return clean, bad
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None, [prefix]
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"Unsupported metric value at {prefix}: {type(value).__name__}")
    return value, []


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_state(value, name):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"Nonfinite checkpoint state: {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_state(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_state(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite checkpoint state: {name}")


def finite_gradients(model):
    """Check the same gradients with one accelerator-to-host synchronization."""
    checks = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
              if parameter.grad is not None]
    return not checks or bool(torch.stack(checks).all().item())


def validate_config(cfg):
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    if cfg.get("blocking_issues") or cfg.get("protocol_status") != "resolved":
        raise ValueError(f"Protocol is unresolved: {cfg.get('blocking_issues')}")
    if cfg["kind"] not in ("vector", "audio", "image"):
        raise ValueError("Unknown benchmark kind")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg["condition_id"]) or cfg["condition_id"] in (".", ".."):
        raise ValueError("condition_id must be a safe single path component")
    train = cfg["training"]
    if train["steps"] not in (10000, 24000, 40000):
        raise ValueError("Final training budget must be a recorded paper budget")
    if train["optimizer"] not in ("Adam", "AdamW"):
        raise ValueError("Invalid optimizer configuration")
    for name in ("lr", "eps"):
        _finite_number(train.get(name, 1e-8), name, positive=True)
    for name in ("batch_size", "checkpoint_every", "log_every"):
        _positive_integer(train.get(name, 200), name)
    if len(train["betas"]) != 2:
        raise ValueError("Adam requires two betas")
    for beta in train["betas"]:
        _finite_number(beta, "Adam beta")
        if not 0 <= beta < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
    if train["batch_rule"] not in ("step_seeded", "global_torch"):
        raise ValueError("Unsupported batch RNG rule")
    if train["precision"] not in ("float32", "bfloat16_autocast"):
        raise ValueError("Unsupported precision")
    if train.get("ema") is not None:
        _finite_number(train["ema"], "EMA")
        if not 0 <= train["ema"] < 1:
            raise ValueError("EMA must lie in [0, 1)")
    if cfg["kind"] == "audio":
        _finite_number(train["class_dropout"], "class dropout")
        if not 0 <= train["class_dropout"] <= 1:
            raise ValueError("Class dropout must lie in [0, 1]")
    if train.get("weight_decay", 0) != 0 or train.get("gradient_clip") is not None:
        raise ValueError("The matched protocol has no weight decay or gradient clipping")
    for name in ("model_evaluations", "n_samples", "sample_batch_size"):
        _positive_integer(cfg["evaluation"][name], name)
    if cfg["evaluation"]["model_evaluations"] % 2:
        raise ValueError("Matched Heun budgets require an even number of actual model evaluations")
    sampler = cfg["sampler"]
    if set(sampler) - {"t_min", "rho", "grid", "sigma_min"}:
        raise ValueError("Sampler configuration contains unsupported or duplicate budget fields")
    _finite_number(sampler["t_min"], "t_min", positive=True)
    if sampler["t_min"] >= 1:
        raise ValueError("t_min must be smaller than one")
    _finite_number(sampler.get("rho", 7), "rho", positive=True)
    if sampler.get("grid", "power_sigma") not in ("linear", "power_time", "power_sigma"):
        raise ValueError("Unknown time-grid convention")
    sigma_min = sampler.get("sigma_min", 0.01)
    _finite_number(sigma_min, "sigma_min", positive=True)
    if sigma_min >= 1 or (sampler.get("grid", "power_sigma") == "power_sigma" and sigma_min >= 1 - sampler["t_min"]):
        raise ValueError("sigma_min must lie below the starting noise level")
    expected = [8925, 77395, 65457] if cfg["kind"] == "vector" else [8925, 1234, 7]
    if cfg["seeds"] != expected:
        raise ValueError("Final model seeds differ from the paper protocol")


def compute_device():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Experiment commands must run inside a compute-node Slurm allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each experiment requires exactly one visible accelerator")
    return torch.device("cuda:0")


def hardware():
    info = {"host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "hip": torch.version.hip}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, vram_bytes=props.total_memory)
    return info


def tensor_fingerprint(value):
    """Hash realized tensor bytes, recording shape and dtype separately."""
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def dataset_manifest(data):
    """Record exact realized splits; an external test is never a validation row."""
    result = {"values": tensor_fingerprint(data.values),
              "preprocessing_mean": tensor_fingerprint(data.mean),
              "split_indices": {name: tensor_fingerprint(index)
                                for name, index in data.indices.items()},
              "splits": {name: tensor_fingerprint(data.split(name))
                         for name in data.indices},
              "external_test_loaded": data.external_test is not None}
    if data.labels is not None:
        result["labels"] = tensor_fingerprint(data.labels)
    if data.external_gains is not None:
        result["external_test_gains"] = tensor_fingerprint(data.external_gains)
    return result


def peak_memory(device, previous=None):
    """Retain measured peaks across checkpoint resumes without fabricating CPU values."""
    if device.type != "cuda":
        return previous
    current = {"allocated_bytes": torch.cuda.max_memory_allocated(device),
               "reserved_bytes": torch.cuda.max_memory_reserved(device)}
    if previous is not None:
        current = {key: max(value, previous.get(key, 0)) for key, value in current.items()}
    return current


def build_model(cfg, dim, device):
    if cfg["kind"] == "vector":
        model_cfg = cfg["model"]
        if model_cfg["kind"] == "mlp":
            from rafm.models.mlp import MLP
            model = MLP(dim, model_cfg["hidden_dim"], model_cfg["n_layers"])
        elif model_cfg["kind"] == "resmlp":
            from rebuttal_experiments.lib.resmlp import ResidualMLP
            model = ResidualMLP(dim, model_cfg["hidden_dim"], model_cfg["n_layers"])
        else:
            raise ValueError("Unsupported vector backbone")
    else:
        from baselines.tflow_downstream import build_backbone
        model = build_backbone(cfg["model"], dim)
    return model.to(device)


def noise_callback(model, cfg, labels=None):
    if cfg["kind"] == "vector":
        return lambda x, t: model(x, t).float()
    from baselines.tflow_downstream import flat_noise_callback
    return flat_noise_callback(model, cfg["model"], labels)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG streams differ from visible accelerators")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def source_from_selection(cfg, path):
    selection = json.loads(Path(path).read_text())
    if selection.get("status") != "frozen" or selection.get("config_sha256") != json_hash(cfg):
        raise ValueError("A frozen validation-only selection for this exact configuration is required")
    if selection.get("selection_split") != "validation":
        raise ValueError("Source selection must use validation data only")
    if selection.get("implementation_sha256") != implementation_sha256(cfg):
        raise ValueError("Frozen selection belongs to different implementation/environment bytes")
    return TFlowSourceConfig(**{k: selection["selected"][k] for k in ("nu", "scale")})


def train(cfg, data, output, seed, source, *, budget=None, stage="final"):
    """Same backbone/optimizer/batches, with direct Student-t noise labels.

    Tuning budgets are explicit shorter experiments, never final checkpoints.
    Final runs always train from step0 to the original complete step budget.
    """
    validate_config(cfg)
    device = compute_device()
    if stage not in ("final", "tuning"):
        raise ValueError("Unknown run stage")
    if stage == "final" and seed not in cfg["seeds"]:
        raise ValueError("A final seed must match the reported seeds")
    full_steps = cfg["training"]["steps"]
    budget = full_steps if budget is None else budget
    _positive_integer(budget, "training budget")
    if not budget <= full_steps or (stage == "final" and budget != full_steps):
        raise ValueError("Final training must use the complete matched budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = run_signature(cfg, seed, source, stage)
    run_hash = json_hash(signature)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != signature:
        raise ValueError("Output directory already belongs to different run/source/code bytes")
    write_json(output / "dataset_manifest.json", dataset_manifest(data))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_all_seeds(seed)
    model = build_model(cfg, data.values.shape[1], device)
    ema_rate = cfg["training"].get("ema")
    ema = copy.deepcopy(model).eval() if ema_rate is not None else None
    if ema is not None:
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
    train_cfg = cfg["training"]
    optimizer = getattr(torch.optim, train_cfg["optimizer"])(
        model.parameters(), lr=train_cfg["lr"], betas=tuple(train_cfg["betas"]),
        eps=train_cfg.get("eps", 1e-8), weight_decay=0)
    train_values = data.split("train").to(device)
    train_labels = data.split_labels("train")
    if train_labels is not None:
        train_labels = train_labels.to(device)
    checkpoint = output / "checkpoint.pt"
    start = 0
    elapsed_before = 0.0
    loss_value = None
    previous_peak_memory = None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["run_sha256"] != run_hash:
            raise ValueError("Checkpoint configuration, source, seed, stage or implementation mismatch")
        for name in ("model", "ema", "optimizer"):
            _finite_state(saved[name], name)
        model.load_state_dict(saved["model"])
        if ema is not None:
            ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        start, elapsed_before = saved["step"], saved["train_time_s"]
        if isinstance(start, bool) or not isinstance(start, Integral) or start < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer")
        _finite_number(elapsed_before, "checkpoint train_time_s")
        if elapsed_before < 0:
            raise ValueError("Checkpoint train_time_s must be nonnegative")
        loss_value = saved.get("last_loss")
        previous_peak_memory = saved.get("peak_memory")
        restore_rng(saved["rng"])
        if start > budget:
            raise ValueError("Checkpoint exceeds requested tuning stage budget")
    # Never replace reviewable metadata before checking existing checkpoint identity.
    write_json(config_path, signature)
    write_json(output / "implementation.json", implementation_manifest(cfg))
    if start == budget:
        existing_stats = output / "training_stats.json"
        if existing_stats.exists():
            stats = json.loads(existing_stats.read_text())
            if stats.get("training_step") == budget and stats.get("run_sha256") == run_hash:
                return stats
        stats = {"run_sha256": run_hash, "total_train_time_s": elapsed_before,
                 "training_step": budget, "final_loss": loss_value, "timing_kind": "measured",
                 "timing_scope": "completed checkpoint snapshot; no additional optimizer steps",
                 "peak_memory": previous_peak_memory,
                 "n_params": sum(p.numel() for p in model.parameters()), "hardware": hardware()}
        write_json(existing_stats, stats)
        return stats
    train_model = torch.compile(model) if train_cfg.get("compile", False) else model
    model.train()
    _synchronize(device)
    begin = time.perf_counter()
    for step in range(start + 1, budget + 1):
        if train_cfg["batch_rule"] == "step_seeded":
            generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
            index = torch.randint(len(train_values), (train_cfg["batch_size"],), generator=generator).to(device)
        else:
            index = torch.randint(len(train_values), (train_cfg["batch_size"],), device=device)
        values = train_values[index]
        labels = None if train_labels is None else train_labels[index].clone()
        if cfg["kind"] == "audio":
            drop = torch.rand(len(values), generator=torch.Generator().manual_seed(seed * 7 + step))
            classes = int(cfg["model"].get("num_classes", cfg["model"].get("ncls", 10)))
            labels[(drop < train_cfg["class_dropout"]).to(device)] = classes
        # SiT performs its own original class dropout while model.training=True.
        predictor = noise_callback(train_model, cfg, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train_cfg["precision"] == "bfloat16_autocast"):
            loss = tflow_loss(predictor, values, source, reduction="batch_mean")
        loss.backward()
        if not finite_gradients(model):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.mul_(ema_rate).add_(current, alpha=1 - ema_rate)
        loss_value = float(loss.detach())
        if step % train_cfg.get("log_every", 200) == 0:
            record = {"step": step, "loss": loss_value}
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
        if step % train_cfg["checkpoint_every"] == 0 or step == budget:
            _synchronize(device)
            elapsed = elapsed_before + time.perf_counter() - begin
            saved = {"run_sha256": run_hash, "model": model.state_dict(),
                     "ema": None if ema is None else ema.state_dict(), "optimizer": optimizer.state_dict(),
                     "step": step, "train_time_s": elapsed, "rng": rng_state(), "stage": stage,
                     "peak_memory": peak_memory(device, previous_peak_memory),
                     "last_loss": loss_value, "implementation_sha256": signature["implementation_sha256"]}
            for name in ("model", "ema", "optimizer"):
                _finite_state(saved[name], name)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(checkpoint)
    _synchronize(device)
    stats = {"run_sha256": run_hash, "total_train_time_s": elapsed_before + time.perf_counter() - begin,
             "training_step": budget, "final_loss": loss_value, "timing_kind": "measured",
             "timing_scope": "training loop including compile, logging, finite checks and checkpoints; excludes data/model setup",
             "peak_memory": peak_memory(device, previous_peak_memory),
             "peak_memory_scope": "model, EMA, optimizer, accelerator training data and training loop; maximum across resumes",
             "n_params": sum(p.numel() for p in model.parameters()), "hardware": hardware()}
    write_json(output / "training_stats.json", stats)
    return stats


def load_trained(cfg, data, output, seed, source, *, stage="final"):
    validate_config(cfg)
    if stage not in ("final", "tuning"):
        raise ValueError("Unknown checkpoint stage")
    device = compute_device()
    saved = torch.load(Path(output) / "checkpoint.pt", map_location="cpu", weights_only=False)
    expected = json_hash(run_signature(cfg, seed, source, stage))
    if saved["run_sha256"] != expected or saved["stage"] != stage:
        raise ValueError("Checkpoint does not match this experiment")
    if stage == "final" and saved["step"] != cfg["training"]["steps"]:
        raise ValueError("Test evaluation requires the complete final training budget")
    for name in ("model", "ema"):
        _finite_state(saved[name], name)
    model = build_model(cfg, data.values.shape[1], device).eval()
    model.load_state_dict(saved["ema"] if saved["ema"] is not None else saved["model"])
    restore_rng(saved["rng"])
    return model, saved


@torch.no_grad()
def sample(cfg, model, source, n, *, seed=None):
    _positive_integer(n, "sample count")
    parameter = next(model.parameters(), None)
    buffer = next(model.buffers(), None)
    device = parameter.device if parameter is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    if seed is not None:
        set_all_seeds(seed)
    labels = None
    if cfg["kind"] != "vector":
        if n % 10:
            raise ValueError("Conditional sample counts must preserve exact class balance")
        labels = torch.arange(10, device=device).repeat_interleave(n // 10)
    dim = cfg["data"]["shape"][1]
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    initial = sample_student_t((n, dim), source, device=device)
    output = []
    batch = cfg["evaluation"]["sample_batch_size"]
    _positive_integer(batch, "sample batch size")
    nfe = cfg["evaluation"]["model_evaluations"]
    _positive_integer(nfe, "model evaluations")
    if nfe % 2:
        raise ValueError("Heun requires an even actual-NFE budget")
    sampler_cfg = cfg["sampler"]
    was_training = model.training
    model.eval()
    total_calls = 0
    try:
        for i in range(0, n, batch):
            predictor = noise_callback(model, cfg, None if labels is None else labels[i:i+batch])
            sampled = heun_sample(predictor, initial[i:i+batch], n_steps=nfe // 2, **sampler_cfg)
            if sampled["nfe"] != nfe:
                raise RuntimeError("Actual Heun call count differs from the requested NFE")
            output.append(sampled["samples"].cpu())
            total_calls += sampled["nfe"]
    finally:
        model.train(was_training)
    _synchronize(device)
    elapsed = time.perf_counter() - begin
    return {"samples": torch.cat(output), "labels": None if labels is None else labels.cpu(),
            "sample_time_s": elapsed, "nfe": sampled["nfe"],
            "time_grid": sampled["time_grid"].cpu(), "timing_kind": "measured",
            "n_batches": len(output), "model_calls_total": total_calls,
            "peak_memory": peak_memory(device),
            "nfe_scope": "model calls per trajectory; model_calls_total includes all minibatches",
            "timing_scope": "source draw, Heun, strict finite checks, and CPU sample transfers; excludes metrics"}


def sample_artifact_path(cfg, seed, source, *, checkpoint_sha256=None):
    root = Path(cfg["evaluation"].get("samples_root") or DATA_ROOT / "samples")
    if not root.is_absolute():
        raise ValueError("evaluation.samples_root must be absolute")
    identity_fields = run_signature(cfg, seed, source)
    if checkpoint_sha256 is not None:
        identity_fields["checkpoint_sha256"] = checkpoint_sha256
    identity = json_hash(identity_fields)
    return root / cfg["condition_id"] / f"seed_{seed}" / identity / "samples.pt"


def vector_metrics(samples, test):
    from rafm.metrics.radial import radial_metrics
    from rafm.metrics.distributional import distributional_metrics
    from rafm.metrics.angular import angular_metrics
    from rafm.metrics.stability import stability_metrics
    # Preserve original metric algorithms and their native field names.
    result = radial_metrics(samples, test)
    result.update(distributional_metrics(samples, test, n_projections=500))
    result.update(angular_metrics(samples, test, n_bins=4, n_projections=200))
    result.update(stability_metrics(samples))
    result["inf_rate"] = float(torch.isinf(samples).any(dim=1).float().mean())
    return result


def evaluate(cfg, data, output, seed, source):
    output = Path(output).resolve()
    model, checkpoint = load_trained(cfg, data, output, seed, source)
    checkpoint_hash = sha256(output / "checkpoint.pt")
    generated = sample(cfg, model, source, cfg["evaluation"]["n_samples"], seed=cfg["evaluation"]["sample_seed"])
    sample_path = sample_artifact_path(cfg, seed, source, checkpoint_sha256=checkpoint_hash)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sample_path.with_suffix(".tmp")
    torch.save(generated, temporary)
    temporary.replace(sample_path)
    samples = generated["samples"]
    downstream_metadata = None
    if cfg["kind"] == "vector":
        set_all_seeds(cfg["evaluation"]["metric_seed"])
        metrics = vector_metrics(samples, data.split("test"))
    elif cfg["kind"] == "audio":
        from experiments.poc_audio.audio_classifier import Clf
        from experiments.poc_audio.audio_empirical_gain import summarize, flat_metrics
        from experiments.tflow.data import load_pinned
        classifier = Clf(ncls=10).to(next(model.parameters()).device).eval()
        classifier.load_state_dict(load_pinned(cfg["evaluation"]["classifier"]))
        with torch.no_grad():
            norms = samples.norm(dim=1, keepdim=True)
            if not bool(torch.isfinite(norms).all().item()) or bool((norms <= 0).any().item()):
                raise FloatingPointError("Audio classifier directions require finite positive norms")
            if data.external_gains is None:
                raise ValueError("Full audio evaluation requires the pinned external test gains")
            directions = (samples / norms).reshape(-1,2,129,63)
            # Match the original audio_eval.py and completed fixed-gain control:
            # one full 2,000-example classifier forward, with no AMP context.
            logits = classifier(directions.to(next(model.parameters()).device)).cpu()
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("Audio classifier produced nonfinite logits")
        audio_summary = summarize(samples, logits, generated["labels"], data.external_gains.numpy())
        metrics = flat_metrics(audio_summary, full_precision=True)
        audio_artifact = sample_path.with_name("audio_evaluation.pt")
        torch.save({"logits": logits, "predictions": logits.argmax(1),
                    "labels": generated["labels"], "generated_radii": norms[:, 0]}, audio_artifact)
        downstream_metadata = {"audio_summary": audio_summary,
                               "classifier_batch_size": len(samples),
                               "classifier_forward_calls": 1,
                               "classifier_precision": "float32_no_autocast",
                               "classifier": cfg["evaluation"]["classifier"],
                               "audio_artifact": {"path": str(audio_artifact), "sha256": sha256(audio_artifact)}}
    else:
        from baselines.tflow_downstream import evaluate_image
        image_cfg = dict(cfg["evaluation"])
        generated_root = Path(image_cfg.get("generated_image_root") or DATA_ROOT / "generated_images")
        if not generated_root.is_absolute():
            raise ValueError("generated_image_root must be absolute")
        # Separate seeds/configurations use distinct new caches; no old cache is removed.
        image_cfg["generated_image_dir"] = str(generated_root / cfg["condition_id"] / f"seed_{seed}" /
                                               sample_path.parent.name)
        image_cfg["device"] = str(next(model.parameters()).device)
        image_result = evaluate_image(samples, data.split("test"), train_mean=data.mean,
                                      cfg=image_cfg, output_dir=output / "image_evaluation",
                                      labels=generated["labels"])
        metrics = {**image_result["image"], **image_result["latent"]}
        # Keep the original image key 'ks', which the image renderer requires.
        downstream_metadata = {key: value for key, value in image_result.items() if key not in ("image", "latent")}
    metrics.update(nfe=generated["nfe"], sample_time_s=generated["sample_time_s"],
                   total_train_time_s=checkpoint["train_time_s"],
                   n_params=sum(p.numel() for p in model.parameters()))
    clean_metrics, bad = sanitize_metrics(metrics)
    envelope = {"schema_version":1, "condition_id":cfg["condition_id"], "method":"tflow", "seed":seed, "stage":"final",
                "status":"failed" if bad else "complete", "config":cfg, "config_sha256":json_hash(cfg),
                "source":vars(source), "hardware":hardware(),
                "dataset_manifest": dataset_manifest(data),
                "peak_memory": {"training": checkpoint.get("peak_memory"),
                                "sampling": generated["peak_memory"],
                                "sampling_and_evaluation": peak_memory(next(model.parameters()).device)},
                "implementation_sha256": implementation_sha256(cfg),
                "checkpoint": {"path": str(output / "checkpoint.pt"), "sha256": checkpoint_hash,
                               "step": checkpoint["step"], "run_sha256": checkpoint["run_sha256"]},
                "sample_artifact": {"path": str(sample_path), "sha256": sha256(sample_path)},
                "sampler": {"nfe": generated["nfe"], "model_calls_total": generated["model_calls_total"],
                            "n_batches": generated["n_batches"], "time_grid": generated["time_grid"].tolist(),
                            "parameters": cfg["sampler"], "timing_scope": generated["timing_scope"]},
                "metrics":clean_metrics, "nonfinite_metric_fields":bad}
    if downstream_metadata is not None:
        envelope["downstream_evaluation"] = downstream_metadata
    write_json(output / "result.json", envelope)
    if bad:
        raise FloatingPointError(f"Nonfinite/undefined metrics, recorded explicitly: {bad}")
    write_json(output / "metrics.json", clean_metrics)
    return envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "train-evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", default="outputs_tflow")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    source = source_from_selection(cfg, args.selection)
    output = Path(args.output) / cfg["condition_id"] / f"seed_{args.seed}"
    if (output / "config.json").exists():
        existing_config = json.loads((output / "config.json").read_text())
        if existing_config != run_signature(cfg, args.seed, source):
            raise ValueError("Output directory contains a different run identity; it was not modified")
    if (output / "result.json").exists():
        prior = json.loads((output / "result.json").read_text())
        if prior.get("status") == "complete":
            if (prior.get("config_sha256") != json_hash(cfg) or prior.get("source") != vars(source)
                    or prior.get("implementation_sha256") != implementation_sha256(cfg)):
                raise ValueError("Existing result belongs to a different configuration/source")
            print(f"Already complete: {output}")
            return
        raise RuntimeError(f"Prior failure is preserved at {output}; review before retrying explicitly")
    try:
        data = load_data(cfg)
        if args.action in ("train", "train-evaluate"):
            train(cfg, data, output, args.seed, source)
        if args.action in ("evaluate", "train-evaluate"):
            evaluate(cfg, data, output, args.seed, source)
    except Exception as error:
        if not (output / "result.json").exists():
            write_json(output / "result.json", {"schema_version":1, "condition_id":cfg["condition_id"],
                       "method":"tflow", "seed":args.seed, "stage":"final", "status":"failed", "metrics":{},
                       "config":cfg, "config_sha256":json_hash(cfg), "source":vars(source),
                       "implementation_sha256": implementation_sha256(cfg), "hardware": hardware(),
                       "failure":{"type":type(error).__name__, "message":str(error), "traceback":traceback.format_exc()}})
        raise


if __name__ == "__main__":
    main()
