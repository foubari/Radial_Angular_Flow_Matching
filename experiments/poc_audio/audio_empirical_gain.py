"""Evaluate post-hoc training-ECDF gain on the original fixed-spherical models.

``preflight`` reads paths and JSON only. ``evaluate`` requires a CUDA compute
allocation and the three original EMA checkpoints; it never trains a model.
The existing audio evaluator and archived results are not modified or imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

SEEDS = (8925, 1234, 7)
STEP, N_GEN, RK4_STEPS, SAMPLE_SEED = 24000, 2000, 40, 0
FREQ, FRAMES, DIM = 129, 63, 16254
METHOD = "fixed_spher_empirical_gain"
DEFAULT_SAMPLES_ROOT = Path("/mnt/vast01/users/fouad.oubari/data/tflow/audio_gain")
REFERENCE_METHODS = {
    "gaussian_euclidean": ("gaussian_euclidean", False),
    "matched_euclidean": ("matched_euclidean", False),
    "rafm": ("rafm", False),
    "angular_rafm": ("rafm", True),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_run_directory(args):
    """Dataset artifacts are separate from reports, with a stable collision guard."""
    report_path = str(Path(args.output_dir).expanduser().resolve())
    run_key = hashlib.sha256(report_path.encode("utf-8")).hexdigest()[:24]
    root = Path(getattr(args, "samples_root", DEFAULT_SAMPLES_ROOT)).expanduser().resolve()
    return root / f"run_{run_key}"


def require_compute_allocation():
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError("evaluate requires SLURM_JOB_ID from an allocated compute job; preflight is available anywhere")


def tensor_artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def validate_metadata(meta, seed, method="fixed_spherical", angular=False):
    """Require the paper's actual fixed-radius training protocol, not defaults."""
    args = meta.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"seed {seed}: metadata must contain an args object")
    expected = {
        "method": method, "seed": seed, "split_seed": 0,
        "arch": "unet", "ch": 96, "ncls": 10, "angular": angular,
        "steps": STEP, "batch": 32, "lr": 2e-4, "ema": 0.999,
        "class_dropout": 0.1,
    }
    for key, value in expected.items():
        if key not in args or args[key] != value:
            raise ValueError(f"seed {seed}: metadata {key} must be {value!r}; got {args.get(key)!r}")
    if "depth" not in args or meta.get("D") != DIM:
        raise ValueError(f"seed {seed}: missing depth or incorrect STFT dimension")
    radius = meta.get("R0")
    if not isinstance(radius, (float, int)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"seed {seed}: missing or invalid trained R0")
    return args


def preflight(args):
    """No torch import, tensor load, RNG draw, output creation, or numerical run."""
    shared = {name: Path(getattr(args, name)).resolve() for name in
              ("train_file", "test_file", "classifier", "reference_aggregate")}
    for name, path in shared.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    if shared["train_file"] == shared["test_file"]:
        raise ValueError("training and external test files must be different")
    if Path(args.output_dir).exists():
        raise FileExistsError("output directory must be new; existing results are never overwritten")
    if sample_run_directory(args).exists():
        raise FileExistsError(f"tensor artifact directory already exists: {sample_run_directory(args)}; use a new report output directory")
    runs = {}
    for seed_text, checkpoint_text, metadata_text in args.run:
        seed = int(seed_text)
        if seed not in SEEDS or seed in runs:
            raise ValueError(f"expected each seed {SEEDS} exactly once; got {seed}")
        checkpoint, metadata = Path(checkpoint_text).resolve(), Path(metadata_text).resolve()
        for path in (checkpoint, metadata):
            if not path.is_file():
                raise FileNotFoundError(f"seed {seed}: missing artifact {path}")
        meta = json.loads(metadata.read_text())
        validate_metadata(meta, seed)
        runs[seed] = {"checkpoint": checkpoint, "metadata": metadata, "meta": meta}
    if set(runs) != set(SEEDS):
        raise ValueError(f"all three original checkpoint seeds {SEEDS} are required")
    if len({row["checkpoint"] for row in runs.values()}) != len(SEEDS):
        raise ValueError("each training seed requires its own original checkpoint")
    archived = json.loads(shared["reference_aggregate"].read_text())
    reference = archived.get(str(STEP), {}).get("fixed_spher", {})
    if reference.get("n_seeds") != 3:
        raise ValueError("reference aggregate must contain the three-seed 24000/fixed_spher row")
    for metric in ("digit_acc", "energy_KS", "cov>q95", "cov>q99", "cov<q10", "PIT"):
        values = reference.get(metric, {}).get("vals", [])
        if len(values) != 3 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
            raise ValueError(f"missing or invalid archived per-seed values for {metric}")
    return shared, runs, reference


def preflight_reference_runs(entries):
    """Optional complete-sample controls also require original three-seed artifacts."""
    groups = {}
    for label, seed_text, checkpoint_text, metadata_text in entries or []:
        if label not in REFERENCE_METHODS:
            raise ValueError(f"unknown reference method {label}; choose {list(REFERENCE_METHODS)}")
        seed = int(seed_text)
        group = groups.setdefault(label, {})
        if seed not in SEEDS or seed in group:
            raise ValueError(f"{label}: each original seed must appear exactly once")
        checkpoint, metadata = Path(checkpoint_text).resolve(), Path(metadata_text).resolve()
        for path in (checkpoint, metadata):
            if not path.is_file():
                raise FileNotFoundError(f"{label}/seed {seed}: missing {path}")
        meta = json.loads(metadata.read_text())
        method, angular = REFERENCE_METHODS[label]
        validate_metadata(meta, seed, method=method, angular=angular)
        group[seed] = {"checkpoint": checkpoint, "metadata": metadata, "meta": meta,
                       "method": method, "angular": angular, "label": label, "seed": seed}
    for label, group in groups.items():
        if set(group) != set(SEEDS) or len({row["checkpoint"] for row in group.values()}) != 3:
            raise ValueError(f"{label}: three distinct original checkpoints for {SEEDS} are required")
    return [groups[label][seed] for label in REFERENCE_METHODS if label in groups for seed in SEEDS]


def sample_training_gains(source, n, seed):
    """Use the repository's exact quantile-interpolation law, with isolated CPU RNG.

    The CPU generator is restored afterward; CUDA direction streams are untouched.
    Seed zero pairs these quantiles with the original RAFM source draws. No labels,
    generated directions, external test gains, or classifier outputs enter the draw.
    """
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return source.sample_radii(n, device="cpu").clone()


def replay_original_source_radii(source, n, dimension, seed):
    """Replay audio_eval's source.sample(...).norm(...) in a separate CPU fork.

    Source.sample draws radii first, then directions. The returned norms differ
    from the same raw quantiles only by floating-point sphere normalization.
    These radii document pairing; the actual intervention uses the raw ECDF G.
    """
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return source.sample(n, dimension).norm(dim=1)


def apply_gains(y, gains):
    """Perform only X = G Y / ||Y||, after the fixed-radius ODE has finished."""
    import torch
    if y.ndim != 2 or gains.ndim != 1 or len(y) != len(gains):
        raise ValueError("expected Y of shape (n,d) and gains of shape (n,)")
    norms = y.norm(dim=1, keepdim=True)
    if not torch.isfinite(y).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("generated Y must have finite, strictly positive norms")
    if not torch.isfinite(gains).all() or (gains <= 0).any():
        raise ValueError("empirical gains must be finite and strictly positive")
    return gains.to(y.device).view(-1, 1) * (y / norms)


def invariant_checks(y, x, gains, labels, logits_before, logits_after):
    """Check geometry and every classifier prediction, not just rounded accuracy."""
    import torch
    yu, xu = y / y.norm(dim=1, keepdim=True), x / x.norm(dim=1, keepdim=True)
    error_direction = (yu - xu).norm(dim=1)
    gain_error = (x.norm(dim=1) - gains.to(x.device)).abs()
    relative_gain_error = gain_error / gains.to(x.device)
    pred_before, pred_after = logits_before.argmax(1), logits_after.argmax(1)
    labels = labels.to(pred_before.device)
    count_before = int((pred_before == labels).sum())
    count_after = int((pred_after == labels).sum())
    disagreements = int((pred_before != pred_after).sum())
    logits_finite = bool(torch.isfinite(logits_before).all() and torch.isfinite(logits_after).all())
    logits_close = logits_finite and bool(torch.allclose(logits_before, logits_after, atol=2e-4, rtol=2e-4))
    passed = bool(error_direction.max() <= 2e-6 and relative_gain_error.max() <= 2e-6
                  and logits_close and disagreements == 0 and count_before == count_after)
    return {
        "passed": passed, "direction_l2_max": float(error_direction.max()),
        "radius_absolute_error_max": float(gain_error.max()),
        "radius_relative_error_max": float(relative_gain_error.max()),
        "logits_finite": logits_finite, "logits_close": logits_close,
        "logit_absolute_error_max": float((logits_before - logits_after).abs().max()),
        "prediction_disagreements": disagreements,
        "correct_before": count_before, "correct_after": count_after,
        "accuracy_difference": (count_after - count_before) / len(labels),
        "tolerances": {"direction_l2_max": 2e-6, "radius_relative_error_max": 2e-6,
                       "logits_atol": 2e-4, "logits_rtol": 2e-4},
    }


def energy_metrics(energy, test_gains):
    """Same test quantiles, strict tails, left-search PIT, and rounding as audio_eval."""
    import numpy as np
    from scipy.stats import ks_2samp, wasserstein_distance
    energy, test_gains = np.asarray(energy), np.asarray(test_gains)
    if not (np.isfinite(energy).all() and np.isfinite(test_gains).all()):
        raise ValueError("nonfinite radii in evaluation")
    q90, q95, q99, q10 = np.quantile(test_gains, [0.90, 0.95, 0.99, 0.10])
    raw = {
        "radial_w1": float(wasserstein_distance(energy, test_gains)),
        "ks": float(ks_2samp(energy, test_gains).statistic),
        "cov_gt_q90": float((energy > q90).mean()),
        "cov_gt_q95": float((energy > q95).mean()),
        "cov_gt_q99": float((energy > q99).mean()),
        "cov_lt_q10": float((energy < q10).mean()),
        "pit_mean": float((np.searchsorted(np.sort(test_gains), energy) / len(test_gains)).mean()),
        "gen_energy_mean": float(energy.mean()), "data_energy_mean": float(test_gains.mean()),
    }
    rounded = {k: round(v, 3 if k.endswith("energy_mean") else 4) for k, v in raw.items()}
    return rounded, raw


def summarize(x, logits, labels, test_gains):
    import numpy as np
    energy = x.norm(dim=1).cpu().numpy()
    predictions, labels = logits.argmax(1).cpu(), labels.cpu()
    accuracy = float((predictions == labels).float().mean())
    confidence = float(logits.softmax(1).max(1).values.mean())
    bins = np.digitize(energy, np.quantile(energy, [1 / 3, 2 / 3]))
    by_energy = {}
    for i, name in enumerate(("low_energy", "mid_energy", "high_energy")):
        mask = bins == i
        by_energy[name] = round(float((predictions.numpy()[mask] == labels.numpy()[mask]).mean()), 3) if mask.any() else None
    rounded_energy, raw_energy = energy_metrics(energy, test_gains)
    return {
        "energy": rounded_energy,
        "content": {"digit_acc": round(accuracy, 3), "mean_confidence": round(confidence, 3),
                    "acc_by_energy": by_energy},
        "unrounded": {"energy": raw_energy,
                      "digit_acc": int((predictions == labels).sum()) / len(labels),
                      "correct_count": int((predictions == labels).sum()),
                      "legacy_float32_accuracy": accuracy, "mean_confidence": confidence},
    }


def evaluate_gain_invariance(y, gains, labels, classifier, test_gains):
    """Reusable complete control for any method's already-generated STFT tensor.

    Gaussian, matched-Euclidean, fixed-spherical and RAFM outputs can all use this
    function without changing their original generation. It returns both sets of
    metrics plus every prediction/logit check; it never samples test-data gains.
    """
    import torch
    with torch.no_grad():
        x = apply_gains(y, gains)
        before = classifier((y / y.norm(dim=1, keepdim=True)).reshape(-1, 2, FREQ, FRAMES))
        after = classifier((x / x.norm(dim=1, keepdim=True)).reshape(-1, 2, FREQ, FRAMES))
        checks = invariant_checks(y, x, gains, labels, before, after)
        baseline = summarize(y, before, labels, test_gains)
        posthoc = summarize(x, after, labels, test_gains)
    return x, before, after, baseline, posthoc, checks


def flat_metrics(result, full_precision=False):
    e = result["unrounded"]["energy"] if full_precision else result["energy"]
    accuracy = result["unrounded"]["digit_acc"] if full_precision else result["content"]["digit_acc"]
    return {"digit_acc": accuracy, "energy_KS": e["ks"],
            "cov>q95": e["cov_gt_q95"], "cov>q99": e["cov_gt_q99"],
            "cov<q10": e["cov_lt_q10"], "PIT": e["pit_mean"],
            "radial_w1": e["radial_w1"], "cov>q90": e["cov_gt_q90"]}


def aggregate_runs(rows, full_precision=False):
    import statistics
    if [row["seed"] for row in rows] != list(SEEDS) or not all(row["invariance"]["passed"] for row in rows):
        raise ValueError("a complete passing result for each original seed is required")
    flat = [flat_metrics(row["posthoc"], full_precision=full_precision) for row in rows]
    return {metric: {"mean": statistics.mean(values), "std": statistics.pstdev(values),
                     "vals": values, "n": 3}
            for metric in flat[0] for values in [[row[metric] for row in flat]]}


def rk4_original(model, x0, labels, spherical=True, angular=False):
    """Original audio_eval.rk4: 40 steps, 160 evaluations.

    Kept local because importing audio_eval imports audio_data, which creates a
    repository-relative data directory. No state renormalization or gain enters
    this solver. The only projection is the original velocity tangent projection.
    """
    import torch
    from rafm.flow_matching.sampler import _project_tangent
    dev, x, n, dt = x0.device, x0, x0.shape[0], 1.0 / RK4_STEPS

    def velocity(xx, tt):
        out = model(xx.reshape(n, 2, FREQ, FRAMES), torch.full((n,), tt, device=dev), labels).reshape(n, -1)
        if angular:
            out = xx.norm(dim=1, keepdim=True) * out
        return _project_tangent(out, xx) if spherical else out

    for i in range(RK4_STEPS):
        t0 = i * dt
        k1 = velocity(x, t0)
        k2 = velocity(x + dt / 2 * k1, t0 + dt / 2)
        k3 = velocity(x + dt / 2 * k2, t0 + dt / 2)
        k4 = velocity(x + dt * k3, t0 + dt)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def generate_original(model, state, method, angular, device, batch_size):
    """Unmodified original method initialization, in an isolated RNG scope."""
    import torch
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.no_grad(), torch.random.fork_rng(devices=[device_index]):
        torch.random.default_generator.manual_seed(SAMPLE_SEED)
        with torch.cuda.device(device):
            torch.cuda.manual_seed(SAMPLE_SEED)
        labels = torch.arange(10).repeat_interleave(N_GEN // 10).to(device)
        if method == "gaussian_euclidean":
            x0 = state["sigma"] * torch.randn(N_GEN, DIM, device=device)
        else:
            r = state["src"].sample(N_GEN, DIM).norm(dim=1, keepdim=True).to(device)
            u0 = torch.randn(N_GEN, DIM, device=device)
            x0 = r * u0 / u0.norm(dim=1, keepdim=True)
        y = torch.cat([rk4_original(model, x0[i:i + batch_size], labels[i:i + batch_size], state["spherical"], angular)
                       for i in range(0, N_GEN, batch_size)])
    return y, labels, x0.norm(dim=1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(args, shared, runs, reference, reference_runs=None):
    require_compute_allocation()
    import torch
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1]))
    sys.path.insert(0, str(here))
    from audio_flow import build, make_model
    from audio_classifier import Clf

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a CUDA compute allocation; CPU fallback is disabled")
    device = torch.device(args.device)
    training = torch.load(shared["train_file"], map_location="cpu", weights_only=True)
    external = torch.load(shared["test_file"], map_location="cpu", weights_only=True)
    for label, blob, count in (("training", training, 12000), ("test", external, 3000)):
        if not isinstance(blob, dict) or not {"x", "g", "digit"} <= blob.keys():
            raise ValueError(f"{label}: requires original x, g, digit tensor dictionary")
        if tuple(blob["x"].shape) != (count, 2, FREQ, FRAMES) or tuple(blob["g"].shape) != (count,) or tuple(blob["digit"].shape) != (count,):
            raise ValueError(f"{label}: incorrect original AudioMNIST tensor shapes")
        if not torch.isfinite(blob["x"]).all() or not torch.isfinite(blob["g"]).all() or (blob["g"] <= 0).any():
            raise ValueError(f"{label}: invalid STFT tensors or gains")
        if (blob["digit"] < 0).any() or (blob["digit"] > 9).any() or blob["digit"].is_floating_point():
            raise ValueError(f"{label}: invalid digit labels")
    xtr = training["x"].reshape(12000, -1).float()
    if not torch.allclose(xtr.norm(dim=1), training["g"].float(), rtol=1e-5, atol=1e-6):
        raise ValueError("training tensor norms do not match the stored gains")
    fixed = build("fixed_spherical", xtr, training["digit"].long(), 0)
    radial = build("rafm", xtr, training["digit"].long(), 0)
    if len(fixed["tr"]) != 10200 or not torch.equal(fixed["tr"], radial["tr"]):
        raise ValueError("unexpected generator training split")
    for seed in SEEDS:
        if not math.isclose(fixed["R0"], runs[seed]["meta"]["R0"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"seed {seed}: trained radius does not match the supplied original training data")
    for spec in reference_runs or []:
        if not math.isclose(fixed["R0"], spec["meta"]["R0"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"{spec['label']}/seed {spec['seed']}: original training radius disagrees with data")
    gains = sample_training_gains(radial["src"], N_GEN, args.gain_seed)
    paired_source_radii = replay_original_source_radii(radial["src"], N_GEN, DIM, args.gain_seed)
    # Draws are shared across all model seeds and are never conditioned on content.
    test_gains = external["g"].numpy()
    clf = Clf(ncls=10).to(device).eval()
    clf.load_state_dict(torch.load(shared["classifier"], map_location="cpu", weights_only=True), strict=True)
    fingerprints = {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in shared.items()}
    source_paths = [Path(__file__), here / "audio_flow.py", here / "audio_classifier.py",
                    here.parents[1] / "rafm/sources/radial_empirical.py",
                    here.parents[1] / "rafm/flow_matching/sampler.py"]
    source_fingerprints = {str(path): file_sha256(path) for path in source_paths}
    output = Path(args.output_dir).resolve()
    sample_output = sample_run_directory(args)
    output.mkdir(parents=True, exist_ok=False)
    sample_output.mkdir(parents=True, exist_ok=False)
    paired_gains_path = sample_output / "paired_gains.pt"
    torch.save({"gains": gains, "original_rafm_source_radii": paired_source_radii,
                "gain_seed": args.gain_seed, "training_indices": radial["tr"],
                "training_radii": radial["src"]._r_train, "training_file": fingerprints["train_file"]}, paired_gains_path)
    paired_gains_artifact = tensor_artifact(paired_gains_path)
    write_json(output / "tensor_artifacts.json", {"samples_directory": str(sample_output),
                                                 "paired_gains": paired_gains_artifact})
    rows = []
    for index, seed in enumerate(SEEDS):
        spec, meta = runs[seed], runs[seed]["meta"]
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint.get("step") != STEP or "ema" not in checkpoint:
            raise ValueError(f"seed {seed}: requires the original EMA checkpoint at step {STEP}")
        model = make_model(meta["args"]["arch"], meta["args"]["ch"], meta["args"]["depth"], 10).to(device).eval()
        model.load_state_dict(checkpoint["ema"], strict=True)
        with torch.no_grad():
            y, labels, initial_radius = generate_original(model, fixed, "fixed_spherical", False, device, args.batch_size)
            x, logits_before, logits_after, before, after, checks = evaluate_gain_invariance(y, gains, labels, clf, test_gains)
            relative_drift = (y.norm(dim=1) - initial_radius).abs() / initial_radius
            checks["original_solver_radius_drift_mean"] = float(relative_drift.mean())
            checks["original_solver_radius_drift_max"] = float(relative_drift.max())
        if not checks["passed"]:
            write_json(output / "failure.json", {"status": "failed", "seed": seed, "invariance": checks})
            raise RuntimeError(f"seed {seed}: invariance failed; no aggregate is published")
        original_flat = flat_metrics(before)
        comparisons = {metric: {"archived": reference[metric]["vals"][index], "recomputed": original_flat[metric],
                                "matches": math.isclose(reference[metric]["vals"][index], original_flat[metric], abs_tol=1e-12)}
                       for metric in reference if metric in original_flat}
        row = {"seed": seed, "step": STEP, "n": N_GEN, "method": METHOD,
               "trained_R0": meta["R0"], "recomputed_R0": fixed["R0"],
               "checkpoint": {"path": str(spec["checkpoint"]), "sha256": file_sha256(spec["checkpoint"])},
               "metadata": {"path": str(spec["metadata"]), "sha256": file_sha256(spec["metadata"]), "contents": meta},
               "baseline": before, "posthoc": after, "invariance": checks, "archived_baseline_comparison": comparisons}
        run_dir = output / f"seed_{seed}"
        run_dir.mkdir()
        sample_dir = sample_output / f"seed_{seed}"
        sample_dir.mkdir()
        samples_path = sample_dir / "samples.pt"
        torch.save({"Y": y.cpu(), "X": x.cpu(), "gains": gains, "digit": labels.cpu(),
                    "initial_radii": initial_radius.cpu(), "logits_before": logits_before.cpu(),
                    "logits_after": logits_after.cpu()}, samples_path)
        row["samples"] = tensor_artifact(samples_path)
        row["paired_gains"] = paired_gains_artifact
        write_json(run_dir / "eval.json", row)
        rows.append(row)
        del model, checkpoint, y, x, logits_before, logits_after
    reference_controls = []
    for spec in reference_runs or []:
        label, seed, meta = spec["label"], spec["seed"], spec["meta"]
        state = build(spec["method"], xtr, training["digit"].long(), 0)
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint.get("step") != STEP or "ema" not in checkpoint:
            raise ValueError(f"{label}/seed {seed}: requires original EMA at step {STEP}")
        model = make_model(meta["args"]["arch"], meta["args"]["ch"], meta["args"]["depth"], 10).to(device).eval()
        model.load_state_dict(checkpoint["ema"], strict=True)
        y, labels, initial_radius = generate_original(model, state, spec["method"], spec["angular"], device, args.batch_size)
        x, before_logits, after_logits, before, after, checks = evaluate_gain_invariance(y, gains, labels, clf, test_gains)
        if not checks["passed"]:
            write_json(output / "failure.json", {"status": "failed", "method": label, "seed": seed, "invariance": checks})
            raise RuntimeError(f"{label}/seed {seed}: complete-sample invariance failed")
        row = {"method": label, "seed": seed, "step": STEP, "n": N_GEN,
               "baseline": before, "posthoc": after, "invariance": checks,
               "checkpoint": {"path": str(spec["checkpoint"]), "sha256": file_sha256(spec["checkpoint"])},
               "metadata": {"path": str(spec["metadata"]), "sha256": file_sha256(spec["metadata"]), "contents": meta}}
        control_dir = output / "reference_controls" / label / f"seed_{seed}"
        control_dir.mkdir(parents=True)
        control_samples = sample_output / "reference_controls" / label / f"seed_{seed}"
        control_samples.mkdir(parents=True)
        samples_path = control_samples / "samples.pt"
        torch.save({"Y": y.cpu(), "X": x.cpu(), "gains": gains, "digit": labels.cpu(),
                    "initial_radii": initial_radius.cpu(), "logits_before": before_logits.cpu(),
                    "logits_after": after_logits.cpu()}, samples_path)
        row["samples"] = tensor_artifact(samples_path)
        row["paired_gains"] = paired_gains_artifact
        write_json(control_dir / "eval.json", row)
        reference_controls.append(row)
        del model, checkpoint, state, y, x, before_logits, after_logits
    reference_matches = all(cell["matches"] for row in rows for cell in row["archived_baseline_comparison"].values())
    result = {
        "schema_version": 1, "status": "complete" if reference_matches else "baseline_mismatch",
        "method": METHOD, "inputs": fingerprints, "source_sha256": source_fingerprints,
        "tensor_artifacts": {"samples_directory": str(sample_output), "paired_gains": paired_gains_artifact},
        "protocol": {"training_seeds": list(SEEDS), "checkpoint_step": STEP, "n_gen": N_GEN,
                     "class_counts": [200] * 10, "sample_seed": SAMPLE_SEED, "gain_seed": args.gain_seed,
                     "rk4_steps": RK4_STEPS, "model_evaluations": 4 * RK4_STEPS,
                     "generation_batch_size": args.batch_size, "classifier_batch_size": N_GEN,
                     "tangent_projection": True, "state_renormalization": False, "cfg": 1,
                     "gain_source": "RAFM training split norms; RadialEmpiricalSource ecdf quantile interpolation",
                     "gain_rng": "isolated CPU generator; shared across training seeds",
                     "gain_pairing": "sample seed 0 original RAFM quantile draws" if args.gain_seed == 0 else "shared new gain draw",
                     "training_count": 10200, "external_test_count": 3000, "split_seed": 0,
                     "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                     "slurm_job_id": os.environ["SLURM_JOB_ID"],
                     "device": str(device), "aggregation": "archived per-seed rounding, population std"},
        "aggregate": aggregate_runs(rows), "aggregate_full_precision": aggregate_runs(rows, full_precision=True),
        "runs": rows,
        "archived_baselines_reproduced": reference_matches,
        "gain_pairing_radius_absolute_error_max": float((gains - paired_source_radii).abs().max()),
        "reference_controls": reference_controls,
        "missing_reference_controls": [method for method in REFERENCE_METHODS
                                       if method not in {row["method"] for row in reference_controls}],
    }
    write_json(output / "aggregate.json", result)
    if not reference_matches:
        raise RuntimeError("original baseline differs from archived rounded metrics; inspect comparisons before using the new row")
    print(f"Completed all three seeds: {output / 'aggregate.json'}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("preflight", "evaluate"))
    for name in ("train-file", "test-file", "classifier", "reference-aggregate", "output-dir"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--run", nargs=3, action="append", required=True, metavar=("SEED", "CHECKPOINT", "METADATA"))
    ap.add_argument("--reference-run", nargs=4, action="append", metavar=("METHOD", "SEED", "CHECKPOINT", "METADATA"),
                    help="optional complete controls: gaussian_euclidean, matched_euclidean, rafm, angular_rafm; all three seeds per supplied method")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT,
                    help="root for generated .pt datasets; each report directory gets a distinct hash-named run directory")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gain-seed", type=int, default=0)
    return ap


def main():
    args = parser().parse_args()
    if args.batch_size <= 0 or args.gain_seed < 0:
        raise ValueError("batch size must be positive and gain seed nonnegative")
    shared, runs, reference = preflight(args)
    reference_runs = preflight_reference_runs(args.reference_run)
    if args.command == "preflight":
        print(json.dumps({"status": "inputs_present_metadata_valid", "seeds": list(SEEDS),
                          "reference_controls": [{"method": row["label"], "seed": row["seed"]} for row in reference_runs],
                          "tensor_artifact_directory": str(sample_run_directory(args)),
                          "note": "No tensors loaded. Checkpoint contents, data provenance and trained radius require evaluation."}, indent=2))
        return
    evaluate(args, shared, runs, reference, reference_runs=reference_runs)


if __name__ == "__main__":
    main()
