#!/usr/bin/env python3
"""Actual-size A/B/C backbone interfaces on one scheduled GPU, without training.

Only deterministic synthetic fixtures are used. There are no optimizer steps,
benchmark data loads, source tuning, or numerical solver integrations. A second
graph check explicitly randomizes the otherwise zero-initialized output heads
inside disposable fixture models so conditioning gradients are observable.
Those modified weights are never used by experiment jobs or saved as models.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from baselines.rafm_input_parameterization import ARMS, build_input_model, fit_radius_statistics
from experiments.tflow import run as common
from experiments.tflow.data import sha256
from rafm.utils.seeds import set_all_seeds


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        item = common.tensor_fingerprint(value)
        digest.update(json.dumps({"name": name, **item}, sort_keys=True).encode())
    return digest.hexdigest()


def module_fingerprint(model):
    structure = [{"name": name, "class": f"{type(module).__module__}.{type(module).__qualname__}"}
                 for name, module in model.named_modules()]
    return common.json_hash(structure)


def fixture(dim):
    generator = torch.Generator().manual_seed(58301)
    values = torch.randn(8, dim, generator=generator)
    values = values / values.norm(dim=1, keepdim=True)
    values *= torch.tensor([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])[:, None]
    indices = torch.arange(8)
    stats = fit_radius_statistics(values,
        training_data_sha256=common.tensor_fingerprint(values)["sha256"],
        training_indices_sha256=common.tensor_fingerprint(indices)["sha256"])
    x = torch.randn(2, dim, generator=generator)
    x = x / x.norm(dim=1, keepdim=True) * torch.tensor([0.8, 2.7])[:, None]
    target = torch.randn(2, dim, generator=generator)
    return x, target, stats


def gradients(model):
    available = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not available:
        raise RuntimeError("No backbone gradients exist")
    finite = bool(torch.stack([torch.isfinite(value).all() for value in available]).all().item())
    maximum = float(torch.stack([value.abs().max() for value in available]).max().item())
    if not finite or maximum == 0:
        raise RuntimeError("Backbone gradients are nonfinite or all zero")
    return {"finite": finite, "nonzero": maximum > 0, "maximum_absolute": maximum,
            "tensors_with_gradients": len(available)}


def condition_gradients(model):
    if model.arm == "A":
        return {"applicable": False}
    if model.kind == "mlp":
        value = model.backbone.net[0].weight.grad
        values = [] if value is None else [value[:, model.dim]]
    else:
        values = [parameter.grad for parameter in model.radius_embedding.parameters() if parameter.grad is not None]
    if not values:
        raise RuntimeError("Radius-conditioning graph has no gradients")
    finite = bool(torch.stack([torch.isfinite(value).all() for value in values]).all().item())
    maximum = float(torch.stack([value.abs().max() for value in values]).max().item())
    if not finite:
        raise RuntimeError("Nonfinite radius-conditioning gradient")
    return {"applicable": True, "finite": finite, "nonzero": maximum > 0,
            "maximum_absolute": maximum, "tensors_with_gradients": len(values)}


def reveal_zero_initialized_graph(model, device):
    """Fixture-only edit; no optimizer, fitting, or retained model artifact."""
    changed = []
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(72919)
        if model.kind == "audio_unet":
            nn.init.normal_(model.backbone.out.weight, std=0.01)
            changed.append("backbone.out.weight")
        elif model.kind == "image_sit":
            nn.init.normal_(model.backbone.final_layer.linear.weight, std=0.01)
            nn.init.normal_(model.backbone.final_layer.adaLN_modulation[-1].weight, std=0.01)
            changed.extend(["backbone.final_layer.linear.weight",
                            "backbone.final_layer.adaLN_modulation.1.weight"])
    return changed


def check_arm(cfg, dim, arm, device):
    x_cpu, target_cpu, stats = fixture(dim)
    x, target = x_cpu.to(device), target_cpu.to(device)
    times = torch.tensor([0.2, 0.6], device=device)
    labels = None if cfg["kind"] == "mlp" else torch.tensor([0, 7], device=device)
    set_all_seeds(46021)
    model = build_input_model(cfg, dim, arm, stats).to(device)
    initial_state = state_fingerprint(model)
    modules = module_fingerprint(model)
    counts = model.parameter_report()
    torch.cuda.reset_peak_memory_stats(device)
    common._synchronize(device)
    begin = time.perf_counter()

    # Actual training dtype and original zero initialization, batch two only.
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["kind"] != "mlp"):
        prediction = model(x, times, labels).float()
        loss = ((prediction - target) ** 2).sum(dim=1).mean()
    if prediction.shape != x.shape or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Initial forward shape or finiteness failure")
    loss.backward()
    initial_gradient = gradients(model)
    initial_condition = condition_gradients(model)
    model.zero_grad(set_to_none=True)
    del prediction, loss

    # Expose the complete gradient graph independently of zero output heads.
    changed = reveal_zero_initialized_graph(model, device)
    model.eval()
    prediction = model(x, times, labels)
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("Nonfinite fixture-head forward")
    ((prediction - target) ** 2).sum(dim=1).mean().backward()
    revealed_gradient = gradients(model)
    revealed_condition = condition_gradients(model)
    if arm == "B" and not revealed_condition["nonzero"]:
        raise RuntimeError("B radius input has no nonzero gradient through the full backbone")
    model.zero_grad(set_to_none=True)
    del prediction

    with torch.no_grad():
        reference = model(x, times, labels)
        same_direction = model(2.0 * x, times, labels)
        if arm == "C" and not torch.equal(reference, same_direction):
            raise RuntimeError("C predictions changed under an exact power-of-two radius scaling")
        radius_effect = float((reference - same_direction).abs().max().item())
        if arm == "B" and radius_effect == 0:
            raise RuntimeError("B radius condition did not influence the full-backbone prediction")
        if arm == "A":
            if cfg["kind"] == "mlp":
                raw = model.backbone(x, times)
            else:
                raw = model.backbone(x.reshape(2, *model.event_shape), times, labels).reshape_as(x)
            if not torch.equal(raw, reference):
                raise RuntimeError("A wrapper differs from the original raw backbone")
        payload = model.checkpoint_payload()
        restored = build_input_model(cfg, dim, arm, stats).to(device).eval()
        restored.load_checkpoint_payload(payload)
        repeat = restored(x, times, labels)
        maximum_reload_error = float((reference - repeat).abs().max().item())
        if not torch.equal(reference, repeat):
            raise RuntimeError(f"Strict checkpoint reload changed output: max error {maximum_reload_error}")

    common._synchronize(device)
    row = {"arm": arm, "status": "passed", "dimension": dim, "batch_size": 2,
        "model_config": cfg, "parameters": counts,
        "initial_state_sha256": initial_state, "module_structure_sha256": modules,
        "original_initialization_forward_backward": {
            "precision": "float32" if cfg["kind"] == "mlp" else "bfloat16_autocast",
            "backbone_gradients": initial_gradient, "condition_gradients": initial_condition},
        "fixture_head_graph_check": {"changed_parameters": changed, "optimizer_updates": 0,
            "backbone_gradients": revealed_gradient, "condition_gradients": revealed_condition,
            "radius_scale": 2.0, "maximum_prediction_change": radius_effect,
            "c_radius_invariant": True if arm == "C" else None},
        "strict_checkpoint_reload": {"passed": True, "maximum_output_error": maximum_reload_error,
                                     "file_written": False},
        "original_a_forward_identical": True if arm == "A" else None,
        "radius_statistics": stats.to_dict(), "model_metadata": model.checkpoint_metadata(),
        "elapsed_s": time.perf_counter() - begin, "peak_memory": common.peak_memory(device),
        "timing_kind": "measured", "benchmark_data_used": False,
        "optimizer_updates": 0, "numerical_solver_steps": 0}
    del payload, restored, model, reference, repeat, same_direction, x, target
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs_rafm_input_study/v1/full_backbone_checks.json")
    parser.add_argument("--sit-source", default="/mnt/vast01/users/fouad.oubari/references/SiT-tflow")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Preserve previous interface-check record: {output}")
    result = {"schema_version": 1, "status": "running", "benchmark_data_used": False,
              "optimizer_updates": 0, "numerical_solver_steps": 0, "rows": [],
              "script_sha256": sha256(Path(__file__).resolve()), "hardware": None}
    try:
        device = common.compute_device()
        result["hardware"] = common.hardware()
        configs = [
            ({"kind": "mlp", "hidden_dim": 128, "n_layers": 3}, 256),
            ({"kind": "audio_unet", "event_shape": [2, 129, 63], "ch": 96,
              "mult": [1, 2, 4], "cond": 256, "attn_from": 2, "num_classes": 10}, 16254),
            ({"kind": "image_sit", "event_shape": [32, 8, 8], "hidden": 384,
              "depth": 12, "heads": 6, "num_classes": 10, "class_dropout": 0.1,
              "source_dir": str(Path(args.sit_source).resolve())}, 2048),
        ]
        for cfg, dim in configs:
            group = []
            for arm in ARMS:
                print(f"Checking actual {cfg['kind']} arm {arm}, dim={dim}, batch=2", flush=True)
                row = check_arm(cfg, dim, arm, device)
                result["rows"].append(row)
                group.append(row)
                common.write_json(output, result)
            b, c = group[1], group[2]
            for key in ("total_parameters", "conditioning_overhead_parameters", "trainable_parameters"):
                if b["parameters"][key] != c["parameters"][key]:
                    raise RuntimeError(f"B/C parameter mismatch for {cfg['kind']}: {key}")
            if b["initial_state_sha256"] != c["initial_state_sha256"]:
                raise RuntimeError(f"B/C initial tensor mismatch for {cfg['kind']}")
            if b["module_structure_sha256"] != c["module_structure_sha256"]:
                raise RuntimeError(f"B/C module mismatch for {cfg['kind']}")
        result["status"] = "passed"
        result["b_c_counts_modules_and_initial_weights_identical"] = True
        common.write_json(output, result)
        print(json.dumps({"status": result["status"], "checks": len(result["rows"]), "output": str(output)}), flush=True)
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        common.write_json(output, result)
        raise


if __name__ == "__main__":
    main()
