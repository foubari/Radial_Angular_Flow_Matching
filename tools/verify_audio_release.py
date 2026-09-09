#!/usr/bin/env python3
"""Strict offline release verification using only Python's standard library.

Checks bytes against the saved authenticated GitHub CLI release responses. It
does not deserialize checkpoints, import torch, or establish a training commit.
Missing evidence is a blocking error, never a reason to infer default metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_REPOSITORY = "foubari/msgm-sparse-control"
CHECKPOINT_TAG = "fixed-spherical-audiomnist-v1"
DATA_TAG = "data-v1"
ARTIFACT_COMMIT = "d3006dc8ee61b2ab6e470115d3f21c778fefcf81"
SEEDS = (8925, 1234, 7)
STEPS = 24000
# These hashes are independently recorded in configs/tflow/suite_manifest.json
# and the cached-input audit. The release digest must agree with both pins.
KNOWN_DATA = {
    "audio_train": ("audiomnist_stft_train.pt", "d4c0e10dcae8bdb2ad6e7bf05202a42e6a6af3b6fb064e6fc97e12432e0fe7be"),
    "audio_test": ("audiomnist_stft_test.pt", "c9ab95b3bcb0049f881a84d180bccbdedf4d25cf6723dc48cafb90e4a0e20209"),
    "audio_classifier": ("digit_classifier.pt", "1c68713e65df8c256bf8dd73edee70ba0e419381258cc46ff497042287946d50"),
}


class VerificationError(RuntimeError):
    """A required identity, byte checksum, or metadata check failed."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise VerificationError(f"Nonfinite JSON constant: {value}")


def _finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON numeric value")
    elif isinstance(value, dict):
        for child in value.values():
            _finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _finite_json(child)


def read_json(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    _finite_json(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value, description):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value),
            f"Missing or invalid SHA256 for {description}")
    return value


def _filename(value):
    require(isinstance(value, str) and value not in (".", "..")
            and re.fullmatch(r"[A-Za-z0-9_.-]+", value),
            f"Unsafe or absent asset filename: {value!r}")
    return value


def _positive_integer(value, description):
    require(type(value) is int and value > 0, f"Invalid positive integer for {description}")
    return value


def _release_index(document, tag):
    # This explicitly supports the observed gh release view --json schema.
    require(isinstance(document, dict), f"Release {tag} response must be a JSON object")
    require(document.get("tagName") == tag, f"Missing or unexpected release tagName: {tag}")
    expected_url = f"https://github.com/{RELEASE_REPOSITORY}/releases/tag/{tag}"
    require(document.get("url") == expected_url, f"Release URL does not identify {expected_url}")
    require(isinstance(document.get("publishedAt"), str) and document["publishedAt"],
            f"Release {tag} lacks publication evidence")
    if "isDraft" in document:
        require(document["isDraft"] is False, f"Release {tag} is a draft")
    if "isPrerelease" in document:
        require(document["isPrerelease"] is False, f"Release {tag} is a prerelease")
    assets = document.get("assets")
    require(isinstance(assets, list) and assets, f"Release {tag} has no assets array")
    indexed = {}
    for asset in assets:
        require(isinstance(asset, dict), f"Invalid asset record in release {tag}")
        name = _filename(asset.get("name"))
        require(name not in indexed, f"Duplicate release asset: {name}")
        indexed[name] = asset
    return indexed


def _verify_file(path, asset, records, *, manifest_sha=None, manifest_bytes=None,
                 known_sha=None):
    name = _filename(asset.get("name"))
    record = {"path": str(Path(path).absolute()), "status": "checking", "asset_id": asset.get("id"),
              "api_url": asset.get("apiUrl"), "download_url": asset.get("url"),
              "api_digest": asset.get("digest"), "api_bytes": asset.get("size")}
    records[name] = record
    require(asset.get("state") == "uploaded", f"Asset {name} is not uploaded")
    digest = asset.get("digest")
    require(isinstance(digest, str) and digest.startswith("sha256:"),
            f"Missing SHA256 API digest for {name}; verification is blocked")
    api_sha = _digest(digest[7:], f"API asset {name}")
    api_bytes = _positive_integer(asset.get("size"), f"API asset {name} size")
    if manifest_sha is not None:
        record["manifest_sha256"] = _digest(manifest_sha, f"manifest entry {name}")
        require(api_sha == manifest_sha, f"API and manifest SHA256 disagree for {name}")
    if manifest_bytes is not None:
        record["manifest_bytes"] = _positive_integer(manifest_bytes, f"manifest entry {name} size")
        require(api_bytes == manifest_bytes, f"API and manifest sizes disagree for {name}")
    if known_sha is not None:
        record["known_sha256"] = _digest(known_sha, f"independently pinned {name}")
        require(api_sha == known_sha, f"API digest differs from the known cache hash for {name}")
    path = Path(path)
    record["exists"] = path.is_file()
    require(record["exists"], f"Missing required file: {path}")
    before = path.stat()
    record["actual_bytes"] = before.st_size
    require(before.st_size == api_bytes, f"File size mismatch for {name}: {before.st_size} != {api_bytes}")
    record["actual_sha256"] = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f"File changed during verification: {name}")
    require(record["actual_sha256"] == api_sha, f"SHA256 mismatch for {name}")
    record["status"] = "verified"
    return record


def _validate_manifest(manifest):
    require(isinstance(manifest, dict), "Manifest must be a JSON object")
    require(manifest.get("method") == "fixed_spherical", "Unexpected manifest method")
    require(manifest.get("dataset") == "AudioMNIST (STFT, D=16254)", "Unexpected manifest dataset")
    require(manifest.get("arch") == "unet ch96 depth5", "Unexpected manifest architecture")
    require(manifest.get("steps") == STEPS, "Manifest must require 24000 steps")
    require(manifest.get("eval_checkpoint") == "ema_24000 (EMA 0.999)", "Unexpected evaluation checkpoint")
    seeds = manifest.get("seeds")
    require(isinstance(seeds, list) and all(type(seed) is int for seed in seeds)
            and len(seeds) == 3 and set(seeds) == set(SEEDS), "Manifest must contain exactly the three original seeds")
    require(manifest.get("source_repo") == "github.com/foubari/radial_angular_FM", "Unexpected artifact source repository")
    require(manifest.get("source_branch") == "iclr-image-experiments", "Unexpected artifact checkout branch")
    require(manifest.get("source_commit") == ARTIFACT_COMMIT, "Unexpected artifact checkout commit")
    note = manifest.get("source_commit_note")
    require(isinstance(note, str) and "did not record a training commit" in note,
            "Manifest must retain its statement that the training commit was not recorded")
    expected_assets = {"manifest.json"} | {
        name for seed in SEEDS for name in
        (f"ema_24000_seed{seed}.pt", f"meta_seed{seed}.json", f"train_log_seed{seed}.json")}
    names = manifest.get("assets")
    require(isinstance(names, list) and len(names) == len(expected_assets)
            and set(names) == expected_assets, "Manifest asset list must contain exactly the ten required files")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == 3, "Manifest requires three seed entries")
    by_seed = {}
    for entry in entries:
        require(isinstance(entry, dict), "Invalid manifest entry")
        seed = entry.get("seed")
        require(type(seed) is int and seed in SEEDS and seed not in by_seed, "Invalid or duplicated manifest entry seed")
        by_seed[seed] = entry
        checkpoint, config, log = (entry.get(key) for key in ("checkpoint", "config", "train_log"))
        require(all(isinstance(value, dict) for value in (checkpoint, config, log)), f"Seed {seed} has missing entry schemas")
        require(checkpoint.get("file") == f"ema_24000_seed{seed}.pt" and checkpoint.get("step") == STEPS
                and checkpoint.get("kind") == "ema", f"Seed {seed} does not identify the required EMA checkpoint")
        _positive_integer(checkpoint.get("bytes"), f"seed {seed} checkpoint bytes")
        require(config.get("file") == f"meta_seed{seed}.json", f"Incorrect metadata filename for seed {seed}")
        require(log.get("file") == f"train_log_seed{seed}.json", f"Incorrect training log filename for seed {seed}")
        for item in (checkpoint, config, log):
            _digest(item.get("sha256"), f"manifest entry {item['file']}")
        require(set(config) == {"file", "sha256", "args", "n_params", "R0", "sigma", "D"},
                f"Missing or unexpected embedded configuration fields for seed {seed}")
        require(type(config["n_params"]) is int and config["n_params"] == 27896802
                and type(config["D"]) is int and config["D"] == 16254, f"Unexpected model size/dimension for seed {seed}")
        for key in ("R0", "sigma"):
            require(type(config[key]) in (int, float) and config[key] > 0, f"Invalid {key} for seed {seed}")
        require(math.isclose(config["sigma"], config["R0"] / math.sqrt(config["D"]), rel_tol=1e-12),
                f"Source sigma disagrees with R0/sqrt(D) for seed {seed}")
        args = config["args"]
        expected = {"method": "fixed_spherical", "steps": STEPS, "batch": 32, "arch": "unet", "ch": 96,
                    "depth": 5, "lr": 0.0002, "ckpt_every": 2000, "log_every": 500, "ema": 0.999,
                    "class_dropout": 0.1, "seed": seed, "split_seed": 0,
                    "data": "experiments/poc_audio/data/audiomnist_stft_train.pt"}
        require(isinstance(args, dict) and all(key in args and args[key] == value for key, value in expected.items()),
                f"Embedded training arguments do not match the fixed-spherical protocol for seed {seed}")
        require(isinstance(args.get("out"), str) and args["out"], f"Missing recorded output path for seed {seed}")
        # Seed 8925 genuinely omits this field. Preserve that omission, and do
        # not add a fabricated default to metadata or the verification report.
        if "angular" in args:
            require(args["angular"] is False, f"Unexpected angular objective for seed {seed}")
    return expected_assets, by_seed


def verify_release(artifact_dir, release_api_path, data_release_api_path,
                   suite_manifest_path, report_path):
    """Write a complete success report, or a failure report and raise.

    API files must be the saved authenticated responses. This function performs
    no network requests and never writes to the downloaded artifact directory.
    """
    artifact_dir = Path(artifact_dir).resolve()
    report_path = Path(report_path).resolve()
    evidence_paths = [Path(path).resolve() for path in
                      (release_api_path, data_release_api_path, suite_manifest_path)]
    require(not report_path.is_relative_to(artifact_dir) and report_path not in evidence_paths,
            "Verification output must not replace downloaded assets or input evidence")
    report = {"schema_version": 1, "status": "checking", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "verification_scope": "Offline byte/API/manifest/config consistency; no checkpoint deserialization or model execution",
              "artifact_dir": str(artifact_dir), "verifier_sha256": sha256(__file__),
              "artifact_checkout": None, "training_commit": None,
              "training_commit_status": "not recorded in released training metadata",
              "release_evidence": {}, "files": {}, "data_files": {}, "metadata": {}, "blocking_issues": []}
    try:
        release, data_release, suite = [read_json(path) for path in evidence_paths]
        for key, path, document in zip(("checkpoint_release_api", "data_release_api", "suite_manifest"),
                                       evidence_paths, (release, data_release, suite)):
            report["release_evidence"][key] = {"path": str(path), "sha256": sha256(path)}
            if key != "suite_manifest":
                report["release_evidence"][key].update(tag=document.get("tagName"), url=document.get("url"),
                                                       published_at=document.get("publishedAt"))
        assets = _release_index(release, CHECKPOINT_TAG)
        data_assets = _release_index(data_release, DATA_TAG)
        require("manifest.json" in assets, "Checkpoint API response lacks manifest.json")
        _verify_file(artifact_dir / "manifest.json", assets["manifest.json"], report["files"])
        manifest = read_json(artifact_dir / "manifest.json")
        names, entries = _validate_manifest(manifest)
        require(set(assets) == names, "Checkpoint API and manifest asset sets differ; missing or unlisted assets block verification")
        report["artifact_checkout"] = {"commit": manifest["source_commit"], "repository": manifest["source_repo"],
                                       "branch": manifest["source_branch"], "note": manifest["source_commit_note"]}
        report["manifest"] = {"path": str(artifact_dir / "manifest.json"),
                              "sha256": report["files"]["manifest.json"]["actual_sha256"],
                              "method": manifest["method"], "seeds": manifest["seeds"], "steps": manifest["steps"]}
        for seed in SEEDS:
            entry = entries[seed]
            for role in ("checkpoint", "config", "train_log"):
                item = entry[role]
                _verify_file(artifact_dir / item["file"], assets[item["file"]], report["files"],
                             manifest_sha=item["sha256"],
                             manifest_bytes=item["bytes"] if role == "checkpoint" else None)
            config = entry["config"]
            embedded = {key: config[key] for key in ("args", "n_params", "R0", "sigma", "D")}
            metadata = read_json(artifact_dir / config["file"])
            require(json.dumps(metadata, sort_keys=True) == json.dumps(embedded, sort_keys=True),
                    f"Metadata contents differ from the embedded manifest configuration for seed {seed}")
            log = read_json(artifact_dir / entry["train_log"]["file"])
            require(isinstance(log, list) and len(log) == STEPS // config["args"]["log_every"],
                    f"Training log length does not cover the recorded budget for seed {seed}")
            expected_steps = list(range(config["args"]["log_every"], STEPS + 1, config["args"]["log_every"]))
            require(all(isinstance(row, dict) for row in log)
                    and [row.get("step") for row in log] == expected_steps,
                    f"Training log steps do not run through 24000 for seed {seed}")
            require(all(type(row.get("loss")) in (int, float) and row["loss"] >= 0 for row in log),
                    f"Invalid recorded training loss for seed {seed}")
            report["metadata"][str(seed)] = {"status": "verified", "exact_embedded_config_match": True,
                                            **metadata, "log_rows": len(log), "last_logged_step": log[-1]["step"]}
        require(isinstance(suite, dict) and isinstance(suite.get("inputs"), dict), "Suite manifest lacks pinned inputs")
        for key, (name, known_sha) in KNOWN_DATA.items():
            require(name in data_assets, f"Data release lacks required audio asset: {name}")
            spec = suite["inputs"].get(key)
            require(isinstance(spec, dict), f"Suite manifest lacks {key}")
            require(spec.get("sha256") == known_sha, f"Suite manifest hash differs from the known cache hash for {key}")
            require(isinstance(spec.get("path"), str) and Path(spec["path"]).is_absolute()
                    and Path(spec["path"]).name == name, f"Missing or invalid absolute cached path for {key}")
            _verify_file(spec["path"], data_assets[name], report["data_files"], known_sha=known_sha)
        report.update(status="verified", required_seeds=list(SEEDS), checkpoint_step=STEPS,
                      verified_release_asset_count=len(report["files"]), verified_audio_data_asset_count=len(report["data_files"]))
        write_json(report_path, report)
        return report
    except Exception as error:
        for group in ("files", "data_files"):
            for record in report[group].values():
                if record["status"] == "checking":
                    record["status"] = "failed"
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["blocking_issues"].append(str(error))
        write_json(report_path, report)
        raise VerificationError(f"{error}; failure report saved to {report_path}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "inputs/recovered_baselines")
    parser.add_argument("--release-api", type=Path, default=REPO_ROOT / "outputs_audio_gain/release_verification/release_api.json")
    parser.add_argument("--data-release-api", type=Path, default=REPO_ROOT / "outputs_audio_gain/release_verification/data_release_api.json")
    parser.add_argument("--suite-manifest", type=Path, default=REPO_ROOT / "configs/tflow/suite_manifest.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs_audio_gain/release_verification/strict_verification.json")
    args = parser.parse_args(argv)
    try:
        report = verify_release(args.artifact_dir, args.release_api, args.data_release_api,
                                args.suite_manifest, args.output)
    except VerificationError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "release_assets": report["verified_release_asset_count"],
                      "audio_data_assets": report["verified_audio_data_asset_count"],
                      "artifact_checkout_commit": report["artifact_checkout"]["commit"],
                      "training_commit": None, "report": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
