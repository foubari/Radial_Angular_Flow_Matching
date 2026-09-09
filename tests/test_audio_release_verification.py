"""Small stdlib-only release fixtures; no real checkpoint or model execution."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "audio_release_verifier", Path(__file__).resolve().parents[1] / "tools/verify_audio_release.py")
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


class ReleaseVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "downloaded"
        self.artifacts.mkdir()
        self.release_path = self.root / "release_api.json"
        self.data_release_path = self.root / "data_release_api.json"
        self.suite_path = self.root / "suite_manifest.json"
        self.report_path = self.root / "checks/report.json"
        self.manifest = {
            "method": "fixed_spherical", "dataset": "AudioMNIST (STFT, D=16254)",
            "arch": "unet ch96 depth5", "steps": 24000, "seeds": [8925, 7, 1234],
            "eval_checkpoint": "ema_24000 (EMA 0.999)",
            "source_repo": "github.com/foubari/radial_angular_FM",
            "source_branch": "iclr-image-experiments", "source_commit": verifier.ARTIFACT_COMMIT,
            "source_commit_note": "meta.json did not record a training commit; current artifact checkout only.",
            "assets": ["manifest.json"], "entries": [],
        }
        for seed in verifier.SEEDS:
            args = {"method": "fixed_spherical", "out": f"runs/seed{seed}",
                    "data": "experiments/poc_audio/data/audiomnist_stft_train.pt",
                    "steps": 24000, "batch": 32, "arch": "unet", "ch": 96, "depth": 5,
                    "lr": 0.0002, "ckpt_every": 2000, "log_every": 500, "ema": 0.999,
                    "class_dropout": 0.1, "seed": seed, "split_seed": 0}
            if seed != 8925:
                args["angular"] = False
            metadata = {"args": args, "n_params": 27896802, "R0": 2.0,
                        "sigma": 2.0 / math.sqrt(16254), "D": 16254}
            checkpoint = self.artifacts / f"ema_24000_seed{seed}.pt"
            checkpoint.write_bytes(f"FAKE checkpoint identity {seed}".encode())
            meta = self.artifacts / f"meta_seed{seed}.json"
            log = self.artifacts / f"train_log_seed{seed}.json"
            verifier.write_json(meta, metadata)
            verifier.write_json(log, [{"step": step, "loss": 1.25} for step in range(500, 24001, 500)])
            self.manifest["assets"].extend([checkpoint.name, meta.name, log.name])
            self.manifest["entries"].append({
                "seed": seed,
                "checkpoint": {"file": checkpoint.name, "step": 24000, "kind": "ema",
                               "bytes": checkpoint.stat().st_size, "sha256": verifier.sha256(checkpoint)},
                "config": {"file": meta.name, "sha256": verifier.sha256(meta), **metadata},
                "train_log": {"file": log.name, "sha256": verifier.sha256(log)},
            })
        self.release = self.api_document(verifier.CHECKPOINT_TAG)
        self.refresh_manifest_and_api()
        known_data, inputs = {}, {}
        self.data_release = self.api_document(verifier.DATA_TAG)
        for key, (name, _) in verifier.KNOWN_DATA.items():
            path = self.root / name
            path.write_bytes(f"FAKE small cached data {name}".encode())
            digest = verifier.sha256(path)
            known_data[key] = (name, digest)
            inputs[key] = {"path": str(path), "sha256": digest}
            self.data_release["assets"].append(self.api_asset(path))
        verifier.write_json(self.data_release_path, self.data_release)
        verifier.write_json(self.suite_path, {"inputs": inputs})
        self.known_hash_patch = patch.object(verifier, "KNOWN_DATA", known_data)
        self.known_hash_patch.start()
        self.addCleanup(self.known_hash_patch.stop)

    @staticmethod
    def api_document(tag):
        return {"tagName": tag, "url": f"https://github.com/{verifier.RELEASE_REPOSITORY}/releases/tag/{tag}",
                "publishedAt": "2026-09-09T22:10:18Z", "assets": []}

    @staticmethod
    def api_asset(path):
        return {"name": path.name, "size": path.stat().st_size,
                "digest": "sha256:" + verifier.sha256(path), "state": "uploaded"}

    def refresh_manifest_and_api(self):
        verifier.write_json(self.artifacts / "manifest.json", self.manifest)
        self.release["assets"] = [self.api_asset(self.artifacts / name) for name in self.manifest["assets"]]
        verifier.write_json(self.release_path, self.release)

    def verify(self):
        return verifier.verify_release(self.artifacts, self.release_path, self.data_release_path,
                                       self.suite_path, self.report_path)

    def assert_failure_report(self, message):
        with self.assertRaisesRegex(verifier.VerificationError, message):
            self.verify()
        report = json.loads(self.report_path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["blocking_issues"])
        self.assertIsNone(report["training_commit"])
        return report

    def test_all_thirteen_files_verify_without_inventing_training_commit(self):
        before = {path.name: path.read_bytes() for path in self.artifacts.iterdir()}
        report = self.verify()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["verified_release_asset_count"], 10)
        self.assertEqual(report["verified_audio_data_asset_count"], 3)
        self.assertIsNone(report["training_commit"])
        self.assertEqual(report["artifact_checkout"]["commit"], verifier.ARTIFACT_COMMIT)
        self.assertNotIn("angular", report["metadata"]["8925"]["args"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.artifacts.iterdir()})

    def test_missing_checkpoint_writes_failure_and_cli_returns_nonzero(self):
        (self.artifacts / "ema_24000_seed8925.pt").unlink()
        report = self.assert_failure_report("Missing required file")
        self.assertFalse(report["files"]["ema_24000_seed8925.pt"]["exists"])
        with contextlib.redirect_stderr(io.StringIO()):
            code = verifier.main(["--artifact-dir", str(self.artifacts), "--release-api", str(self.release_path),
                                  "--data-release-api", str(self.data_release_path),
                                  "--suite-manifest", str(self.suite_path), "--output", str(self.report_path)])
        self.assertEqual(code, 1)

    def test_same_size_checkpoint_corruption_is_detected(self):
        path = self.artifacts / "ema_24000_seed8925.pt"
        path.write_bytes(b"X" + path.read_bytes()[1:])
        report = self.assert_failure_report("SHA256 mismatch")
        self.assertEqual(report["files"][path.name]["status"], "failed")

    def test_missing_api_digest_blocks_even_when_manifest_has_hash(self):
        next(asset for asset in self.release["assets"] if asset["name"] == "meta_seed8925.json").pop("digest")
        verifier.write_json(self.release_path, self.release)
        self.assert_failure_report("Missing SHA256 API digest")

    def test_manifest_itself_must_match_api(self):
        with (self.artifacts / "manifest.json").open("a") as stream:
            stream.write(" ")
        self.assert_failure_report("size mismatch for manifest.json")

    def test_api_size_must_agree_with_manifest(self):
        next(asset for asset in self.release["assets"] if asset["name"] == "ema_24000_seed8925.pt")["size"] += 1
        verifier.write_json(self.release_path, self.release)
        self.assert_failure_report("API and manifest sizes disagree")

    def test_metadata_must_equal_embedded_configuration_after_all_hashes_match(self):
        entry = self.manifest["entries"][0]
        path = self.artifacts / entry["config"]["file"]
        metadata = copy.deepcopy({key: entry["config"][key] for key in ("args", "n_params", "R0", "sigma", "D")})
        metadata["args"]["batch"] = 64
        verifier.write_json(path, metadata)
        entry["config"]["sha256"] = verifier.sha256(path)
        self.refresh_manifest_and_api()
        self.assert_failure_report("Metadata contents differ")

    def test_data_release_digest_must_match_independent_known_hash(self):
        self.data_release["assets"][0]["digest"] = "sha256:" + "0" * 64
        verifier.write_json(self.data_release_path, self.data_release)
        self.assert_failure_report("API digest differs from the known cache hash")

    def test_missing_data_api_file_saves_explicit_block(self):
        self.data_release_path.unlink()
        report = self.assert_failure_report("data_release_api.json")
        self.assertEqual(report["error_type"], "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
