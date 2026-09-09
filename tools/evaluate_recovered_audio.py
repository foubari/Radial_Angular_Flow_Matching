"""Run only the authorized, checksum-verified fixed-spherical gain evaluation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "inputs/recovered_baselines"
AUDIT = ROOT / "outputs_audio_gain/release_verification"
OUTPUT = Path(os.environ.get("AUDIO_GAIN_OUTPUT_DIR", ROOT / "outputs_audio_gain/fixed_spherical_empirical_gain_v1"))
DATA = ROOT.parent / "msgm-sparse-control/data/experiments/poc_audio"


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This command requires a compute-node Slurm allocation")
    started = time.perf_counter()
    status = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "scope": "three original fixed-spherical checkpoints and independent post-hoc empirical gain only",
              "training_or_tflow_launched": False,
              "output": str(OUTPUT)}
    try:
        from tools.verify_audio_release import verify_release
        verify_release(ARTIFACTS, AUDIT / "release_api.json", AUDIT / "data_release_api.json",
                       ROOT / "configs/tflow/suite_manifest.json", AUDIT / "strict_verification_compute.json")
        source = json.loads((AUDIT / "source_compatibility.json").read_text())
        # The audit must explicitly leave training provenance unverified.
        provenance = source["provenance"]
        if (source.get("status") != "passed" or provenance.get("training_commit") is not None
                or provenance.get("training_commit_verified") is not False):
            raise ValueError("Invalid artifact-source audit or unexpected training-commit claim")
        from experiments.poc_audio import audio_empirical_gain as audio
        for record in source["source_files"]:
            if (record["byte_identical"] is not True or record["artifact_checkout_sha256"] != record["worktree_sha256"]
                    or audio.file_sha256(ROOT / record["path"]) != record["worktree_sha256"]):
                raise ValueError("Source changed since compatibility audit: " + record["path"])
        argv = ["evaluate", "--train-file", str(DATA / "data/audiomnist_stft_train.pt"),
                "--test-file", str(DATA / "data/audiomnist_stft_test.pt"),
                "--classifier", str(DATA / "digit_classifier.pt"),
                "--reference-aggregate", str(ROOT / "experiments/poc_audio/stage2_3seed.json"),
                "--output-dir", str(OUTPUT)]
        for seed in audio.SEEDS:
            argv += ["--run", str(seed), str(ARTIFACTS / f"ema_24000_seed{seed}.pt"),
                     str(ARTIFACTS / f"meta_seed{seed}.json")]
        args = audio.parser().parse_args(argv)
        shared, runs, reference = audio.preflight(args)
        status["argv"] = argv
        status["evaluator_sha256"] = audio.file_sha256(Path(audio.__file__))
        status["provenance"] = provenance
        status["source_compatibility_sha256"] = audio.file_sha256(AUDIT / "source_compatibility.json")
        status["status"] = "running"
        (AUDIT / "execution_status.json").write_text(json.dumps(status, indent=2) + "\n")
        audio.evaluate(args, shared, runs, reference)
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        status["elapsed_wall_s"] = time.perf_counter() - started
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        (AUDIT / "execution_status.json").write_text(json.dumps(status, indent=2) + "\n")
        (AUDIT / f"execution_job_{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
