# Recovered PIV32 and ImageNette continuation

The user authorized retrieval, verification, short checks, and the remaining
RAFM-Ang A/B/C training/evaluation jobs on 2026-09-10. This continuation adds
18 final jobs to the existing study; it does not rerun valid completed jobs or
start additional t-Flow tuning. User-facing times use Europe/Paris.

## Verified input provenance

`release_verification.json` verifies every downloaded asset, including both
manifests, the audit report, code and logs, against authenticated GitHub API
SHA-256 digests. Manifest and addendum checksums are checked independently.
The existing scaled latent and label files match both the original data-v1
pins and the authenticated release digests. Missing files or any mismatch abort.
Negative checks are recorded in `verification_checks.json`.

The assets are under
`/mnt/vast01/users/fouad.oubari/data/rafm_input_study/releases/piv-imagenette-provenance-v1_20260910/`.
CPU Slurm job 765953 independently verified both recovered conditions:

- PIV32 is the original finite contiguous 998×32 tensor, not a truncation of
  PIV64 or PIV256. The original full-data centering and CPU torch seed-0
  598/199/201 split are preserved.
- ImageNette's saved mean and complete centered tensor are bitwise identical
  to the unchanged loader applied to the original scaled latents. Saved PT
  indices, NPZ indices and CPU torch seed-0 permutation agree exactly:
  8,036/2,678/2,680 generator training/validation/test rows.
- All 13,394 filenames, row indices, class labels and archived re-encoding
  matches agree. The supplied audit records an identity bijection and zero
  maximum L2 distance. Re-encoding was not repeated here, and no images or
  latent datasets were regenerated.

The exact historical extraction/training commit remains unverified.
`d3006dc8…` identifies the checkout holding the artifacts, not a verified
training commit. The provenance code bundle and each member's hash are retained.

## FID reference and interpretation

The original 3,925 PNG bytes and complete reference digest remain unchanged.
The supplied audit reports overlap with the generator split: **2,339 training,
775 validation, and 811 test images**. The row-map flags, declared NumPy seed-1
selection and intersections with saved generator indices were checked here.
PNG-to-original-JPEG identity was not independently re-encoded on this cluster;
that attribution comes from the VM audit and supplied mapping.

This reference is not held out from generator training. The user explicitly
authorized retaining it for the present matched A/B/C comparison. No reference
replacement or main manuscript revision is made in this continuation.

## Unchanged experiments

PIV32 uses the existing three-hidden-layer width-128 MLP, 10,000 updates,
batch 256, Adam LR 0.001 and 512 actual network evaluations. Seeds are
8925, 77395 and 65457.

ImageNette uses the pinned SiT backbone (hidden384, depth12, six heads),
40,000 updates, batch64, AdamW LR0.0001, EMA0.9999 and the existing bfloat16
training autocast. Each final model generates 3,000 balanced samples with
sampling seed0 and 100 actual evaluations under the agreed ambient sampler.
Training seeds are 8925, 1234 and 7. Exact settings live in
`configs/rafm_input_study_recovered/prepared/`.

Model implementations, losses, coupling, normalization safeguards, and ambient
sampling code are unchanged. B and C retain identical modules/parameter counts;
C receives the constant radius input. Radius normalization statistics use only
the generator training split.

## Orchestration and results

The original 26-condition launch/configuration files remain immutable.
`configs/rafm_input_study_complete/` combines byte-identical copies of those
26 prepared configurations with the two verified recovered configurations.
The additive launcher uses the original per-task executor, exclusive run locks,
sample auditor, and full source/configuration checksum gates.

All six full-batch A/B/C smoke checks and the offline image evaluator check
passed in GPU job **765955**. Image training peak allocations were 1.72 GiB
for A and 2.34 GiB for B/C. The 40-validation-row image evaluator used the
complete original 3,925-image reference and finished in 197.7 seconds, with
9.30 GiB peak allocation. These are checks, not generator benchmark results.

Array **765988** was submitted at **16:17 Europe/Paris on 2026-09-10**, with
six single-GPU MI210 workers on `auh7-3b-gpu-029`. Its dependency on existing
ABC array 765748 retains the six-GPU study concurrency limit. The original
pending submission 765979 targeted node008, which another user occupied during
the final checks. Slurm accounting confirms it was cancelled before starting:
zero runtime and no assigned node. The relocation record and both immutable
submission manifests are preserved; no experiment was interrupted or duplicated.

The final submitted manifest is
`outputs_rafm_input_study/recovered/launch/tasks_20260910T141752_da3e0210.json`,
SHA-256 `76cbb3534bc5182bc65e79580afbe344980ea0c52f2bddee8763905dc1200546`.
Its source commit is `4064f66`; experimental source bytes remain unchanged
from the original study. CPU-only aggregation/plot job **765992** depends on
the recovered array ending. It reports all 252 prescribed A/B/C outcomes
separately from the existing t-Flow reference results and missing slots.

Conservative planning from the short checks is 1–2 hours training and
15–30 minutes sampling/evaluation per image model, or about 2.5–5 hours
for the two image waves after the array starts. These are estimates;
the first full-run logs will provide measured throughput. Neighboring PIV
configurations took 30–38 seconds to train, with additional metric time.

Final results continue under
`outputs_rafm_input_study/v1/final/<condition>/{A,B,C}/seed_<seed>/`.
Recovered launch/check logs are under `outputs_rafm_input_study/recovered/`.
Consolidated output uses `outputs_rafm_input_study/complete_report/`.
Only three complete, finite, matching seeds acquire a mean and standard
deviation; failures remain explicit and are not averaged away.
