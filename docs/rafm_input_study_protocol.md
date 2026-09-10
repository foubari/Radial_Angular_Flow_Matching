# Shared protocol for the RAFM-Ang input study and t-Flow

The user authorized the full study on 2026-09-10, after correctness checks. This document supersedes earlier launch and synthetic-cache prohibitions **only for this explicitly authorized study**. Published values and historical input files remain unchanged. A is original RAFM-Ang, B uses unit-direction input plus standardized log-radius conditioning, C uses the same B modules with constant zero radius conditioning, and t-Flow remains a separate published-method baseline.

The new study contains 28 unique dataset conditions, with three training seeds for each of A/B/C/t-Flow: 336 prescribed full training runs if all inputs become verifiable. Twenty synthetic conditions now have permission for **new, explicitly seeded, shared realizations**, because the original tensors were not recovered. Six existing input protocols can be reused. PIV d32 and ImageNette remain genuinely blocked; the 26 available conditions correspond to 312 full runs. No historical synthetic A row is a comparator on a newly generated realization: A is rerun on every new cache. The study also reruns A on existing inputs so comparisons share measured hardware and code environment.

`t-Flow` retains its documented validation-only tuning allocation: nine candidates at 5% training budget, then two candidates continued to 10% total, or 0.55 full-run equivalents per condition. That is additional compute, separate from the three final training seeds. B and C receive **zero additional tuning steps**. These run counts are not GPU-hour estimates.

## Prepared interface

The import-safe script is `tools/prepare_shared_study.py`. Planning imports only the standard library. The materialization action checks for a Slurm allocation **before importing Torch or NumPy**. Run it on a cluster compute node with the existing experiment environment:

```bash
PYTHONPATH="$PWD" ../msgm-sparse-control/.venv/bin/python \
  tools/prepare_shared_study.py --materialize
```

The `--condition ID` option can be repeated to prepare a subset. The default materializes the 26 verifiable conditions, while keeping the two blocked cases explicit. `--plan-only` creates only JSON plans and drafts and is safe on a login node.

Planning outputs are `configs/rafm_input_study/plan.json` and `drafts/<condition>.json`. Materialization produces `prepared/<condition>.json` only after input loading, finite-value checks, split validation, and hashing succeed. These retain the existing t-Flow configuration schema: `data.input` is a pinned tensor, and `data.split.kind="indices"` references a pinned dictionary of `train`, `val`, and `test` integer-index tensors. The original numerical sampler definitions remain method-specific: A/B/C retain the ambient RAFM RK4 sampler; t-Flow uses its own documented schedule and field at the same actual network-call budget. The preparation script does not implement or select any model or sampler.

Caches are stored beneath `/mnt/vast01/users/fouad.oubari/data/rafm_input_study/v1/<condition>/`. Each complete directory includes:

- `values.pt` for newly created synthetic inputs; verified existing input files retain their original locations and are not rewritten.
- `split_indices.pt`, transformed `train.pt`, `val.pt`, and `test.pt`, plus class-label files where applicable.
- `preprocessing_mean.pt`, and `generator_matrices.pt` where the original generator uses a mixing matrix.
- `manifest.json` with asset SHA-256 values, contiguous tensor SHA-256/MD5 values, exact indices and counts, generator/source hashes, generation and split seeds, environment, source commit, and Slurm job ID.

For audio, the frozen `test.pt` is the separate 3,000-example external test file; `split_indices.pt["test"]` is intentionally empty because the 12,000-example nominal training file is partitioned into generator train and internal validation only. Every original input, external-test and classifier checksum is rechecked. The manifest distinguishes these two index spaces explicitly.

A cache directory is published by atomic rename only after every check passes. Reruns verify all asset checksums and preparation identity and reuse the directory. Missing assets, changed bytes, or a different preparation identity cause a hard failure; a cache is never silently overwritten. Failed construction directories and an error JSON remain available for investigation. This script does not regenerate original historical files.

## Data and budget decisions

The supplied PDF was read directly on pages 23, 33, 34 and 35 to verify the figure, preprocessing, batch-size and downstream statements. In particular, p34 C.7 explicitly specifies **Student-t/PIV batch256, Weather2048, and other vector benchmarks4096**. The sparse-MSGM Weather batch4096 exception on p35 C.11 does not change the Flow Matching protocol.

| Conditions | Realization and split | Batch | Steps | Generated samples | Actual network calls |
|---|---|---:|---:|---:|---:|
| Student-t dimensions d2,8,16,32,64,128,256 at df3; d16 tail df1.5,2,3,5,10,50 | New shared tensor; 30,000/10,000/10,000 CPU Torch permutation, seed0 | 256 | 10,000 | 10,000 | 512 |
| Gaussian control d16 | New shared tensor; 30,000/10,000/10,000 permutation, seed0 | 4096 | 10,000 | 10,000 | 512 |
| Anisotropy d32, kappa1,3,10,30,100,300 | New shared tensors; contiguous 30,000/10,000/10,000 | 4096 | 10,000 | 10,000 | 512 |
| Toy2D | New shared tensor; 30,000/10,000/10,000 permutation, seed0 | 4096 | 10,000 | 10,000 | 512 |
| PIV d16,d64,d256 | Existing pinned inputs; 598/199/201 permutation, seed0 | 256 | 10,000 | 10,000 | 512 |
| Finance | Existing input; chronological 8,609/2,869/2,871 | 4096 | 10,000 | 2,871 | 512 |
| Weather | Existing input; chronological 3,508/1,169/1,170 | 2048 | 10,000 | 1,170 | 512 |
| AudioMNIST | Existing data-v1 files; generator 10,200/internal-val1,800 plus external-test3,000 | 32 | 24,000 | 2,000 balanced | 160 |
| PIV d32 — blocked | Verified native tensor is missing | 256 | 10,000 | 10,000 | 512 |
| ImageNette — blocked | Historical generator indices and row-to-image/reference disjointness are unresolved | 64 | 40,000 | 3,000 balanced | 100 |

Student-t d16/df3 is shared between dimension and tail sweeps, so the table lists twelve unique Student-t conditions. The user-authorized replacement comparison resolves the former 256-versus4096 Student-t ambiguity using the explicit paper protocol. This establishes a new matched comparison and does not assert that earlier unverified run configs used batch256. PIV d16 likewise uses the explicit paper batch256 and a newly run A arm; PIV d64/d256 already have archived completion metadata verifying that batch and their exact train/test tensor hashes.

All vector methods keep the original three-hidden-layer width128 MLP, Adam, learning rate0.001, betas(0.9,0.999), epsilon1e-8, zero weight decay, no EMA, no clipping, no augmentation and the final 10,000-step checkpoint. Downstream configs retain the existing U-Net/SiT dimensions, optimizer, learning rate, EMA, class dropout, batch and update counts. The new B/C conditioning overhead is reported by the model implementation; this script does not widen any backbone.

Vector training seeds are 8925/77395/65457; downstream seeds are 8925/1234/7. New synthetic comparisons explicitly use sampling seed0; existing PIV/Finance/Weather configurations retain their recorded `sample_seed:null` setting, whose post-training RNG semantics must be respected or separately disclosed by the execution runner. Audio retains sampling seed0. New metrics retain common seed0 projection directions, with the existing image evaluator's distinct seed2020; this does not assert recovery of unarchived historical projection arrays or MMD kernel widths.

## Exact synthetic generation and source laws

The new Student-t, Gaussian and toy realization seed is the first four bytes of SHA-256 of `rafm_input_parameterization_v1:20260910:<condition>`, interpreted big-endian and reduced modulo 2^31. The full integer is recorded in every draft and manifest. This seed is distinct from each model's training seed and from split seed0. Generation uses isolated CPU Torch RNG state and calls the existing dataset class; it does not change the target distribution.

For Student-t, the existing target is **independent univariate Student-t coordinates mixed by a fixed Gaussian matrix A**, sampled at matrix seed42. This target must not be confused with t-Flow's multivariate Student-t source, which uses one shared chi-square scale per example. The target df is recorded only as a target-generator parameter and never assigned automatically to the t-Flow prior. All twelve target conditions retain the paper's distribution, including df1.5/2 with nonfinite population variance.

The toy calls the existing four-mode Gaussian approximation to von Mises angles with concentration5 and half-Student-t(df3) radius. It does not replace that approximation with a different angular distribution. The anisotropy generator reproduces the *algorithm and RNG order* in `gen_aniso.py`: NumPy PCG64 seed42, fixed QR factors U/V, then one 50,000x32 Gaussian array per kappa in order1,3,10,30,100,300. Each kappa reconstruction advances past preceding arrays. Its bytes are still labelled a **new realization**, because seed replay across unverified environments is not proof of historical identity. U, V and A are archived.

Every explicit `angular_rafm` entry in E1, E5, E6 and E8 uses `radial_empirical_ecdf`. The neighbouring `rafm_oracle` rows are separate comparisons, not the original Angular method's source. `run_real.py`, PIV configs and downstream builders likewise use empirical radii for RAFM-Ang. The shared config field `rafm_sampling_source` therefore pins the interpolated training-radius ECDF for A/B/C in every condition. It must be fitted on generator training values only; radius standardization for B uses the same restricted split but does not change the sampling law. Matrices and target df remain archived for provenance and any separately authorized oracle diagnostic.

## Existing-data compatibility

PIV retains the original full-dataset recentering before the seed0 split because it is part of the published preprocessing. `load_data` checks each cached input hash, then computes exactly the original mean subtraction. The archived PIV64/PIV256 train/test MD5 values are checked again against the actual transformed splits; a mismatch aborts preparation. Finance and Weather stay in their committed raw coordinate systems and chronological order.

Audio loads the unchanged data-v1 files. Internal train/validation selection is `torch.randperm(12000, generator=torch.Generator().manual_seed(0))` on CPU, with the first10,200 rows used for the generator and the remaining1,800 for internal validation. External gains never enter radius fitting or conditioning-statistic fitting. The completed fixed-spherical+empirical-gain experiment used this same split, fixed classifier, 2,000 class-balanced samples, sampling seed0 and160 network calls. Its original comparison has a disclosed backend reproduction discrepancy; it remains a measured reference with its own runtime/environment record, not a substituted A checkpoint. The new audio comparison should report that the benchmark constructs gains independently of direction and digit, so a benefit for B over C is not presumed.

## Unresolved artifact blockers

**PIV d32:** no `piv_d32.pt` was found in the relevant msgm, data or reference directories, including ignored files, or in the tracked `iclr-image-experiments` tree/history. The PDF p33 specifies a native8x4 spatial grid. A first32-coordinate truncation of the native8x8/d64 or16x16/d256 tensor is a different grid; the backing storage of `piv_d16.pt` lacks an authoritative identity chain for the native8x4 tensor. Nothing is reconstructed from those files. Required resolution: the original native d32 tensor with retained-snapshot ordering and checksum, or authoritative provenance establishing the full underlying d32 storage's identity.

**ImageNette:** the exact scaled latents and labels are pinned, and all3,925 reference PNG bytes are verified. This establishes asset identity, not generator/reference disjointness. The extraction code sorts and concatenates official train+validation images (13,394 latents). The available SiT trainer applies a seed0 random60/20/20 split to all13,394 rows and records no row-to-image map. The PDF p35 says all3,925 official-validation reference images are disjoint from generator training. The source split cannot establish that condition. No new split is invented. Required resolution: the actual paper generator train/validation/test indices plus a row-to-image/official-split mapping or equivalent authoritative manifest; original per-seed configs/evals would also resolve the launcher3000-versus5000 sample-count history. The target new FM evaluation count remains the paper's3,000.

Authorized GitHub API queries on 2026-09-10 found no releases in `foubari/Radial_Angular_Flow_Matching`. `foubari/msgm-sparse-control` has `data-v1` (the six known data/reference assets) and `fixed-spherical-audiomnist-v1` (the ten verified checkpoint/metadata/log/manifest assets). Neither provides native PIV d32 or image row/split manifests. Authentication succeeded; this is missing provenance, not an authentication blocker. Synthetic replacement authorization does not extend to synthesizing unverified real-image or PIV identity.

The blocked configurations stay in the complete plan. They cannot enter a successful aggregate as absent or silently dropped rows. The parent execution plan must report these two conditions and their24 remaining A/B/C/t-Flow full runs until the missing evidence is supplied.

## Checks and remaining execution work

Twelve standard-library tests passed for scope completeness, budgets/seeds, deterministic generation-seed derivation, preservation of existing inputs/split definitions, no historical reuse, import safety, the login-node execution guard, and hard failures on changed/missing immutable assets. Planning compiled and emitted28 drafts. Actual tensor generation, finite checks, exact split hashes and complete prepared configs are performed only in the parent-managed Slurm materialization job. Model interface, conditioning leakage, target correctness, checkpoint, gradient, finite-sampling and network-call tests are separate execution gates before full model launches. No benchmark results are claimed by this preparation document.
