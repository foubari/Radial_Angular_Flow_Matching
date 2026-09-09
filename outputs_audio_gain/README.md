All three recovered fixed-spherical AudioMNIST EMA checkpoints were evaluated with independent empirical-gain rescaling. **All 6,000 digit predictions are unchanged.** All release assets, existing data-v1 inputs, checkpoint compatibility, and exact split checks passed. Historical accuracy reproduction remains discrepant and is explicitly recorded; the original paper results are preserved.

| Training seed | Original accuracy | Gain-rescaled accuracy | Original energy KS | Gain-rescaled energy KS | Changed predictions |
|---|---:|---:|---:|---:|---:|
| 8925 | 0.8025 | 0.8025 | 0.59233333 | 0.02183333 | 0 |
| 1234 | 0.8005 | 0.8005 | 0.59233333 | 0.02183333 | 0 |
| 7 | 0.8170 | 0.8170 | 0.59233333 | 0.02183333 | 0 |

New full-precision mean ± population SD: accuracy **0.806667 ± 0.007352**, energy KS **0.0218333 ± 0**, mass above external-test q95 **0.0500 ± 0**, and above q99 **0.0145 ± 0**. The paired original outputs have the same measured accuracy and energy KS **0.592333 ± 0**. Full radial W1, q90/q95/q99/q10 coverage, PIT, confidence, energy-bin accuracy, numerical tolerances and runtimes are retained below.

The new accuracy mean is higher than the published RAFM-Ang **0.764 ± 0.025** and RAFM-Vel **0.711 ± 0.014**; the new energy KS and tail coverage match those historical RAFM rows at their reported precision. This is a comparison to preserved historical measurements, not a claim of identical execution environments or statistical significance. It exploits AudioMNIST's explicit independence of gain, direction, and digit. The separate RAFM-Ang versus RAFM-Vel ablation remains a comparison of their training targets within RAFM. Because every model seed uses the same paired gain vector, zero radial-metric SD does not represent independent Monte Carlo gain replications.

The archived fixed-spherical accuracies are **0.805 / 0.798 / 0.828**, versus current **0.8025 / 0.8005 / 0.8170** in seed order 8925 / 1234 / 7. The published **0.810 ± 0.013** is unchanged. Source, RNG order, RK4 formulas, radius, conditioning and classifier checks found no concrete implementation mismatch. Original generated tensors, logits and exact historical execution records are unavailable, so the residual cause is unresolved. The different Torch/CUDA versus Torch/ROCm environments are possible context, not a demonstrated explanation. See the [reproduction audit](../docs/audio_gain_reproduction_audit.md).

The raw result remains `status="baseline_mismatch"`. Job 763355 therefore exits **1 after saving all three complete paired measurements**, as required by the historical-reproduction guard. Reporting uses the explicit audited-discrepancy option, and the new figure and captions disclose the issue. No measurement or reproduction flag was rewritten.

The protocol uses EMA step 24,000; UNet ch96/depth5 metadata (27,896,802 parameters; depth is unused by the original UNet factory); 2,000 generated clips per seed, 200 per digit; sample seed 0; original trained radius **2.2613651752471924**; 40 RK4 steps = **160 actual network evaluations per trajectory**; generation batch 128; full 2,000-sample classifier batch; original conditioning/CFG 1; Float32 evaluation; and original tangent projection without state renormalization. Gains are independent interpolated ECDF draws from the **10,200 generator-training samples only**, with isolated CPU gain seed 0. The exact internal split uses a local CPU `torch.Generator().manual_seed(0)` and `torch.randperm(12000)`; 1,800 internal validation examples remain separate from the external 3,000-example evaluation test file. The fixed-source initialization is unchanged, and gain rescaling is applied only to completed outputs. Pairing follows the original RAFM source draw order; the replay's normalization error is recorded separately.

The release checkout `d3006dc8ee61b2ab6e470115d3f21c778fefcf81` holds the recovered runs; **the training commit is unknown**, not verified by this checkout. Authenticated SHA-256 checks include EMA files, metadata, training logs, the manifest itself, and the reused data-v1 train/test/classifier files. All three EMA state dictionaries strictly loaded before generation. Missing legacy metadata fields use only the original evaluator's recorded fallback semantics.

Measured execution used one **AMD Instinct MI210 (64 GB class)** on `auh7-3b-gpu-019`, through Slurm, with Torch **2.7.1+rocm6.3**. Evaluation wall time was **2,683.44 s (44.72 min)**; allocated job elapsed time was **47 min 03 s**, including cluster setup. The preserved first attempt, job 763350, failed before generation because the MIOpen kernel cache was read-only; its allocation lasted **4 min 26 s**. A writable per-job cache fixed that execution issue without any model or sampling change. Both attempts together occupied one GPU for **51 min 29 s**. These are measured inference/evaluation times, not training times or matched-hardware speed comparisons.

| Seed | Generation seconds | Paired evaluation seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| 8925 | 916.961 | 6.913 | 3.768 | 5.301 |
| 1234 | 869.811 | 0.084 | 3.768 | 4.779 |
| 7 | 870.152 | 0.083 | 3.768 | 4.779 |

Deliverables:

- [Raw aggregate, including all per-seed results](fixed_spherical_empirical_gain_v1_attempt2/aggregate.json); separate seed JSONs: [8925](fixed_spherical_empirical_gain_v1_attempt2/seed_8925/eval.json), [1234](fixed_spherical_empirical_gain_v1_attempt2/seed_1234/eval.json), [7](fixed_spherical_empirical_gain_v1_attempt2/seed_7/eval.json).
- [Full original-versus-rescaled comparison](comparison_v1/comparison.md) and [machine-readable comparison](comparison_v1/comparison.json), with full-precision means and population SDs.
- [Updated Figure 2 PDF](manuscript_v1/figure2_audio_controls.pdf), [PNG](manuscript_v1/figure2_audio_controls.png), [Table 5 LaTeX](manuscript_v1/table5_audio_with_gain.tex), [complete-sample invariance table](manuscript_v1/table6_complete_gain_controls.tex), and [manuscript discussion](../docs/audio_gain_manuscript_addition.md). Historical table values are preserved; the main LaTeX source was not supplied, so these are standalone replacements.
- [Strict input verification](release_verification/strict_verification_compute.json), [source audit](release_verification/source_compatibility.json), [checkpoint/split validation](fixed_spherical_empirical_gain_v1_attempt2/compatibility.json), [test summary](release_verification/validation_summary.json), [package versions](release_verification/environment_packages.json), and [scheduler accounting](release_verification/slurm_accounting.txt).
- [Exact submitted Slurm script](release_verification/submitted_job_763355.sh), [actual invocation/status](release_verification/execution_job_763355.json), and [implementation and commands](../docs/audio_empirical_gain.md). Generated Y/X tensors, labels, gains, initial radii and before/after logits are saved under the `/mnt/vast01/users/fouad.oubari/data/tflow/audio_gain/` paths and SHA-256 values recorded in the raw JSONs.

Polling stopped after verified recovery. No retraining, t-Flow tuning, or additional benchmarks were launched. The missing original Student-t/anisotropy/toy caches remain a separate unresolved issue.
