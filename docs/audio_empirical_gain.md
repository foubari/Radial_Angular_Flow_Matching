# AudioMNIST fixed-spherical + empirical gain

This addition reuses the original fixed-spherical EMA checkpoints and evaluates
`X = G * Y / ||Y||` **after** generating `Y`. It neither retrains the generator
nor changes its initial radius, vector field, integrator, or trained targets.
`G` is drawn independently of content and digit identity from the same
training-derived quantile-interpolation law used by RAFM.

The paper's Section 4.2 / Figure 2 (page 8), Table 5 / Table 6 (page 26), and
Appendix C.10 specify the experiment. Figure 2 is the accuracy-versus-energy-KS
scatter, with a log KS axis and individual runs plus mean/SD. The older local
`make_audio_fig.py` mechanism plot is a different asset. Retain RAFM-Vel versus
RAFM-Ang in the table and scatter; the added control tests the effect of an
independent post-generation gain. The supplied PDF tables and Figure 2 were
visually checked against the extracted text during this implementation.

## Fixed protocol and required artifacts

The evaluator requires all three original seeds, in order **8925, 1234, 7**,
using final **EMA step 24000**, UNet base width 96. It checks each run's metadata,
including fixed-spherical method, non-angular target, split seed 0, 24,000
training steps, batch 32, AdamW learning rate 0.0002, and EMA 0.999. Missing
fields are errors; defaults do not establish checkpoint provenance.

Required explicit paths are the original 12,000-example nominal training file,
3,000-example external test file, digit classifier, each EMA checkpoint and its
`meta.json`, and the archived `stage2_3seed.json` aggregate. Place dataset copies
and generated sample tensors under `/mnt/vast01/users/fouad.oubari/data/`.
Original metadata/checkpoints must be recovered; the MSGM checkpoints are not
substitutes. No assets are downloaded or reconstructed by this script.

The evaluator recreates `audio_flow.build`'s 85% training split (10,200 examples)
and verifies its mean radius against each checkpoint's `R0`. It uses the
original fixed-source initialization, which samples at the trained radius up
to floating-point normalization error. There is no gain-dependent generation.
Input SHA-256 hashes, original metadata, recomputed radius, and solver drift are
recorded. Radius agreement alone is not a substitute for original artifact
provenance; the reproduced baseline is also compared to its archived metrics.

Generation uses sample seed **0**, 2,000 samples with 200 per digit in the
original order, CFG 1, **40 RK4 steps / 160 model evaluations**, tangent velocity
projection, and no state renormalization. The original argument named `nfe`
counts RK4 steps. Default generation batch size is 128; classifier evaluation
uses the original complete 2,000-example batch. There is no CPU fallback.

## Exact gain law and pairing

`RadialEmpiricalSource(mode="ecdf").fit(x[training_indices])` uses STFT norms of
the actual generator training split. The draw is
`torch.quantile(training_norms, torch.rand(n))`, with linear interpolation.
Neither the theoretical mixture, empirical-index resampling, all 12,000
nominal training examples, nor external test gains define the fitted law.

The default gain seed is 0 in an isolated CPU RNG scope. The implementation
seeds the **CPU default generator only**, restores its state, and does not seed
CUDA while drawing gains. Fixed-spherical generation uses its own saved/restored
CPU and CUDA scope at sample seed 0. Gain sampling does not read generated
directions, labels, or classifier outputs. All model seeds share the same gain
vector, so their radial metrics have shared Monte Carlo variation.

The repository's `source.sample` draws radii **before** directions. Therefore
the isolated `sample_radii` call reproduces the original RAFM quantiles. A
separate replay also stores `source.sample(n,d).norm(dim=1)`, exactly as the
original evaluator initialized RAFM; its difference from the raw gains records
floating-point normalization error. Post-processing uses the raw ECDF draws.
The optional `--gain-seed` changes this draw only and is always reported; seed
0 is the paper-paired default. No empirical gain vector is supplied from test
data.

## Commands and execution status

Implementation and static review do not authorize an experiment. Run numerical
evaluation only after the user's approval and on an allocated compute node.
The `preflight` command checks file presence and JSON metadata without importing
torch, deserializing tensors, creating output directories, or drawing samples.
The placeholders below must be replaced with actual original artifact paths.

```bash
python experiments/poc_audio/audio_empirical_gain.py preflight \
  --train-file /mnt/vast01/users/fouad.oubari/data/rafm/audio/audiomnist_stft_train.pt \
  --test-file /mnt/vast01/users/fouad.oubari/data/rafm/audio/audiomnist_stft_test.pt \
  --classifier /path/to/original/digit_classifier.pt \
  --reference-aggregate /path/to/original/stage2_3seed.json \
  --run 8925 /path/to/runs_unet/fixed_spherical/ema_24000.pt /path/to/runs_unet/fixed_spherical/meta.json \
  --run 1234 /path/to/runs_s1234/fixed_spherical/ema_24000.pt /path/to/runs_s1234/fixed_spherical/meta.json \
  --run 7 /path/to/runs_s7/fixed_spherical/ema_24000.pt /path/to/runs_s7/fixed_spherical/meta.json \
  --output-dir /path/to/new/audio_reports/run_001 \
  --samples-root /mnt/vast01/users/fouad.oubari/data/tflow/audio_gain
```

After approval, change `preflight` to `evaluate` in the allocated GPU process.
`evaluate` requires `SLURM_JOB_ID` before importing ML libraries or loading any
tensor; this guard also covers direct calls to the evaluator. Preflight and
small reusable correctness helpers do not require a Slurm environment.
Use a new output directory; existing results are never overwritten. Numerical
unit checks are in `tests/test_audio_empirical_gain.py`; they require no model
checkpoints, training, or benchmark data. Their execution should occur on the
approved CPU compute allocation.

## Measurements and fail-closed reporting

The primary classifier receives `Y/||Y||` and `X/||X||` directly in STFT space,
as in `audio_eval.py`. It does not use waveform decoding/re-STFT as the primary
score. Each seed records direction error, `||X||-G` absolute/relative error,
logit differences, every changed prediction, exact correct counts, and accuracy
change. Positive finite directions/gains are required. Any prediction change,
accuracy change, or failed numerical tolerance prevents a complete aggregate.
The original solver's radial drift is measured separately and is not corrected.

Metrics reproduce the original external-test conventions: radial Wasserstein-1,
KS, strict mass above test q90/q95/q99, below q10, and left-search PIT. Confidence
and accuracy by generated-energy terciles are included. Tercile accuracy can
change because independently assigning energy changes bin membership; global
content accuracy and individual predictions must remain unchanged.

Full-precision metrics and exact correct counts are retained in each result's
`unrounded` object. Original per-seed rounding is also retained for comparison
and for the legacy-style aggregate (population SD). A mismatch with the
archived fixed-spherical baseline yields `status="baseline_mismatch"` and a
nonzero exit status; measured values remain available for investigation and
are not silently replaced with archived numbers. Only `status="complete"`
is eligible for adding the new paper row. Existing metrics/assets remain intact.

The reusable `evaluate_gain_invariance(Y, gains, labels, classifier, test_gains)`
helper applies the complete control to already-generated outputs from any of
the four Table 6 methods. It supplies paired before/after metrics and checks,
without changing that method's generation. Additional reference checks require
the respective original checkpoint outputs; the old Table 6 diagnostic used
1,500 examples, decode/re-STFT, and test-sampled gains, and is not reported here
as a completed version of this new control.

The CLI also accepts repeated
`--reference-run METHOD SEED CHECKPOINT METADATA` entries for
`gaussian_euclidean`, `matched_euclidean`, `rafm` (RAFM-Vel), and `angular_rafm`.
Each supplied method must have all three original seeds; metadata is checked
against that method and angular-target flag. These optional runs retain their
own original source and sampler, generate all 2,000 clips, and apply the same
independent gain vector afterward. They are additional inference/evaluation
work and need experiment approval together with the main control. Their
per-method samples and checks are stored under `reference_controls/`; failed
invariance in any requested control prevents a complete aggregate. No reference
checkpoint is inferred from a fixed-spherical checkpoint.

All generated tensor datasets (`paired_gains.pt`, per-seed `samples.pt` with Y,
X, labels, gains, initial radii, and before/after logits, including optional
reference controls) are written beneath `--samples-root`, defaulting to
`/mnt/vast01/users/fouad.oubari/data/tflow/audio_gain`. The run directory is a
24-character SHA-256 prefix of the absolute report output directory; an existing
sample run directory is rejected even if its old report directory was removed.
An explicit `--samples-root` overrides the default. Preflight prints the resolved
sample directory without creating it. Tensor paths, byte counts, and SHA-256
checksums are recorded in JSON alongside input/source hashes; the Slurm job ID
is recorded with the execution protocol.

Per-seed `eval.json`, `tensor_artifacts.json`, and `aggregate.json` stay in the
chosen report directory, which may be a repository output location. Figures
produced by the separate renderer likewise remain in its output directory.
The aggregate contains `method="fixed_spher_empirical_gain"`,
`status`, `protocol`, full per-seed `runs`, `aggregate` metric objects with
`mean`, `std`, `vals`, and `n`, and parallel `aggregate_full_precision` objects.
A failed run does not produce a complete aggregate.

## Standalone Table 5 / Figure 2 updates

`render_gain_results.py` reads JSON measurements only. Its original six Table 5
rows are transcribed exactly from the supplied PDF (SHA-256
`97f709df2a45e4acf4ba186379c60e9c75120d95e09af782d7a18c8b56d00367`).
The Figure 2 reference points come from `stage2_3seed.json`; their recomputed
means and population SDs must reproduce the published values. Sparse MSGM stays
in Table 5; the original Figure 2 contains the five FM controls.

An explicit preview command produces only existing measurements and labels both
the figure and table as pending. It does not invent a post-processing point:

```bash
python experiments/poc_audio/render_gain_results.py \
  --reference-aggregate experiments/poc_audio/stage2_3seed.json \
  --pending --output-dir /path/to/new/pending_audio_assets
```

For measured additions, replace `--pending` with
`--posthoc-result /path/to/evaluation/aggregate.json`. The renderer requires all
three original seeds, 2,000 samples, the matched protocol, finite measurements,
checkpoint hashes, passing per-sample/logit/direction/radius checks, exact
before/after counts, and successful original-baseline reproduction. It derives
the plotted means from full-precision per-seed measurements rather than trusting
supplied aggregate summaries. The scatter uses hollow individual-run markers,
filled means with population SD error bars, log KS, and linear accuracy. Its
original RAFM-Ang versus RAFM-Vel comparison remains present.

The optional `--tflow-result` accepts the actual complete suite condition report:
`<suite_report>/conditions/audiomnist_stft.json`. It re-inspects all three original
`result.json` and `implementation.json` records, their canonical configuration
and implementation hashes, checkpoint and generated-sample hashes, the frozen
source parameters, all applicable metrics, and the exact audio protocol. Original
per-seed artifacts must remain accessible. When combined with the empirical-gain
result, training, test and classifier hashes must also agree. Two valid seeds,
missing metrics, a changed artifact or an unsupported custom score envelope
cannot produce a new row. No t-Flow experiment has been completed yet.

Outputs include the standalone replacement Table 5, Figure 2 PDF/PNG, an
unchanged historical Table 6 (1,200 real-classifier validation examples;
1,500 radius-replacement examples), and a separate table of the new complete
2,000-sample gain checks. `reporting_manifest.json` records source hashes and
whether assets are a pending preview or measured additions. Complete-sample
Gaussian, matched-source, RAFM-Vel, and RAFM-Ang controls remain missing until
their original generated outputs are supplied to the generic invariance helper
or their original checkpoints are explicitly supplied through `--reference-run`.
Validated optional reference controls are included in the new complete-sample
table; the manifest names any methods whose complete controls remain missing.
Neither their results nor the historical 1,500-sample result are silently
substituted for the complete new checks.
