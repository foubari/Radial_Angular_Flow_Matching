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
core fields are errors. The recovered legacy metadata omits `ncls` for all
three seeds and `angular` for seed 8925. The inspected original evaluator
explicitly defaults these to 10 and false; those resolutions are recorded
separately while preserving the released files. They establish compatibility,
not a verified training commit. See [the release audit](audio_release_compatibility.md).

Required explicit paths are the original 12,000-example nominal training file,
3,000-example external test file, digit classifier, each EMA checkpoint and its
`meta.json`, and the archived `stage2_3seed.json` aggregate. Place dataset copies
and generated sample tensors under `/mnt/vast01/users/fouad.oubari/data/`.
Original metadata/checkpoints were recovered from `fixed-spherical-audiomnist-v1`.
The release verifier checks checkpoints, metadata, logs, and the manifest against
authenticated GitHub asset digests, plus the embedded manifest checksums; any
mismatch stops execution. Existing data-v1 tensors are checked in place. No
assets are downloaded or reconstructed by this evaluator itself.

The evaluator recreates `audio_flow.build`'s 85% training split (10,200 examples)
and verifies its mean radius against each checkpoint's `R0`. It uses the
original fixed-source initialization, which samples at the trained radius up
to floating-point normalization error. There is no gain-dependent generation.
Input SHA-256 hashes, original metadata, recomputed radius, and solver drift are
recorded. Radius agreement alone is not a substitute for original artifact
provenance; the reproduced baseline is also compared to its archived metrics.
The exact split uses a separate CPU `torch.Generator().manual_seed(0)` and
`torch.randperm(12000)`, then selects the first 10,200 permutation entries.
The remaining 1,800 rows are internal validation, distinct from the external
3,000-row test file. Both index vectors are hashed in the evaluation report.
All three EMA state dictionaries are checked for strict model compatibility
and finite weights before any original-output generation starts.

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

The user authorized the recovered-checkpoint fixed-spherical + empirical-gain
evaluation. Job **763355** completed all three original-seed measurements; this approval
does not include t-Flow training, tuning, or evaluation. The original checkpoints
are available and verified, so missing fixed-spherical checkpoints are no longer
a blocker. All 6,000 paired predictions are unchanged; the historical accuracy
reproduction guard remains flagged. See [the completed report](../outputs_audio_gain/README.md).

The checkout is `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions`.
Recovered artifacts are in `inputs/recovered_baselines/`, with
`ema_24000_seed{8925,1234,7}.pt`, corresponding `meta_seed*.json`, logs, and
`manifest.json`. The launcher
[`tools/audio_gain_release_job.sh`](../tools/audio_gain_release_job.sh) uses
[`tools/evaluate_recovered_audio.py`](../tools/evaluate_recovered_audio.py),
which verifies release checksums and the source-compatibility audit before
calling the evaluator. Its interpreter is
`../msgm-sparse-control/.venv/bin/python`; additional validation dependencies
are in `tools/validation_deps`.

The recorded submission command was `sbatch --parsable tools/audio_gain_release_job.sh`.
This evaluation is finished; the command below is a reproducibility record. The exact submitted
script and job settings are preserved in
[`submitted_job_763355.sh`](../outputs_audio_gain/release_verification/submitted_job_763355.sh)
and [`submission_job_763355.json`](../outputs_audio_gain/release_verification/submission_job_763355.json).
Current execution status is recorded in
[`execution_status.json`](../outputs_audio_gain/release_verification/execution_status.json).

The completed report directory is
`outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2`.
The earlier `fixed_spherical_empirical_gain_v1` directory is preserved from
job 763350, which failed at the first convolution because the MIOpen kernel
cache was read-only, before producing samples. The retry uses writable per-job
caches without changing the algorithm, batch, precision, checkpoints, or sampling;
see [`cache_failure_and_retry.json`](../outputs_audio_gain/release_verification/cache_failure_and_retry.json).

The following **preflight only** command uses the recovered release paths and a
new prospective report directory. It checks file presence and JSON metadata
without importing torch, deserializing tensors, creating output directories,
or drawing samples. Run from the checkout root:

```bash
python3 experiments/poc_audio/audio_empirical_gain.py preflight \
  --train-file /mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/poc_audio/data/audiomnist_stft_train.pt \
  --test-file /mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/poc_audio/data/audiomnist_stft_test.pt \
  --classifier /mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/poc_audio/digit_classifier.pt \
  --reference-aggregate experiments/poc_audio/stage2_3seed.json \
  --run 8925 inputs/recovered_baselines/ema_24000_seed8925.pt inputs/recovered_baselines/meta_seed8925.json \
  --run 1234 inputs/recovered_baselines/ema_24000_seed1234.pt inputs/recovered_baselines/meta_seed1234.json \
  --run 7 inputs/recovered_baselines/ema_24000_seed7.pt inputs/recovered_baselines/meta_seed7.json \
  --output-dir outputs_audio_gain/preflight_check_only \
  --samples-root /mnt/vast01/users/fouad.oubari/data/tflow/audio_gain
```

The launcher invoked `evaluate` in its allocated GPU process.
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
are not silently replaced with archived numbers. The renderer requires
`status="complete"` by default. After a documented audit, its explicit
`--baseline-discrepancy-note PATH` option can also report a complete three-seed
`baseline_mismatch` result with a visible discrepancy label. This does not
convert the evaluator status to success or claim historical reproduction.
All protocol, finite-metric, checkpoint, and invariance checks still apply.
Existing metrics/assets remain intact.

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

For measured additions, use
`--posthoc-result outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json`.
The renderer requires all
three original seeds, 2,000 samples, the matched protocol, finite measurements,
checkpoint hashes, passing per-sample/logit/direction/radius checks, exact
before/after counts, and, by default, successful original-baseline reproduction. It derives
the plotted means from full-precision per-seed measurements rather than trusting
supplied aggregate summaries. The scatter uses hollow individual-run markers,
filled means with population SD error bars, log KS, and linear accuracy. Its
original RAFM-Ang versus RAFM-Vel comparison remains present.

If all three seeds pass these measurement checks but the evaluator records
`baseline_mismatch` and `archived_baselines_reproduced=false`, a completed audit
can be supplied with `--baseline-discrepancy-note`. The note must be a nonempty
UTF-8 file. Its absolute path, exact content and SHA-256 are saved in
`reporting_manifest.json`. The gain row and figure legend are marked, the figure
has a visible disclosure, and both the table caption and `figure2_caption.txt`
state that original-checkpoint reevaluation differs from the archive. New points
are actual measurements; the original fixed-spherical, RAFM-Vel and RAFM-Ang
points and all published table numbers stay unchanged. Missing seeds, changed
predictions, failed invariance, invalid metrics or mismatched protocols remain
errors even with the note. The option does not relax t-Flow validation.

After all three seeds finish and the audit note exists, set
`AUDIO_GAIN_AUDIT_NOTE` to that note's path and render into a new directory:

```bash
PYTHONPATH="$PWD/tools/validation_deps:$PWD" \
  ../msgm-sparse-control/.venv/bin/python experiments/poc_audio/render_gain_results.py \
  --reference-aggregate experiments/poc_audio/stage2_3seed.json \
  --posthoc-result outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json \
  --baseline-discrepancy-note "${AUDIO_GAIN_AUDIT_NOTE:?Set this to the completed audit note}" \
  --output-dir outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/figures_with_discrepancy
```

Omit the discrepancy-note flag for a result that reproduces the baseline.
The JSON-only summary command below reports per-seed original/posthoc metrics,
full-precision means and population SDs, changed-prediction counts, hardware,
runtime and provenance. It preserves mismatch and failure labels and withholds
means when a seed is missing or failed:

```bash
python3 tools/summarize_audio_gain.py \
  --input outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json \
  --output-dir outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/comparison
```

The explicit `--input` selects the active retry; the summary script's default
still names the preserved first-attempt directory. Neither reporting command
trains a model or reruns sampling. Their output directories must be new.

The optional `--tflow-result` accepts the actual complete suite condition report:
`<suite_report>/conditions/audiomnist_stft.json`. It re-inspects all three original
`result.json` and `implementation.json` records, their canonical configuration
and implementation hashes, checkpoint and generated-sample hashes, the frozen
source parameters, all applicable metrics, and the exact audio protocol. Original
per-seed artifacts must remain accessible. When combined with the empirical-gain
result, training, test and classifier hashes must also agree. Two valid seeds,
missing metrics, a changed artifact or an unsupported custom score envelope
cannot produce a new row. t-Flow training, tuning and evaluation remain
unapproved; the fixed-spherical checkpoint-evaluation authorization does not
extend to them.

Outputs include the standalone replacement Table 5, Figure 2 PDF/PNG, an
unchanged historical Table 6 (1,200 real-classifier validation examples;
1,500 radius-replacement examples), and a separate table of the new complete
2,000-sample gain checks. `reporting_manifest.json` records source hashes and
whether assets are a pending preview, measured additions, or measured additions
with an audited baseline discrepancy. Complete-sample
Gaussian, matched-source, RAFM-Vel, and RAFM-Ang controls remain missing until
their original generated outputs are supplied to the generic invariance helper
or their original checkpoints are explicitly supplied through `--reference-run`.
Validated optional reference controls are included in the new complete-sample
table; the manifest names any methods whose complete controls remain missing.
Neither their results nor the historical 1,500-sample result are silently
substituted for the complete new checks.
