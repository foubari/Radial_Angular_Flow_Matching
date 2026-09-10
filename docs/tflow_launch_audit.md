# t-Flow launch audit, 2026-09-10

The user has authorized the full t-Flow study and a separate RAFM-Ang A/B/C study.
This audit covers t-Flow implementation and launch preparation. It launches no
training, tuning, or benchmark jobs itself. The parent scheduler coordinates the
two studies and records their actual job IDs and outcomes separately.

## Method and source verification

The core implements direct multivariate Student-t noise prediction, not ordinary
velocity regression with a changed prior. It draws one chi-square scale for each
example, shared by every coordinate, draws uniform continuous training times,
forms the straight interpolant, and uses the corresponding noise-to-velocity
field. Tests separately check the shared scale, the interpolation, the noise
loss gradient and its distinction from uniformly weighted velocity MSE.

No official authors' t-Flow implementation has been verified. This remains an
independent reproduction. The author's page and NVIDIA publication page were
rechecked; neither provides a t-Flow repository. The later PhysicsNeMo t-EDM
implementation is not treated as t-Flow provenance.

Targeted indexed text from the final ICLR proceedings verifies the formulation
at Appendix B Eqs. 167–169, corresponding to arXiv-v2 Eqs. 164–166. The indexed
final Algorithms 3/4 retain ambiguous training-time and Heun notation. Full PDF
download still fails with proxy CONNECT 403; the web reader rejects its 33.6 MB
size. Final-page images have not been inspected. These distinctions, links and
the exact numerical adaptations are in [tflow_method.md](tflow_method.md).

The configured sampler starts at `t_min=0.001`, interprets a fresh prior draw as
that starting state, uses the documented rho-7 noise grid ending at sigma 0.01,
and appends the data endpoint. The omitted initial interval introduces an
approximation; the noise-prediction error is amplified by `1/t`. There is no
claimed exact reproduction of an unpublished endpoint implementation. An
analytic conditional-noise oracle test checks truncation and integration
behavior, separately from any learned-model quality claim. All model times are
positive, ordered and representable, with explicit failure rather than clipping
or repairing nonfinite values.

## Architectures, budgets and currently resolved configurations

| Configuration | Backbone | Batch | Updates | Actual sampling model calls/trajectory |
|---|---|---:|---:|---:|
| Finance | original three-hidden-layer width-128 MLP | 4096 | 10,000 | 512 |
| Weather | same MLP | 2048 | 10,000 | 512 |
| PIV d64 | same MLP | 256 | 10,000 | 512 |
| PIV d256 | same MLP | 256 | 10,000 | 512 |
| AudioMNIST | original ch96 UNetVel, 27,896,802 parameters | 32 | 24,000 | 160 |

These are resolved entries in the original prepared inventory. Other conditions
must use their individually resolved configurations or explicitly labelled new
shared-cache comparisons. This audit does not resolve historical synthetic
realization gaps or PIV protocol disagreements by adopting silent defaults.
Image SiT architecture support exists; its reference/model/data prerequisites
must pass separately before its launch.

The runner retains optimizer, learning rate, EMA, class dropout, precision and
training seeds in each reviewed configuration. It uses the original backbone
classes, with no capacity increase. Direct Euclidean noise MSE uses the
documented sum-over-coordinates, mean-over-examples reduction. Per-method final
training budgets remain unchanged. Heun uses two actual network calls per
interval; vector, audio and image call budgets are 512, 160 and 100 respectively.
Reports include calls across sample minibatches as well as per trajectory.

## Validation-only source selection

There are nine candidates: prior degrees of freedom 3, 5 and 7 crossed with
source-scale multipliers 0.5, 1 and 2. Each degree of freedom uses a scale
calibrated to the generator-training median radius via the multivariate-t radial
F distribution. No target degrees of freedom, target variance, validation radius
fit, test gain or test metric sets the source scale.

All nine train for 5% of the matched updates using tuning seed 46021. The best
two continue to 10% total. The full budget is therefore **0.55 final-training
equivalents per condition**, plus 11 validation generations of 1,000 examples.
Selection uses an equally weighted radial KS and projected KS statistic on the
internal validation split; the external audio test file is not read. The winner
is frozen once. All three final model seeds start fresh at step zero.

There is no automatic replacement of a failed candidate, scale, source, sampler,
or missing score. A failed condition is retained for review, and unrelated
conditions can proceed under the parent's orchestration.

## Runtime fixes made before launch

These changes do not alter the loss, solver, backbone, optimizer or sample budget:

- Gradient finiteness now reduces all per-parameter checks on the accelerator
  before one host synchronization, replacing a synchronization for each
  parameter. The boolean decision and missing-gradient behavior are unchanged.
- Training and sampling record measured allocated/reserved memory. Checkpoints
  retain training peaks across resumptions. Final results include parameter
  counts and exact realized tensor/split hashes, dtype, shape and preprocessing
  mean. An empty internal test partition remains distinct from an external test.
- Audio classification now uses the original single full 2,000-example FP32
  classifier forward, matching `audio_eval.py` and the completed fixed-spherical
  gain evaluation. Full audio summaries, logits, predictions and reference
  identities are saved alongside flattened metrics.
- The original job template sets separate per-job writable MIOpen kernel-cache
  and performance-database directories, preventing the previously observed
  SQLite cache failure and avoiding cache collisions between independent jobs.

Checkpoint identity binds configuration, source choice, seed, stage and relevant
source/environment hashes. Model, EMA, optimizer and all RNG states are restored.
A completed checkpoint performs no extra training steps. Final test evaluation
requires the complete matched training budget. Exceptions and undefined metrics
are saved as failures; partial seeds never become an all-seed aggregate.

## Exact validation and launch commands

Run numerical checks inside a cluster allocation. The following is a command
specification; it is not evidence that these commands ran during this audit.

```bash
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
export PYTHONPATH="$PWD/tools/validation_deps:$PWD"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
"$PY" -m pytest -q tests/test_tflow_core.py tests/test_tflow_runtime.py \
  tests/test_tflow_tuning.py tests/test_tflow_validation_policy.py \
  tests/test_tflow_reporting.py
```

Use an unused output root for the newly approved study. All three entry points
already accept custom configuration and output paths; no CLI patch is required.

```bash
CFG=configs/tflow/prepared/finance_ff49.json
OUT=outputs_tflow/approved_study_20260910
"$PY" -m experiments.tflow.sanity --config "$CFG" --output "$OUT/sanity"
"$PY" -m experiments.tflow.tune --config "$CFG" --output "$OUT/tuning"
for SEED in 8925 77395 65457; do
  "$PY" -m experiments.tflow.run train-evaluate --config "$CFG" \
    --selection "$OUT/tuning/finance_ff49/selection.json" \
    --seed "$SEED" --output "$OUT/final"
done
```

One visible accelerator is enforced per process, and experiment commands reject
login-node execution. The scheduler may run distinct conditions/seeds in
parallel. Update source files only before freezing source selections; a later
source change correctly invalidates old selections and resumptions.

Static parsing/compilation of the changed runtime and tests, plus shell syntax
checking of the template, passed. Numerical validation of the new edits is a
parent-scheduled compute-node gate. Prior unit-check artifacts establish the
earlier implementation only; they are not represented as tests of these edits.
