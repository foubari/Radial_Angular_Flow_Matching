# PoC-B Stages 1–3 — is RAFM's content gap directional, and does scale-free angular RAFM close it?

Follow-up to PoC-B (UNet 27.9M, complex-STFT x=g·s so ‖x‖≡g, bimodal energy). Standard RAFM won energy
calibration (KS 0.022) but trailed matched-Euclidean on digit content (0.692 vs 0.736). Question: is the
gap in the learned **directions**, and does a **scale-free angular** regression target close it?

## Stage 1 — the content gap is directional (existing checkpoints, no retraining)

**1.1 Classifier gain-invariance** (`stage1_diagnostics.json`): real clips scaled to low/central/high
real-radius quantiles (0.79 / 1.28 / 4.73) → digit accuracy **0.939 at every level**. The classifier is
gain-invariant; energy does not create content-accuracy artifacts.

**1.2 Radius-controlled direction comparison** (n=1500, decode→re-STFT→classify; replace each generated
radius by the real median, or by shared empirical radii):

| method | acc natural | acc @common radius | acc @empirical radii | by orig energy lo/mid/hi |
|---|---|---|---|---|
| gaussian | 0.790 | 0.790 | 0.790 | .71/.83/.83 |
| matched | 0.772 | 0.772 | 0.772 | .72/.80/.80 |
| fixed_spherical | 0.865 | 0.865 | 0.865 | .83/.85/.92 |
| rafm (std) | 0.766 | 0.766 | 0.766 | .74/.78/.78 |

Radius replacement changes accuracy by **exactly 0.000** for every method → digit content depends only on
the **direction**. **Conclusion: the gap is directional**, not an amplitude/evaluation interaction. (After
the audio decode round-trip the std-rafm↔matched gap is small here, ~0.6 pt; on raw STFT it was ~4.4 pt.)

## Target-norm distributions — standard velocity vs angular (`target_norm_diag.json`)

Regression-target norm over the rafm coupling (empirical radial source + spherical path), n=8000:

| target | mean | std | CoV | corr(radius, target-norm) |
|---|---|---|---|---|
| standard velocity ‖u_t‖ | 3.57 | 2.64 | 0.739 | **+1.000** |
| angular ‖u_t‖/‖x_t‖ | 1.571 | 0.008 | 0.005 | **−0.012** |

The standard-velocity target norm **scales with the radius** (corr = 1.00) → training difficulty is
radius-coupled across the bimodal energy law. The angular target norm is **scale-free with respect to the
radius** (corr ≈ 0) **and bounded along the spherical path** (here it concentrates near ~π/2 because, in
this high-dimensional latent, a random source direction and the target direction are nearly orthogonal;
this is a property of the geometry, not a universal constant). This is the mechanism angular RAFM exploits.

## Stage 3 — scale-free angular RAFM (1 seed = 8925)

Network predicts A = v/‖x_t‖; velocity reconstructed as v = ‖x‖·A; same spherical path, empirical radial
source, UNet, initialization, optimizer, conditioning, budget (24k), sampling, tangent projection; **no
state renormalization**. Live target norm logged at 1.57±0.01, matching the diagnostic.

**Checkpoint trajectories (digit accuracy / energy KS):**

| step | matched | std rafm | **angular rafm** |
|---|---|---|---|
| 12k | 0.695 / 0.062 | 0.642 / 0.022 | **0.762 / 0.022** |
| 18k | 0.714 / 0.099 | 0.658 / 0.022 | **0.785 / 0.022** |
| 24k | 0.736 / 0.110 | 0.692 / 0.022 | **0.798 / 0.022** |

**@24k full comparison:**

| method | digit acc | energy KS | cov>q95 (.05) | cov<q10 (.10) | PIT (.5) |
|---|---|---|---|---|---|
| gaussian | 0.739 | 0.122 | 0.059 | 0.174 | 0.482 |
| matched | 0.736 | 0.110 | 0.039 | 0.200 | 0.443 |
| fixed_spherical | 0.805 | 0.592 | 0.000 | 0.000 | 0.592 |
| rafm (std) | 0.692 | 0.022 | 0.050 | 0.098 | 0.493 |
| **rafm (angular)** | **0.798** | **0.022** | **0.050** | **0.098** | **0.493** |

**Angular RAFM: content 0.798 (beats matched 0.736 and std-rafm 0.692 by +10.6 pts) while keeping RAFM's
exact energy calibration (KS 0.022, cov/PIT identical).** Per-energy accuracy is lifted and flattened
(.75/.77/.77 vs std-rafm .64/.64/.65), i.e. the low-energy content — where std-rafm was worst — is fixed.

**Radial drift (no state renorm, `drift_check.json`):** source radius vs generated radius, per-sample
relative drift — std rafm: mean 0.0016 %, max 0.042 %; **angular rafm: mean 0.0042 %, max 0.055 %**. Both
negligible → empirical-radius preservation holds for the angular variant without any state renormalization.

## Stage 2 — 3-seed uncertainty (seeds 8925, 1234, 7; `stage2_3seed.json`)

Mean ± std over 3 seeds, same UNet/24k/batch32/eval:

**@ 24k — all 5 methods, 3 seeds:**

| method | digit acc | energy KS | cov>q95 (.05) | cov>q99 (.01) | cov<q10 (.10) | PIT (.5) |
|---|---|---|---|---|---|---|
| gaussian | 0.734 ± 0.004 | 0.117 ± 0.008 | 0.059 | 0.019 | 0.171 | 0.487 |
| matched | 0.750 ± 0.018 | 0.095 ± 0.010 | 0.040 | 0.012 | 0.179 | 0.450 |
| fixed_spherical | **0.810 ± 0.013** | 0.592 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.592 |
| std rafm | 0.711 ± 0.014 | **0.022 ± 0.000** | 0.050 | 0.015 | 0.097 | 0.492 |
| **angular rafm** | 0.764 ± 0.025 | **0.022 ± 0.000** | 0.050 | 0.015 | 0.097 | 0.492 |

**Key cross-method reading:** content ranks fixed_spherical 0.810 > **angular 0.764** > matched 0.750 >
gaussian 0.734 > std-rafm 0.711; energy-KS ranks angular = std-rafm 0.022 ≪ matched 0.095 < gaussian
0.117 ≪ fixed_spherical 0.592. **fixed_spherical (constant radius) has the best content but destroys energy
(point mass, zero tail); angular RAFM is the only method that is top-tier on *both* axes.** This confirms
the mechanism: constant-radius (fixed_spherical) and scale-free-target (angular) both make directional
learning easy → high content; std-rafm's radius-coupled velocity target is hardest → lowest content.
Angular recovers most of fixed_spherical's directional advantage **while keeping** the energy calibration
fixed_spherical throws away.

Trajectory (digit acc, 3-seed mean): 12k 0.693/0.657/0.721 · 18k 0.722/0.681/0.746 · 24k
0.750/0.711/**0.764** (matched / std-rafm / angular). Angular leads std-rafm at every checkpoint.

- **Energy:** angular = std-rafm, **KS 0.022 ± 0.000** (seed-invariant, source-determined) — 4.3× better
  than matched (0.095), with cov<q10 0.097 ≈ target vs matched's 0.179 over-coverage, PIT 0.492 vs 0.450.
- **Content:** angular **0.764 ± 0.025** > std-rafm 0.711 ± 0.014 (gap +0.053, CIs separated) and
  **matches-to-slightly-exceeds** matched 0.750 ± 0.018 (overlapping — statistical parity with a small
  positive mean edge).

## Final verdict — scale-free angular regression closes the trade-off

Across 3 seeds, angular RAFM **retains RAFM's exact energy-tail calibration** (KS 0.022, unchanged from
std-rafm; 4.3× better than matched, better low-tail coverage and PIT) **while lifting digit content by
+0.05 over std-rafm to parity-or-better with matched-Euclidean.** The mechanism is confirmed: the angular
target is scale-free w.r.t. the radius (corr ≈ 0, vs std-velocity corr = 1.00) and bounded along the
spherical path, so directional learning is no longer harder at extreme radii — with negligible radial
drift and no state renormalization. **The content–energy trade-off of standard RAFM is resolved:** you
keep the calibrated energy and recover the directional/content quality.

Caveats: single dataset (AudioMNIST digits), 3 seeds, 24k steps (content still rising — not saturated);
angular-vs-matched content is a statistical tie with a small positive edge, not a decisive win.

**Language note:** gaussian is not "structurally unable" to model the bimodal energy — it fails **under
this architecture/budget** (KS 0.12). The angular target is **scale-free w.r.t. the radius and bounded
along the spherical path**, not universally constant.
