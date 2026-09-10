# Measured study findings

Scope: **partial_suite**; 0/28 conditions have all A/B/C/t-Flow methods complete.

Final seed status counts: {"awaiting_sample_audit": 1, "blocked": 24, "complete": 16, "failed": 10, "in_progress_or_interrupted": 8, "missing": 277}.

New synthetic realizations are new shared comparisons; published historical results remain unchanged.

**1.** Normalized input plus radius conditioning (B) versus original RAFM-Ang (A): no complete compatible three-seed pair is available yet; no measured effect can be assigned.

**2.** Explicit radius conditioning (B) versus identical zero-conditioned modules (C): no complete compatible three-seed pair is available yet; no measured effect can be assigned.

**3.** B-A: 0 conditions improve in every seed, 0 worsen in every seed, 0 tie in every seed, and 0 have mixed/tied seed outcomes; B-C: 0 conditions improve in every seed, 0 worsen in every seed, 0 tie in every seed, and 0 have mixed/tied seed outcomes. Coverage and individual deltas, including negative results, are reported without a cross-dataset pooled score.

**4.** Recorded hardware/precision permit descriptive timing ratios for 0 B/A and 0 B/C condition pairs. The parameter table reports absolute counts and the recorded conditioning overhead; B/C receive no additional tuning. t-Flow validation-only training and sampling seconds are separately reconstructed only from complete, consistent trial records.

## Paired primary effects

| Condition | Contrast | Metric (better) | Left mean ± SD | Right mean ± SD | Mean delta | Three seed deltas | Mean outcome / consistency |
|---|---|---|---|---|---|---|---|

Deltas are left minus right. Positive favors the left method for digit accuracy; negative favors it for sliced W1/FID. Standard deviations are population SD. Seed order is recorded per condition in findings.json.

## Conditional angular effects

| Condition | Contrast | Angular metric | Mean delta | Three seed deltas | Mean outcome / consistency |
|---|---|---|---|---|---|

Angular sliced W1 is lower-is-better. Empty/nonfinite bins and incomplete seed groups do not acquire averages.

## Recorded cost

| Condition | Ratio | Training: mean seed ratio | Sampling: mean seed ratio |
|---|---|---|---|

These are raw ratios of measured times on matching recorded GPU/software and precision, not significance claims. Ratios of means and each seed ratio are also saved in findings.json.

| Condition | Method | Total parameters | Added conditioning parameters | Added fraction |
|---|---|---|---|---|
| aniso_k3 | tflow | 41504 | unavailable | unavailable |
| aniso_k300 | tflow | 41504 | unavailable | unavailable |
| finance_ff49 | tflow | 45873 | unavailable | unavailable |

## t-Flow

Against newly run A, t-Flow has 0 complete comparisons: {"improved": 0, "tied": 0, "worsened": 0} on each condition’s primary metric. Ranking is descriptive and limited to the complete methods listed below.

| Condition | Primary metric | Methods ordered by mean | Missing methods |
|---|---|---|---|

| Condition | Validation training-loop seconds | Validation sampling seconds | Training-update equivalents |
|---|---|---|---|
| aniso_k1 | 20.49362 | 4.163436 | 0.55 |
| aniso_k10 | 24.3679 | 4.199747 | 0.55 |
| aniso_k100 | 20.17464 | 4.204593 | 0.55 |
| aniso_k3 | 20.69776 | 4.205527 | 0.55 |
| aniso_k300 | 20.6037 | 4.200517 | 0.55 |
| finance_ff49 | 19.71517 | 4.354534 | 0.55 |
| gaussian_aniso_d16_cor | 22.53189 | 4.293257 | 0.55 |
| piv_d16 | 20.83483 | 4.162566 | 0.55 |
| piv_d256 | 22.0748 | 4.203993 | 0.55 |
| piv_d64 | 19.79262 | 4.305727 | 0.55 |
| student_t_d128_df3.0_cor | 19.07718 | 4.042663 | 0.55 |
| student_t_d16_df1.5_cor | 19.25178 | 4.148318 | 0.55 |
| student_t_d16_df2.0_cor | 21.7214 | 4.472294 | 0.55 |
| student_t_d16_df3.0_cor | 19.56787 | 4.104465 | 0.55 |
| student_t_d16_df5.0_cor | 18.70861 | 4.064509 | 0.55 |
| student_t_d16_df50.0_cor | 17.81318 | 4.143133 | 0.55 |
| student_t_d256_df3.0_cor | 20.46317 | 4.21281 | 0.55 |
| student_t_d32_df3.0_cor | 19.23536 | 4.167972 | 0.55 |
| student_t_d8_df3.0_cor | 23.21805 | 4.25049 | 0.55 |
| toy_radial_angular | 18.80803 | 3.870854 | 0.55 |
| weather_au_wind | 20.70263 | 4.269771 | 0.55 |

Tuning training time sums initial cumulative times plus each finalist’s cumulative time minus its own initial time. It does not double-count continuation checkpoints. Validation metrics, data loading and scheduling overhead are unmeasured here; incomplete/inconsistent trials are marked unavailable.

## Audio checkpoint reference

needs verified reference and completed compatible new audio methods.

These are newly measured metrics for the same saved Y/X under the current study backend. Archived metrics and raw audit status remain unchanged. Digit predictions, accuracy, energy KS and coverage rates are identical; PIT, radial W1, logits and near-fixed-radius energy-bin assignments can differ. Only these current-backend values enter new comparisons; no historical training-time comparison.

The raw backend audit remains compatibility_discrepancy; numerical differences are retained in findings.json and the report reference record.

Measured fixed-spherical accuracy is about 0.8067, not an exact reproduction of the reported 0.810 ± 0.013. The original baseline mismatch is preserved.

The checkpoint-based fixed-spherical+gain row is not new A and is not used for matched training-time claims. Constructed gains are independent of content; a B/C benefit is not presumed. The RAFM-Ang versus RAFM-Vel interpretation remains separate.

## Missing, failed and blocked work

- imagenette_dcae: image_split_conflict, historical_eval_provenance
- piv_d32: piv32_identity_missing
- Current final seed aniso_k10 / tflow / 77395: failed; []
- Current final seed piv_d256 / tflow / 8925: failed; []
- Current final seed piv_d256 / tflow / 77395: failed; []
- Current final seed piv_d256 / tflow / 65457: failed; []
- Current final seed piv_d64 / tflow / 8925: failed; []
- Current final seed piv_d64 / tflow / 77395: failed; []
- Current final seed piv_d64 / tflow / 65457: failed; []
- Current final seed student_t_d128_df3.0_cor / tflow / 8925: failed; []
- Current final seed student_t_d128_df3.0_cor / tflow / 77395: failed; []
- Current final seed student_t_d128_df3.0_cor / tflow / 65457: failed; []
- Unavailable B-A comparisons: aniso_k1, aniso_k10, aniso_k100, aniso_k3, aniso_k30, aniso_k300, audiomnist_stft, finance_ff49, gaussian_aniso_d16_cor, imagenette_dcae, piv_d16, piv_d256, piv_d32, piv_d64, student_t_d128_df3.0_cor, student_t_d16_df1.5_cor, student_t_d16_df10.0_cor, student_t_d16_df2.0_cor, student_t_d16_df3.0_cor, student_t_d16_df5.0_cor, student_t_d16_df50.0_cor, student_t_d256_df3.0_cor, student_t_d2_df3.0_cor, student_t_d32_df3.0_cor, student_t_d64_df3.0_cor, student_t_d8_df3.0_cor, toy_radial_angular, weather_au_wind
- Unavailable B-C comparisons: aniso_k1, aniso_k10, aniso_k100, aniso_k3, aniso_k30, aniso_k300, audiomnist_stft, finance_ff49, gaussian_aniso_d16_cor, imagenette_dcae, piv_d16, piv_d256, piv_d32, piv_d64, student_t_d128_df3.0_cor, student_t_d16_df1.5_cor, student_t_d16_df10.0_cor, student_t_d16_df2.0_cor, student_t_d16_df3.0_cor, student_t_d16_df5.0_cor, student_t_d16_df50.0_cor, student_t_d256_df3.0_cor, student_t_d2_df3.0_cor, student_t_d32_df3.0_cor, student_t_d64_df3.0_cor, student_t_d8_df3.0_cor, toy_radial_angular, weather_au_wind
- Unavailable tflow-A comparisons: aniso_k1, aniso_k10, aniso_k100, aniso_k3, aniso_k30, aniso_k300, audiomnist_stft, finance_ff49, gaussian_aniso_d16_cor, imagenette_dcae, piv_d16, piv_d256, piv_d32, piv_d64, student_t_d128_df3.0_cor, student_t_d16_df1.5_cor, student_t_d16_df10.0_cor, student_t_d16_df2.0_cor, student_t_d16_df3.0_cor, student_t_d16_df5.0_cor, student_t_d16_df50.0_cor, student_t_d256_df3.0_cor, student_t_d2_df3.0_cor, student_t_d32_df3.0_cor, student_t_d64_df3.0_cor, student_t_d8_df3.0_cor, toy_radial_angular, weather_au_wind

Separately preserved: 2 implementation-check failure archives and 1 earlier t-Flow startup-attempt archives. They are not current scientific seed failures.

## Interpretation limits

- t-Flow denotes an independent direct-noise reproduction with documented backbone and endpoint adaptations. For vector dimensions above 128, the width-128 affine output bottleneck leaves noise components amplified by 1/t_min=1000. These adaptation failures do not establish intrinsic inferiority of published t-Flow; see docs/tflow_matched_backbone_failure_analysis.md.
- New synthetic realizations are shared A/B/C/t-Flow comparisons, never historical tensors or replacement historical results.
- Three-seed mean differences and sign consistency are descriptive, not tests of statistical significance.
- No pooling of metric units across datasets, no averaging over missing/failed seeds, and no data-dependent tolerance for a tie.
- B versus A changes input parameterization and includes radius conditioning; B versus C isolates access to the scalar radius in identical modules.
- Condition sweeps share related generators and are not independent statistical replicates.
- This input reparameterization preserves the population angular transport objective and creates no new theoretical guarantee.
