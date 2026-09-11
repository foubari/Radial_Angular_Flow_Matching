# RAFM A/B/C completion report

Snapshot generated **2026-09-10T18:59:58.769297+02:00 (Paris)**; results collected at 2026-09-10T16:59:44Z.

**finished_with_failures**: 245/252 verified complete, 7 failed, 0 missing, 0 in progress or interrupted, 0 awaiting sample audit, 0 otherwise unresolved.

26/28 conditions have all nine A/B/C seeds verified. Scientific failures are recorded outcomes, not missing jobs or zero measurements. Finished-with-failures means every prescribed outcome was recorded; it does not mean all experiments passed.

No live scheduler query. A training directory without final result is labelled in_progress_or_interrupted; it does not establish that a job is running.

A = original RAFM-Ang; B = unit direction plus radius; C = unit direction with constant-zero radius input. Every displayed aggregate requires three compatible audited seeds and uses population SD.

| Condition | Primary metric | A | B | C | Existing t-Flow reference |
|---|---|---|---|---|---|
| aniso_k1 | sliced_w1 | 0.0211472 ± 0.000229 | 0.0211655 ± 0.000372 | 0.0211521 ± 0.000376 | 0.504725 ± 0.047 |
| aniso_k10 | sliced_w1 | 0.134339 ± 0.0134 | 0.154051 ± 0.0335 | 0.134463 ± 0.00985 | — (2 complete, 1 failed) |
| aniso_k100 | sliced_w1 | 1.47619 ± 0.206 | 1.20148 ± 0.148 | 1.25819 ± 0.0936 | 13.6145 ± 0.467 |
| aniso_k3 | sliced_w1 | 0.0469288 ± 0.00187 | 0.0490563 ± 0.00372 | 0.0477054 ± 0.00104 | 0.516602 ± 0.152 |
| aniso_k30 | sliced_w1 | 0.34036 ± 0.036 | 0.300452 ± 0.012 | 0.35292 ± 0.0311 | 4.83029 ± 0.464 |
| aniso_k300 | sliced_w1 | 6.04078 ± 0.522 | 2.5119 ± 0.343 | 3.74025 ± 0.406 | 191.466 ± 20.3 |
| audiomnist_stft | digit_acc | 0.788333 ± 0.0113 | 0.7765 ± 0.0135 | 0.786833 ± 0.00821 | 0.144833 ± 0.00691 |
| finance_ff49 | sliced_w1 | 0.181095 ± 0.00316 | 0.181921 ± 0.00277 | 0.192922 ± 0.00323 | 5.95169 ± 0.306 |
| gaussian_aniso_d16_cor | sliced_w1 | 0.122398 ± 0.014 | 0.139822 ± 0.0198 | 0.140565 ± 0.0201 | 1.18918 ± 0.283 |
| imagenette_dcae | fid | 147.957 ± 0.378 | 311.403 ± 11.1 | 227.437 ± 67.7 | 223.796 ± 6.27 |
| piv_d16 | sliced_w1 | 0.012025 ± 0.00127 | 0.0126237 ± 0.0011 | 0.0123003 ± 0.00104 | 0.0645829 ± 0.0214 |
| piv_d256 | sliced_w1 | 0.0232868 ± 0.00129 | 0.0243504 ± 0.00153 | 0.0248361 ± 0.0018 | — (3 failed) |
| piv_d32 | sliced_w1 | 0.0202315 ± 0.00157 | 0.0210898 ± 0.003 | 0.0206341 ± 0.00125 | 0.893375 ± 0.617 |
| piv_d64 | sliced_w1 | 0.028788 ± 0.00197 | 0.0290825 ± 0.00398 | 0.039156 ± 0.0018 | — (3 failed) |
| student_t_d128_df3.0_cor | sliced_w1 | 1.50997 ± 0.084 | 1.11209 ± 0.0896 | 1.10854 ± 0.0918 | — (3 failed) |
| student_t_d16_df1.5_cor | sliced_w1 | 2.11712 ± 0.121 | 1.86505 ± 0.0262 | 1.80441 ± 0.0396 | 271.166 ± 39.5 |
| student_t_d16_df10.0_cor | sliced_w1 | 0.158327 ± 0.0152 | 0.185174 ± 0.00553 | 0.184642 ± 0.00476 | 2.71041 ± 0.739 |
| student_t_d16_df2.0_cor | sliced_w1 | 0.627396 ± 0.0182 | 0.575626 ± 0.0517 | 0.570111 ± 0.0527 | 89.9565 ± 2.73 |
| student_t_d16_df3.0_cor | sliced_w1 | 0.310411 ± 0.0313 | 0.323474 ± 0.0355 | 0.319628 ± 0.0385 | 4.80983 ± 1.31 |
| student_t_d16_df5.0_cor | sliced_w1 | 0.20445 ± 0.02 | 0.213148 ± 0.0314 | 0.229948 ± 0.0144 | 3.73209 ± 1.79 |
| student_t_d16_df50.0_cor | sliced_w1 | 0.1811 ± 0.0455 | 0.200706 ± 0.0493 | 0.207581 ± 0.0458 | 3.1381 ± 1.39 |
| student_t_d256_df3.0_cor | sliced_w1 | 1.76484 ± 0.0796 | 1.21185 ± 0.051 | 1.20176 ± 0.0425 | — (3 failed) |
| student_t_d2_df3.0_cor | sliced_w1 | — (2 complete, 1 failed) | 0.435777 ± 0.0471 | 0.472682 ± 0.00828 | 0.819154 ± 0.112 |
| student_t_d32_df3.0_cor | sliced_w1 | 0.468922 ± 0.0366 | 0.418465 ± 0.0194 | 0.41523 ± 0.0253 | 7.95471 ± 1.64 |
| student_t_d64_df3.0_cor | sliced_w1 | 0.91761 ± 0.0666 | 0.733907 ± 0.116 | 0.728216 ± 0.114 | — (3 failed) |
| student_t_d8_df3.0_cor | sliced_w1 | 0.197703 ± 0.0269 | 0.182934 ± 0.013 | 0.242447 ± 0.0191 | 5.25255 ± 1.12 |
| toy_radial_angular | sliced_w1 | — (1 complete, 2 failed) | — (1 complete, 2 failed) | — (1 complete, 2 failed) | 0.296139 ± 0.11 |
| weather_au_wind | sliced_w1 | 0.0766653 ± 0.000749 | 0.0819225 ± 0.00282 | 0.0813396 ± 0.00158 | — (3 failed) |

## Measured findings

- B-A: 26 complete paired conditions; B has a better primary-metric mean in 10, a worse mean in 16, and ties in 0. Better on every seed in 7; worse on every seed in 8. These are condition counts, not pooled effect sizes.
- B-C: 27 complete paired conditions; B has a better primary-metric mean in 11, a worse mean in 16, and ties in 0. Better on every seed in 7; worse on every seed in 2. These are condition counts, not pooled effect sizes.

## Costs and protocol

| Condition | Method | Parameters | Training seconds | Sampling seconds |
|---|---|---|---|---|
| aniso_k1 | A | 41504 | 29.9516 ± 1.4 | 0.189125 ± 0.00188 |
| aniso_k1 | B | 41632 | 29.199 ± 0.13 | 0.28631 ± 0.00548 |
| aniso_k1 | C | 41632 | 29.8209 ± 0.733 | 0.252535 ± 0.00709 |
| aniso_k10 | A | 41504 | 28.4954 ± 1.85 | 0.192833 ± 0.00799 |
| aniso_k10 | B | 41632 | 29.6053 ± 4.02 | 0.282808 ± 0.000816 |
| aniso_k10 | C | 41632 | 28.9598 ± 0.25 | 0.253448 ± 0.00342 |
| aniso_k100 | A | 41504 | 28.7543 ± 1.87 | 0.187184 ± 0.000213 |
| aniso_k100 | B | 41632 | 27.2852 ± 0.639 | 0.284107 ± 0.00202 |
| aniso_k100 | C | 41632 | 26.8849 ± 0.177 | 0.249511 ± 0.00318 |
| aniso_k3 | A | 41504 | 28.6249 ± 1.95 | 0.191957 ± 0.00625 |
| aniso_k3 | B | 41632 | 29.5231 ± 1.07 | 0.290025 ± 0.00779 |
| aniso_k3 | C | 41632 | 29.1922 ± 1.78 | 0.254722 ± 0.00798 |
| aniso_k30 | A | 41504 | 27.8225 ± 2.01 | 0.187382 ± 0.000226 |
| aniso_k30 | B | 41632 | 29.1311 ± 0.551 | 0.29306 ± 0.00152 |
| aniso_k30 | C | 41632 | 29.3881 ± 2.06 | 0.255487 ± 0.00671 |
| aniso_k300 | A | 41504 | 27.991 ± 2.27 | 0.198728 ± 0.0077 |
| aniso_k300 | B | 41632 | 28.229 ± 1.15 | 0.282241 ± 0.00053 |
| aniso_k300 | C | 41632 | 29.4933 ± 1.43 | 0.251041 ± 0.00516 |
| audiomnist_stft | A | 27896802 | 8903.6 ± 19.5 | 873.544 ± 3.28 |
| audiomnist_stft | B | 27901186 | 8921.08 ± 13.5 | 876.744 ± 7.97 |
| audiomnist_stft | C | 27901186 | 8906.96 ± 11.8 | 869.893 ± 2.9 |
| finance_ff49 | A | 45873 | 30.5516 ± 0.274 | 0.148687 ± 0.006 |
| finance_ff49 | B | 46001 | 31.0265 ± 2.63 | 0.228026 ± 0.00749 |
| finance_ff49 | C | 46001 | 31.8723 ± 1.47 | 0.192296 ± 0.00489 |
| gaussian_aniso_d16_cor | A | 37392 | 28.0462 ± 1.1 | 0.171548 ± 0.000541 |
| gaussian_aniso_d16_cor | B | 37520 | 28.8855 ± 1.24 | 0.248022 ± 0.00383 |
| gaussian_aniso_d16_cor | C | 37520 | 29.5518 ± 1.5 | 0.225154 ± 0.00191 |
| imagenette_dcae | A | 32515616 | 1775.4 ± 33.1 | 49.1546 ± 0.755 |
| imagenette_dcae | B | 32522176 | 1805.7 ± 41.3 | 49.0762 ± 0.254 |
| imagenette_dcae | C | 32522176 | 1785.28 ± 15.5 | 49.11 ± 0.0955 |
| piv_d16 | A | 37392 | 30.6702 ± 5.14 | 0.177942 ± 0.00708 |
| piv_d16 | B | 37520 | 30.9233 ± 2.11 | 0.254992 ± 0.00206 |
| piv_d16 | C | 37520 | 31.0793 ± 1.48 | 0.229063 ± 0.00573 |
| piv_d256 | A | 99072 | 29.2802 ± 2.17 | 0.323549 ± 0.00284 |
| piv_d256 | B | 99200 | 31.7043 ± 4.29 | 0.442795 ± 0.00615 |
| piv_d256 | C | 99200 | 29.9871 ± 1.9 | 0.418333 ± 0.00645 |
| piv_d32 | A | 41504 | 30.3151 ± 0.525 | 0.18823 ± 0.00482 |
| piv_d32 | B | 41632 | 30.3454 ± 1.62 | 0.288978 ± 0.00365 |
| piv_d32 | C | 41632 | 29.627 ± 1.39 | 0.263008 ± 0.00298 |
| piv_d64 | A | 49728 | 31.7201 ± 4.89 | 0.21524 ± 0.00499 |
| piv_d64 | B | 49856 | 29.4505 ± 1.97 | 0.322287 ± 0.00292 |
| piv_d64 | C | 49856 | 30.6621 ± 0.563 | 0.295359 ± 0.00703 |
| student_t_d128_df3.0_cor | A | 66176 | 28.0057 ± 2.15 | 0.25957 ± 0.000994 |
| student_t_d128_df3.0_cor | B | 66304 | 29.5101 ± 1.73 | 0.412964 ± 0.00519 |
| student_t_d128_df3.0_cor | C | 66304 | 29.4188 ± 0.698 | 0.366984 ± 0.00739 |
| student_t_d16_df1.5_cor | A | 37392 | 27.201 ± 1.42 | 0.173043 ± 0.00168 |
| student_t_d16_df1.5_cor | B | 37520 | 28.4937 ± 1.51 | 0.255521 ± 0.00166 |
| student_t_d16_df1.5_cor | C | 37520 | 29.6707 ± 0.691 | 0.232372 ± 0.0067 |
| student_t_d16_df10.0_cor | A | 37392 | 27.9963 ± 1.4 | 0.182596 ± 0.00937 |
| student_t_d16_df10.0_cor | B | 37520 | 28.4805 ± 0.471 | 0.253895 ± 0.00094 |
| student_t_d16_df10.0_cor | C | 37520 | 29.8873 ± 1.02 | 0.226393 ± 0.00341 |
| student_t_d16_df2.0_cor | A | 37392 | 27.6914 ± 0.909 | 0.177961 ± 0.00866 |
| student_t_d16_df2.0_cor | B | 37520 | 30.3725 ± 4.88 | 0.254751 ± 0.00159 |
| student_t_d16_df2.0_cor | C | 37520 | 29.0208 ± 1.63 | 0.227622 ± 0.00436 |
| student_t_d16_df3.0_cor | A | 37392 | 28.781 ± 1.54 | 0.174132 ± 0.00215 |
| student_t_d16_df3.0_cor | B | 37520 | 30.2524 ± 1.69 | 0.265351 ± 0.00612 |
| student_t_d16_df3.0_cor | C | 37520 | 29.9041 ± 1.28 | 0.227943 ± 0.0055 |
| student_t_d16_df5.0_cor | A | 37392 | 28.0019 ± 1.51 | 0.172241 ± 0.00145 |
| student_t_d16_df5.0_cor | B | 37520 | 28.1324 ± 1.1 | 0.258299 ± 0.00427 |
| student_t_d16_df5.0_cor | C | 37520 | 28.6862 ± 0.63 | 0.228391 ± 0.0019 |
| student_t_d16_df50.0_cor | A | 37392 | 27.1946 ± 0.87 | 0.172532 ± 0.00038 |
| student_t_d16_df50.0_cor | B | 37520 | 28.8886 ± 1.65 | 0.2548 ± 0.00092 |
| student_t_d16_df50.0_cor | C | 37520 | 30.2752 ± 2.95 | 0.223911 ± 0.00101 |
| student_t_d256_df3.0_cor | A | 99072 | 27.3535 ± 0.891 | 0.326177 ± 0.00694 |
| student_t_d256_df3.0_cor | B | 99200 | 29.6656 ± 1.2 | 0.447702 ± 0.000674 |
| student_t_d256_df3.0_cor | C | 99200 | 27.8212 ± 1.29 | 0.411618 ± 0.000261 |
| student_t_d2_df3.0_cor | A | — (see per-seed records) | — (2 complete, 1 failed) | — (2 complete, 1 failed) |
| student_t_d2_df3.0_cor | B | 33922 | 30.1976 ± 1.3 | 0.248265 ± 0.00792 |
| student_t_d2_df3.0_cor | C | 33922 | 29.5117 ± 1.88 | 0.219524 ± 0.00695 |
| student_t_d32_df3.0_cor | A | 41504 | 30.5915 ± 2.76 | 0.187334 ± 0.000721 |
| student_t_d32_df3.0_cor | B | 41632 | 28.2699 ± 1.48 | 0.276602 ± 0.00324 |
| student_t_d32_df3.0_cor | C | 41632 | 30.6231 ± 0.383 | 0.248724 ± 0.000156 |
| student_t_d64_df3.0_cor | A | 49728 | 29.4447 ± 2.68 | 0.214585 ± 0.000493 |
| student_t_d64_df3.0_cor | B | 49856 | 29.8089 ± 1.53 | 0.336745 ± 0.00468 |
| student_t_d64_df3.0_cor | C | 49856 | 30.2207 ± 4.89 | 0.296558 ± 0.00382 |
| student_t_d8_df3.0_cor | A | 35336 | 28.7421 ± 1.19 | 0.170762 ± 0.000557 |
| student_t_d8_df3.0_cor | B | 35464 | 29.2805 ± 2.01 | 0.247496 ± 0.00782 |
| student_t_d8_df3.0_cor | C | 35464 | 30.9332 ± 3.15 | 0.216585 ± 0.000221 |
| toy_radial_angular | A | — (see per-seed records) | — (1 complete, 2 failed) | — (1 complete, 2 failed) |
| toy_radial_angular | B | — (see per-seed records) | — (1 complete, 2 failed) | — (1 complete, 2 failed) |
| toy_radial_angular | C | — (see per-seed records) | — (1 complete, 2 failed) | — (1 complete, 2 failed) |
| weather_au_wind | A | 57952 | 28.7054 ± 0.489 | 0.14456 ± 0.00103 |
| weather_au_wind | B | 58080 | 29.2762 ± 2 | 0.215265 ± 0.00449 |
| weather_au_wind | C | 58080 | 30.1516 ± 2.63 | 0.183938 ± 0.00287 |

Per-seed parameter overhead, peak memory, hardware, precision, actual network calls, radius drift and timing scopes are retained in abc_completion.json. Matched runtime ratios are included only where the recorded environments match.

## Existing t-Flow reference

65/84 complete, 19 failed, 0 missing; these counts are separate from the 252 A/B/C runs. Missing recovered-condition t-Flow jobs are not failures and do not block A/B/C completion. No tuning or training is performed by this reporter.

The qualified comparison concerns this published-noise-objective adaptation to matched backbones. Undefined metrics and nonfinite samples remain failures; finite metrics from failed runs are retained only per seed.

## All per-seed outcomes

| Condition | Method | Seed | Status | Primary value | Record |
|---|---|---|---|---|---|
| aniso_k1 | A | 8925 | complete | 0.020843118 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/A/seed_8925/result.json>) |
| aniso_k1 | A | 77395 | complete | 0.021204235 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/A/seed_77395/result.json>) |
| aniso_k1 | A | 65457 | complete | 0.021394288 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/A/seed_65457/result.json>) |
| aniso_k1 | B | 8925 | complete | 0.020838117 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/B/seed_8925/result.json>) |
| aniso_k1 | B | 77395 | complete | 0.020972019 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/B/seed_77395/result.json>) |
| aniso_k1 | B | 65457 | complete | 0.021686232 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/B/seed_65457/result.json>) |
| aniso_k1 | C | 8925 | complete | 0.02062978 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/C/seed_8925/result.json>) |
| aniso_k1 | C | 77395 | complete | 0.021325687 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/C/seed_77395/result.json>) |
| aniso_k1 | C | 65457 | complete | 0.021500941 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k1/C/seed_65457/result.json>) |
| aniso_k1 | tflow_reference | 8925 | complete | 0.56619513 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k1/seed_8925/result.json>) |
| aniso_k1 | tflow_reference | 77395 | complete | 0.45194119 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k1/seed_77395/result.json>) |
| aniso_k1 | tflow_reference | 65457 | complete | 0.49603885 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k1/seed_65457/result.json>) |
| aniso_k10 | A | 8925 | complete | 0.14504325 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/A/seed_8925/result.json>) |
| aniso_k10 | A | 77395 | complete | 0.14249727 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/A/seed_77395/result.json>) |
| aniso_k10 | A | 65457 | complete | 0.1154756 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/A/seed_65457/result.json>) |
| aniso_k10 | B | 8925 | complete | 0.20141555 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/B/seed_8925/result.json>) |
| aniso_k10 | B | 77395 | complete | 0.12878402 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/B/seed_77395/result.json>) |
| aniso_k10 | B | 65457 | complete | 0.13195403 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/B/seed_65457/result.json>) |
| aniso_k10 | C | 8925 | complete | 0.1466547 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/C/seed_8925/result.json>) |
| aniso_k10 | C | 77395 | complete | 0.13419817 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/C/seed_77395/result.json>) |
| aniso_k10 | C | 65457 | complete | 0.12253702 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k10/C/seed_65457/result.json>) |
| aniso_k10 | tflow_reference | 8925 | complete | 2.3203363 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k10/seed_8925/result.json>) |
| aniso_k10 | tflow_reference | 77395 | failed | 2.3383231 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k10/seed_77395/result.json>) |
| aniso_k10 | tflow_reference | 65457 | complete | 2.3100009 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k10/seed_65457/result.json>) |
| aniso_k100 | A | 8925 | complete | 1.2413416 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/A/seed_8925/result.json>) |
| aniso_k100 | A | 77395 | complete | 1.741948 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/A/seed_77395/result.json>) |
| aniso_k100 | A | 65457 | complete | 1.445294 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/A/seed_65457/result.json>) |
| aniso_k100 | B | 8925 | complete | 0.99408168 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/B/seed_8925/result.json>) |
| aniso_k100 | B | 77395 | complete | 1.2844746 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/B/seed_77395/result.json>) |
| aniso_k100 | B | 65457 | complete | 1.3258899 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/B/seed_65457/result.json>) |
| aniso_k100 | C | 8925 | complete | 1.1865475 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/C/seed_8925/result.json>) |
| aniso_k100 | C | 77395 | complete | 1.1976418 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/C/seed_77395/result.json>) |
| aniso_k100 | C | 65457 | complete | 1.3903714 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k100/C/seed_65457/result.json>) |
| aniso_k100 | tflow_reference | 8925 | complete | 13.196778 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k100/seed_8925/result.json>) |
| aniso_k100 | tflow_reference | 77395 | complete | 13.379903 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k100/seed_77395/result.json>) |
| aniso_k100 | tflow_reference | 65457 | complete | 14.266746 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k100/seed_65457/result.json>) |
| aniso_k3 | A | 8925 | complete | 0.048059374 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/A/seed_8925/result.json>) |
| aniso_k3 | A | 77395 | complete | 0.044294283 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/A/seed_77395/result.json>) |
| aniso_k3 | A | 65457 | complete | 0.048432775 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/A/seed_65457/result.json>) |
| aniso_k3 | B | 8925 | complete | 0.048828088 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/B/seed_8925/result.json>) |
| aniso_k3 | B | 77395 | complete | 0.044615399 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/B/seed_77395/result.json>) |
| aniso_k3 | B | 65457 | complete | 0.053725317 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/B/seed_65457/result.json>) |
| aniso_k3 | C | 8925 | complete | 0.048867993 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/C/seed_8925/result.json>) |
| aniso_k3 | C | 77395 | complete | 0.046347629 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/C/seed_77395/result.json>) |
| aniso_k3 | C | 65457 | complete | 0.047900442 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k3/C/seed_65457/result.json>) |
| aniso_k3 | tflow_reference | 8925 | complete | 0.39191639 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k3/seed_8925/result.json>) |
| aniso_k3 | tflow_reference | 77395 | complete | 0.42743716 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k3/seed_77395/result.json>) |
| aniso_k3 | tflow_reference | 65457 | complete | 0.73045182 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k3/seed_65457/result.json>) |
| aniso_k30 | A | 8925 | complete | 0.34864464 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/A/seed_8925/result.json>) |
| aniso_k30 | A | 77395 | complete | 0.37973881 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/A/seed_77395/result.json>) |
| aniso_k30 | A | 65457 | complete | 0.29269642 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/A/seed_65457/result.json>) |
| aniso_k30 | B | 8925 | complete | 0.31669116 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/B/seed_8925/result.json>) |
| aniso_k30 | B | 77395 | complete | 0.28802121 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/B/seed_77395/result.json>) |
| aniso_k30 | B | 65457 | complete | 0.2966429 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/B/seed_65457/result.json>) |
| aniso_k30 | C | 8925 | complete | 0.39605165 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/C/seed_8925/result.json>) |
| aniso_k30 | C | 77395 | complete | 0.33858457 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/C/seed_77395/result.json>) |
| aniso_k30 | C | 65457 | complete | 0.32412514 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k30/C/seed_65457/result.json>) |
| aniso_k30 | tflow_reference | 8925 | complete | 4.1774015 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k30/seed_8925/result.json>) |
| aniso_k30 | tflow_reference | 77395 | complete | 5.0974364 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k30/seed_77395/result.json>) |
| aniso_k30 | tflow_reference | 65457 | complete | 5.2160292 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k30/seed_65457/result.json>) |
| aniso_k300 | A | 8925 | complete | 6.7698479 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/A/seed_8925/result.json>) |
| aniso_k300 | A | 77395 | complete | 5.7739358 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/A/seed_77395/result.json>) |
| aniso_k300 | A | 65457 | complete | 5.5785642 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/A/seed_65457/result.json>) |
| aniso_k300 | B | 8925 | complete | 2.9050715 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/B/seed_8925/result.json>) |
| aniso_k300 | B | 77395 | complete | 2.0694077 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/B/seed_77395/result.json>) |
| aniso_k300 | B | 65457 | complete | 2.561234 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/B/seed_65457/result.json>) |
| aniso_k300 | C | 8925 | complete | 3.6322377 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/C/seed_8925/result.json>) |
| aniso_k300 | C | 77395 | complete | 3.3064373 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/C/seed_77395/result.json>) |
| aniso_k300 | C | 65457 | complete | 4.2820654 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/aniso_k300/C/seed_65457/result.json>) |
| aniso_k300 | tflow_reference | 8925 | complete | 216.24522 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k300/seed_8925/result.json>) |
| aniso_k300 | tflow_reference | 77395 | complete | 191.56604 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k300/seed_77395/result.json>) |
| aniso_k300 | tflow_reference | 65457 | complete | 166.58655 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k300/seed_65457/result.json>) |
| audiomnist_stft | A | 8925 | complete | 0.801 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/A/seed_8925/result.json>) |
| audiomnist_stft | A | 1234 | complete | 0.7735 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/A/seed_1234/result.json>) |
| audiomnist_stft | A | 7 | complete | 0.7905 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/A/seed_7/result.json>) |
| audiomnist_stft | B | 8925 | complete | 0.78 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/B/seed_8925/result.json>) |
| audiomnist_stft | B | 1234 | complete | 0.791 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/B/seed_1234/result.json>) |
| audiomnist_stft | B | 7 | complete | 0.7585 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/B/seed_7/result.json>) |
| audiomnist_stft | C | 8925 | complete | 0.784 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/C/seed_8925/result.json>) |
| audiomnist_stft | C | 1234 | complete | 0.798 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/C/seed_1234/result.json>) |
| audiomnist_stft | C | 7 | complete | 0.7785 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/audiomnist_stft/C/seed_7/result.json>) |
| audiomnist_stft | tflow_reference | 8925 | complete | 0.147 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/audiomnist_stft/seed_8925/result.json>) |
| audiomnist_stft | tflow_reference | 1234 | complete | 0.1355 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/audiomnist_stft/seed_1234/result.json>) |
| audiomnist_stft | tflow_reference | 7 | complete | 0.152 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/audiomnist_stft/seed_7/result.json>) |
| finance_ff49 | A | 8925 | complete | 0.1777522 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/A/seed_8925/result.json>) |
| finance_ff49 | A | 77395 | complete | 0.1801988 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/A/seed_77395/result.json>) |
| finance_ff49 | A | 65457 | complete | 0.18533491 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/A/seed_65457/result.json>) |
| finance_ff49 | B | 8925 | complete | 0.18173079 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/B/seed_8925/result.json>) |
| finance_ff49 | B | 77395 | complete | 0.17862552 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/B/seed_77395/result.json>) |
| finance_ff49 | B | 65457 | complete | 0.18540598 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/B/seed_65457/result.json>) |
| finance_ff49 | C | 8925 | complete | 0.1952467 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/C/seed_8925/result.json>) |
| finance_ff49 | C | 77395 | complete | 0.18835212 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/C/seed_77395/result.json>) |
| finance_ff49 | C | 65457 | complete | 0.19516811 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/finance_ff49/C/seed_65457/result.json>) |
| finance_ff49 | tflow_reference | 8925 | complete | 5.7633715 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/finance_ff49/seed_8925/result.json>) |
| finance_ff49 | tflow_reference | 77395 | complete | 5.7090354 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/finance_ff49/seed_77395/result.json>) |
| finance_ff49 | tflow_reference | 65457 | complete | 6.3826599 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/finance_ff49/seed_65457/result.json>) |
| gaussian_aniso_d16_cor | A | 8925 | complete | 0.10305622 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/A/seed_8925/result.json>) |
| gaussian_aniso_d16_cor | A | 77395 | complete | 0.12834162 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/A/seed_77395/result.json>) |
| gaussian_aniso_d16_cor | A | 65457 | complete | 0.13579492 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/A/seed_65457/result.json>) |
| gaussian_aniso_d16_cor | B | 8925 | complete | 0.13926604 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/B/seed_8925/result.json>) |
| gaussian_aniso_d16_cor | B | 77395 | complete | 0.11584435 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/B/seed_77395/result.json>) |
| gaussian_aniso_d16_cor | B | 65457 | complete | 0.16435662 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/B/seed_65457/result.json>) |
| gaussian_aniso_d16_cor | C | 8925 | complete | 0.12461072 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/C/seed_8925/result.json>) |
| gaussian_aniso_d16_cor | C | 77395 | complete | 0.12809657 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/C/seed_77395/result.json>) |
| gaussian_aniso_d16_cor | C | 65457 | complete | 0.16898893 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/gaussian_aniso_d16_cor/C/seed_65457/result.json>) |
| gaussian_aniso_d16_cor | tflow_reference | 8925 | complete | 1.3808029 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/gaussian_aniso_d16_cor/seed_8925/result.json>) |
| gaussian_aniso_d16_cor | tflow_reference | 77395 | complete | 0.78977382 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/gaussian_aniso_d16_cor/seed_77395/result.json>) |
| gaussian_aniso_d16_cor | tflow_reference | 65457 | complete | 1.3969766 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/gaussian_aniso_d16_cor/seed_65457/result.json>) |
| imagenette_dcae | A | 8925 | complete | 147.48065 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/A/seed_8925/result.json>) |
| imagenette_dcae | A | 1234 | complete | 147.98536 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/A/seed_1234/result.json>) |
| imagenette_dcae | A | 7 | complete | 148.4052 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/A/seed_7/result.json>) |
| imagenette_dcae | B | 8925 | complete | 324.62677 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/B/seed_8925/result.json>) |
| imagenette_dcae | B | 1234 | complete | 297.56142 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/B/seed_1234/result.json>) |
| imagenette_dcae | B | 7 | complete | 312.02024 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/B/seed_7/result.json>) |
| imagenette_dcae | C | 8925 | complete | 182.24609 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/C/seed_8925/result.json>) |
| imagenette_dcae | C | 1234 | complete | 323.16382 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/C/seed_1234/result.json>) |
| imagenette_dcae | C | 7 | complete | 176.90036 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/imagenette_dcae/C/seed_7/result.json>) |
| imagenette_dcae | tflow_reference | 8925 | complete | 215.62454 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/imagenette_dcae/seed_8925/result.json>) |
| imagenette_dcae | tflow_reference | 1234 | complete | 230.86925 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/imagenette_dcae/seed_1234/result.json>) |
| imagenette_dcae | tflow_reference | 7 | complete | 224.89443 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/imagenette_dcae/seed_7/result.json>) |
| piv_d16 | A | 8925 | complete | 0.012023802 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/A/seed_8925/result.json>) |
| piv_d16 | A | 77395 | complete | 0.01046734 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/A/seed_77395/result.json>) |
| piv_d16 | A | 65457 | complete | 0.013583726 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/A/seed_65457/result.json>) |
| piv_d16 | B | 8925 | complete | 0.012550483 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/B/seed_8925/result.json>) |
| piv_d16 | B | 77395 | complete | 0.011311729 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/B/seed_77395/result.json>) |
| piv_d16 | B | 65457 | complete | 0.014008744 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/B/seed_65457/result.json>) |
| piv_d16 | C | 8925 | complete | 0.011949611 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/C/seed_8925/result.json>) |
| piv_d16 | C | 77395 | complete | 0.011236212 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/C/seed_77395/result.json>) |
| piv_d16 | C | 65457 | complete | 0.013715027 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d16/C/seed_65457/result.json>) |
| piv_d16 | tflow_reference | 8925 | complete | 0.054612149 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d16/seed_8925/result.json>) |
| piv_d16 | tflow_reference | 77395 | complete | 0.094367951 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d16/seed_77395/result.json>) |
| piv_d16 | tflow_reference | 65457 | complete | 0.044768505 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d16/seed_65457/result.json>) |
| piv_d256 | A | 8925 | complete | 0.024627589 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/A/seed_8925/result.json>) |
| piv_d256 | A | 77395 | complete | 0.021541212 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/A/seed_77395/result.json>) |
| piv_d256 | A | 65457 | complete | 0.023691455 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/A/seed_65457/result.json>) |
| piv_d256 | B | 8925 | complete | 0.026371974 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/B/seed_8925/result.json>) |
| piv_d256 | B | 77395 | complete | 0.022657197 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/B/seed_77395/result.json>) |
| piv_d256 | B | 65457 | complete | 0.024022168 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/B/seed_65457/result.json>) |
| piv_d256 | C | 8925 | complete | 0.02714394 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/C/seed_8925/result.json>) |
| piv_d256 | C | 77395 | complete | 0.022747993 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/C/seed_77395/result.json>) |
| piv_d256 | C | 65457 | complete | 0.024616497 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d256/C/seed_65457/result.json>) |
| piv_d256 | tflow_reference | 8925 | failed | 72.586609 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d256/seed_8925/result.json>) |
| piv_d256 | tflow_reference | 77395 | failed | 72.490273 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d256/seed_77395/result.json>) |
| piv_d256 | tflow_reference | 65457 | failed | 72.731689 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d256/seed_65457/result.json>) |
| piv_d32 | A | 8925 | complete | 0.020979734 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/A/seed_8925/result.json>) |
| piv_d32 | A | 77395 | complete | 0.021674331 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/A/seed_77395/result.json>) |
| piv_d32 | A | 65457 | complete | 0.018040571 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/A/seed_65457/result.json>) |
| piv_d32 | B | 8925 | complete | 0.025045155 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/B/seed_8925/result.json>) |
| piv_d32 | B | 77395 | complete | 0.020455757 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/B/seed_77395/result.json>) |
| piv_d32 | B | 65457 | complete | 0.017768575 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/B/seed_65457/result.json>) |
| piv_d32 | C | 8925 | complete | 0.022188757 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/C/seed_8925/result.json>) |
| piv_d32 | C | 77395 | complete | 0.020591155 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/C/seed_77395/result.json>) |
| piv_d32 | C | 65457 | complete | 0.019122416 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d32/C/seed_65457/result.json>) |
| piv_d32 | tflow_reference | 8925 | complete | 0.42530334 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d32/seed_8925/result.json>) |
| piv_d32 | tflow_reference | 77395 | complete | 0.49008131 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d32/seed_77395/result.json>) |
| piv_d32 | tflow_reference | 65457 | complete | 1.7647396 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d32/seed_65457/result.json>) |
| piv_d64 | A | 8925 | complete | 0.027250763 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/A/seed_8925/result.json>) |
| piv_d64 | A | 77395 | complete | 0.027542265 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/A/seed_77395/result.json>) |
| piv_d64 | A | 65457 | complete | 0.031571094 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/A/seed_65457/result.json>) |
| piv_d64 | B | 8925 | complete | 0.024801075 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/B/seed_8925/result.json>) |
| piv_d64 | B | 77395 | complete | 0.028065497 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/B/seed_77395/result.json>) |
| piv_d64 | B | 65457 | complete | 0.034381043 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/B/seed_65457/result.json>) |
| piv_d64 | C | 8925 | complete | 0.038323689 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/C/seed_8925/result.json>) |
| piv_d64 | C | 77395 | complete | 0.037486117 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/C/seed_77395/result.json>) |
| piv_d64 | C | 65457 | complete | 0.041658312 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/piv_d64/C/seed_65457/result.json>) |
| piv_d64 | tflow_reference | 8925 | failed | 9.777153 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d64/seed_8925/result.json>) |
| piv_d64 | tflow_reference | 77395 | failed | 10.421619 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d64/seed_77395/result.json>) |
| piv_d64 | tflow_reference | 65457 | failed | 9.6439199 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d64/seed_65457/result.json>) |
| student_t_d128_df3.0_cor | A | 8925 | complete | 1.6286643 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d128_df3.0_cor | A | 77395 | complete | 1.446135 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d128_df3.0_cor | A | 65457 | complete | 1.4551144 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d128_df3.0_cor | B | 8925 | complete | 0.99860662 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d128_df3.0_cor | B | 77395 | complete | 1.1198771 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d128_df3.0_cor | B | 65457 | complete | 1.2177852 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d128_df3.0_cor | C | 8925 | complete | 0.99057549 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d128_df3.0_cor | C | 77395 | complete | 1.1207283 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d128_df3.0_cor | C | 65457 | complete | 1.2143312 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d128_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d128_df3.0_cor | tflow_reference | 8925 | failed | 4096.5181 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d128_df3.0_cor/seed_8925/result.json>) |
| student_t_d128_df3.0_cor | tflow_reference | 77395 | failed | 4028.3938 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d128_df3.0_cor/seed_77395/result.json>) |
| student_t_d128_df3.0_cor | tflow_reference | 65457 | failed | 4086.2446 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d128_df3.0_cor/seed_65457/result.json>) |
| student_t_d16_df1.5_cor | A | 8925 | complete | 2.0229745 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/A/seed_8925/result.json>) |
| student_t_d16_df1.5_cor | A | 77395 | complete | 2.2883961 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/A/seed_77395/result.json>) |
| student_t_d16_df1.5_cor | A | 65457 | complete | 2.0399983 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/A/seed_65457/result.json>) |
| student_t_d16_df1.5_cor | B | 8925 | complete | 1.8536872 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/B/seed_8925/result.json>) |
| student_t_d16_df1.5_cor | B | 77395 | complete | 1.9012088 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/B/seed_77395/result.json>) |
| student_t_d16_df1.5_cor | B | 65457 | complete | 1.8402512 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/B/seed_65457/result.json>) |
| student_t_d16_df1.5_cor | C | 8925 | complete | 1.8538908 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/C/seed_8925/result.json>) |
| student_t_d16_df1.5_cor | C | 77395 | complete | 1.80248 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/C/seed_77395/result.json>) |
| student_t_d16_df1.5_cor | C | 65457 | complete | 1.7568578 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df1.5_cor/C/seed_65457/result.json>) |
| student_t_d16_df1.5_cor | tflow_reference | 8925 | complete | 326.49496 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df1.5_cor/seed_8925/result.json>) |
| student_t_d16_df1.5_cor | tflow_reference | 77395 | complete | 249.75523 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df1.5_cor/seed_77395/result.json>) |
| student_t_d16_df1.5_cor | tflow_reference | 65457 | complete | 237.24803 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df1.5_cor/seed_65457/result.json>) |
| student_t_d16_df10.0_cor | A | 8925 | complete | 0.13687539 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/A/seed_8925/result.json>) |
| student_t_d16_df10.0_cor | A | 77395 | complete | 0.17076883 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/A/seed_77395/result.json>) |
| student_t_d16_df10.0_cor | A | 65457 | complete | 0.16733548 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/A/seed_65457/result.json>) |
| student_t_d16_df10.0_cor | B | 8925 | complete | 0.17885694 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/B/seed_8925/result.json>) |
| student_t_d16_df10.0_cor | B | 77395 | complete | 0.19232467 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/B/seed_77395/result.json>) |
| student_t_d16_df10.0_cor | B | 65457 | complete | 0.18433996 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/B/seed_65457/result.json>) |
| student_t_d16_df10.0_cor | C | 8925 | complete | 0.17940992 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/C/seed_8925/result.json>) |
| student_t_d16_df10.0_cor | C | 77395 | complete | 0.19091782 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/C/seed_77395/result.json>) |
| student_t_d16_df10.0_cor | C | 65457 | complete | 0.18359929 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df10.0_cor/C/seed_65457/result.json>) |
| student_t_d16_df10.0_cor | tflow_reference | 8925 | complete | 3.5599613 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df10.0_cor/seed_8925/result.json>) |
| student_t_d16_df10.0_cor | tflow_reference | 77395 | complete | 2.8131757 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df10.0_cor/seed_77395/result.json>) |
| student_t_d16_df10.0_cor | tflow_reference | 65457 | complete | 1.7580965 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df10.0_cor/seed_65457/result.json>) |
| student_t_d16_df2.0_cor | A | 8925 | complete | 0.64685363 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/A/seed_8925/result.json>) |
| student_t_d16_df2.0_cor | A | 77395 | complete | 0.60303158 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/A/seed_77395/result.json>) |
| student_t_d16_df2.0_cor | A | 65457 | complete | 0.63230419 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/A/seed_65457/result.json>) |
| student_t_d16_df2.0_cor | B | 8925 | complete | 0.61548561 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/B/seed_8925/result.json>) |
| student_t_d16_df2.0_cor | B | 77395 | complete | 0.50265127 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/B/seed_77395/result.json>) |
| student_t_d16_df2.0_cor | B | 65457 | complete | 0.60874218 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/B/seed_65457/result.json>) |
| student_t_d16_df2.0_cor | C | 8925 | complete | 0.62408555 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/C/seed_8925/result.json>) |
| student_t_d16_df2.0_cor | C | 77395 | complete | 0.498687 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/C/seed_77395/result.json>) |
| student_t_d16_df2.0_cor | C | 65457 | complete | 0.58756059 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df2.0_cor/C/seed_65457/result.json>) |
| student_t_d16_df2.0_cor | tflow_reference | 8925 | complete | 93.280937 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df2.0_cor/seed_8925/result.json>) |
| student_t_d16_df2.0_cor | tflow_reference | 77395 | complete | 89.993309 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df2.0_cor/seed_77395/result.json>) |
| student_t_d16_df2.0_cor | tflow_reference | 65457 | complete | 86.595284 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df2.0_cor/seed_65457/result.json>) |
| student_t_d16_df3.0_cor | A | 8925 | complete | 0.30215353 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d16_df3.0_cor | A | 77395 | complete | 0.27686328 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d16_df3.0_cor | A | 65457 | complete | 0.35221761 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d16_df3.0_cor | B | 8925 | complete | 0.3053911 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d16_df3.0_cor | B | 77395 | complete | 0.29192063 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d16_df3.0_cor | B | 65457 | complete | 0.37310952 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d16_df3.0_cor | C | 8925 | complete | 0.32136226 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d16_df3.0_cor | C | 77395 | complete | 0.2716153 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d16_df3.0_cor | C | 65457 | complete | 0.36590552 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d16_df3.0_cor | tflow_reference | 8925 | complete | 5.6929603 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df3.0_cor/seed_8925/result.json>) |
| student_t_d16_df3.0_cor | tflow_reference | 77395 | complete | 5.7793703 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df3.0_cor/seed_77395/result.json>) |
| student_t_d16_df3.0_cor | tflow_reference | 65457 | complete | 2.9571598 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df3.0_cor/seed_65457/result.json>) |
| student_t_d16_df5.0_cor | A | 8925 | complete | 0.17775659 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/A/seed_8925/result.json>) |
| student_t_d16_df5.0_cor | A | 77395 | complete | 0.22588401 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/A/seed_77395/result.json>) |
| student_t_d16_df5.0_cor | A | 65457 | complete | 0.20971012 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/A/seed_65457/result.json>) |
| student_t_d16_df5.0_cor | B | 8925 | complete | 0.17441516 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/B/seed_8925/result.json>) |
| student_t_d16_df5.0_cor | B | 77395 | complete | 0.21377833 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/B/seed_77395/result.json>) |
| student_t_d16_df5.0_cor | B | 65457 | complete | 0.25125089 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/B/seed_65457/result.json>) |
| student_t_d16_df5.0_cor | C | 8925 | complete | 0.2109547 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/C/seed_8925/result.json>) |
| student_t_d16_df5.0_cor | C | 77395 | complete | 0.23321395 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/C/seed_77395/result.json>) |
| student_t_d16_df5.0_cor | C | 65457 | complete | 0.24567544 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df5.0_cor/C/seed_65457/result.json>) |
| student_t_d16_df5.0_cor | tflow_reference | 8925 | complete | 6.2532406 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df5.0_cor/seed_8925/result.json>) |
| student_t_d16_df5.0_cor | tflow_reference | 77395 | complete | 2.6618013 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df5.0_cor/seed_77395/result.json>) |
| student_t_d16_df5.0_cor | tflow_reference | 65457 | complete | 2.2812345 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df5.0_cor/seed_65457/result.json>) |
| student_t_d16_df50.0_cor | A | 8925 | complete | 0.13869588 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/A/seed_8925/result.json>) |
| student_t_d16_df50.0_cor | A | 77395 | complete | 0.24417125 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/A/seed_77395/result.json>) |
| student_t_d16_df50.0_cor | A | 65457 | complete | 0.16043265 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/A/seed_65457/result.json>) |
| student_t_d16_df50.0_cor | B | 8925 | complete | 0.141317 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/B/seed_8925/result.json>) |
| student_t_d16_df50.0_cor | B | 77395 | complete | 0.26195797 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/B/seed_77395/result.json>) |
| student_t_d16_df50.0_cor | B | 65457 | complete | 0.19884166 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/B/seed_65457/result.json>) |
| student_t_d16_df50.0_cor | C | 8925 | complete | 0.15139957 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/C/seed_8925/result.json>) |
| student_t_d16_df50.0_cor | C | 77395 | complete | 0.26350319 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/C/seed_77395/result.json>) |
| student_t_d16_df50.0_cor | C | 65457 | complete | 0.20783904 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d16_df50.0_cor/C/seed_65457/result.json>) |
| student_t_d16_df50.0_cor | tflow_reference | 8925 | complete | 5.0777893 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df50.0_cor/seed_8925/result.json>) |
| student_t_d16_df50.0_cor | tflow_reference | 77395 | complete | 2.4614344 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df50.0_cor/seed_77395/result.json>) |
| student_t_d16_df50.0_cor | tflow_reference | 65457 | complete | 1.8750814 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d16_df50.0_cor/seed_65457/result.json>) |
| student_t_d256_df3.0_cor | A | 8925 | complete | 1.6607053 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d256_df3.0_cor | A | 77395 | complete | 1.85383 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d256_df3.0_cor | A | 65457 | complete | 1.779982 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d256_df3.0_cor | B | 8925 | complete | 1.1397771 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d256_df3.0_cor | B | 77395 | complete | 1.2452443 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d256_df3.0_cor | B | 65457 | complete | 1.2505208 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d256_df3.0_cor | C | 8925 | complete | 1.1421168 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d256_df3.0_cor | C | 77395 | complete | 1.2376775 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d256_df3.0_cor | C | 65457 | complete | 1.2254837 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d256_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d256_df3.0_cor | tflow_reference | 8925 | failed | 8701.0332 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d256_df3.0_cor/seed_8925/result.json>) |
| student_t_d256_df3.0_cor | tflow_reference | 77395 | failed | 8726.2881 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d256_df3.0_cor/seed_77395/result.json>) |
| student_t_d256_df3.0_cor | tflow_reference | 65457 | failed | 8730.1641 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d256_df3.0_cor/seed_65457/result.json>) |
| student_t_d2_df3.0_cor | A | 8925 | complete | 0.42651856 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d2_df3.0_cor | A | 77395 | complete | 0.38326511 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d2_df3.0_cor | A | 65457 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d2_df3.0_cor | B | 8925 | complete | 0.47281781 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d2_df3.0_cor | B | 77395 | complete | 0.36936831 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d2_df3.0_cor | B | 65457 | complete | 0.46514356 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d2_df3.0_cor | C | 8925 | complete | 0.46101651 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d2_df3.0_cor | C | 77395 | complete | 0.479343 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d2_df3.0_cor | C | 65457 | complete | 0.47768739 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d2_df3.0_cor | tflow_reference | 8925 | complete | 0.97364497 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d2_df3.0_cor/seed_8925/result.json>) |
| student_t_d2_df3.0_cor | tflow_reference | 77395 | complete | 0.7717213 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d2_df3.0_cor/seed_77395/result.json>) |
| student_t_d2_df3.0_cor | tflow_reference | 65457 | complete | 0.7120949 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d2_df3.0_cor/seed_65457/result.json>) |
| student_t_d32_df3.0_cor | A | 8925 | complete | 0.49914789 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d32_df3.0_cor | A | 77395 | complete | 0.4173798 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d32_df3.0_cor | A | 65457 | complete | 0.4902392 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d32_df3.0_cor | B | 8925 | complete | 0.40444437 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d32_df3.0_cor | B | 77395 | complete | 0.40509811 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d32_df3.0_cor | B | 65457 | complete | 0.44585219 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d32_df3.0_cor | C | 8925 | complete | 0.39793 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d32_df3.0_cor | C | 77395 | complete | 0.39672139 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d32_df3.0_cor | C | 65457 | complete | 0.45103851 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d32_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d32_df3.0_cor | tflow_reference | 8925 | complete | 10.226751 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d32_df3.0_cor/seed_8925/result.json>) |
| student_t_d32_df3.0_cor | tflow_reference | 77395 | complete | 7.2304096 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d32_df3.0_cor/seed_77395/result.json>) |
| student_t_d32_df3.0_cor | tflow_reference | 65457 | complete | 6.4069595 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d32_df3.0_cor/seed_65457/result.json>) |
| student_t_d64_df3.0_cor | A | 8925 | complete | 0.85258222 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d64_df3.0_cor | A | 77395 | complete | 0.89109987 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d64_df3.0_cor | A | 65457 | complete | 1.009148 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d64_df3.0_cor | B | 8925 | complete | 0.89271921 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d64_df3.0_cor | B | 77395 | complete | 0.68944943 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d64_df3.0_cor | B | 65457 | complete | 0.61955351 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d64_df3.0_cor | C | 8925 | complete | 0.88113391 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d64_df3.0_cor | C | 77395 | complete | 0.69624698 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d64_df3.0_cor | C | 65457 | complete | 0.60726774 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d64_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d64_df3.0_cor | tflow_reference | 8925 | failed | 617.51355 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d64_df3.0_cor/seed_8925/result.json>) |
| student_t_d64_df3.0_cor | tflow_reference | 77395 | failed | 570.93591 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d64_df3.0_cor/seed_77395/result.json>) |
| student_t_d64_df3.0_cor | tflow_reference | 65457 | failed | 705.21075 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d64_df3.0_cor/seed_65457/result.json>) |
| student_t_d8_df3.0_cor | A | 8925 | complete | 0.16208304 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/A/seed_8925/result.json>) |
| student_t_d8_df3.0_cor | A | 77395 | complete | 0.20385841 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/A/seed_77395/result.json>) |
| student_t_d8_df3.0_cor | A | 65457 | complete | 0.22716725 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/A/seed_65457/result.json>) |
| student_t_d8_df3.0_cor | B | 8925 | complete | 0.16474107 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/B/seed_8925/result.json>) |
| student_t_d8_df3.0_cor | B | 77395 | complete | 0.18935642 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/B/seed_77395/result.json>) |
| student_t_d8_df3.0_cor | B | 65457 | complete | 0.19470359 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/B/seed_65457/result.json>) |
| student_t_d8_df3.0_cor | C | 8925 | complete | 0.21576577 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/C/seed_8925/result.json>) |
| student_t_d8_df3.0_cor | C | 77395 | complete | 0.25197834 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/C/seed_77395/result.json>) |
| student_t_d8_df3.0_cor | C | 65457 | complete | 0.25959799 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d8_df3.0_cor/C/seed_65457/result.json>) |
| student_t_d8_df3.0_cor | tflow_reference | 8925 | complete | 4.8856535 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d8_df3.0_cor/seed_8925/result.json>) |
| student_t_d8_df3.0_cor | tflow_reference | 77395 | complete | 4.1022434 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d8_df3.0_cor/seed_77395/result.json>) |
| student_t_d8_df3.0_cor | tflow_reference | 65457 | complete | 6.7697449 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d8_df3.0_cor/seed_65457/result.json>) |
| toy_radial_angular | A | 8925 | complete | 0.65533823 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/A/seed_8925/result.json>) |
| toy_radial_angular | A | 77395 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/A/seed_77395/result.json>) |
| toy_radial_angular | A | 65457 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/A/seed_65457/result.json>) |
| toy_radial_angular | B | 8925 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/B/seed_8925/result.json>) |
| toy_radial_angular | B | 77395 | complete | 71.365982 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/B/seed_77395/result.json>) |
| toy_radial_angular | B | 65457 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/B/seed_65457/result.json>) |
| toy_radial_angular | C | 8925 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/C/seed_8925/result.json>) |
| toy_radial_angular | C | 77395 | failed | — | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/C/seed_77395/result.json>) |
| toy_radial_angular | C | 65457 | complete | 0.99493939 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/C/seed_65457/result.json>) |
| toy_radial_angular | tflow_reference | 8925 | complete | 0.38544205 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/toy_radial_angular/seed_8925/result.json>) |
| toy_radial_angular | tflow_reference | 77395 | complete | 0.14136679 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/toy_radial_angular/seed_77395/result.json>) |
| toy_radial_angular | tflow_reference | 65457 | complete | 0.3616077 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/toy_radial_angular/seed_65457/result.json>) |
| weather_au_wind | A | 8925 | complete | 0.075812772 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/A/seed_8925/result.json>) |
| weather_au_wind | A | 77395 | complete | 0.07654617 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/A/seed_77395/result.json>) |
| weather_au_wind | A | 65457 | complete | 0.077636957 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/A/seed_65457/result.json>) |
| weather_au_wind | B | 8925 | complete | 0.078426704 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/B/seed_8925/result.json>) |
| weather_au_wind | B | 77395 | complete | 0.082020611 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/B/seed_77395/result.json>) |
| weather_au_wind | B | 65457 | complete | 0.085320175 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/B/seed_65457/result.json>) |
| weather_au_wind | C | 8925 | complete | 0.080385394 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/C/seed_8925/result.json>) |
| weather_au_wind | C | 77395 | complete | 0.083560839 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/C/seed_77395/result.json>) |
| weather_au_wind | C | 65457 | complete | 0.080072507 | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/weather_au_wind/C/seed_65457/result.json>) |
| weather_au_wind | tflow_reference | 8925 | failed | 195.85287 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/weather_au_wind/seed_8925/result.json>) |
| weather_au_wind | tflow_reference | 77395 | failed | 198.82127 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/weather_au_wind/seed_77395/result.json>) |
| weather_au_wind | tflow_reference | 65457 | failed | 194.36093 (partial; excluded from means) | [result.json](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/weather_au_wind/seed_65457/result.json>) |

## Failures and unresolved records

- aniso_k10 / tflow_reference / seed 77395: failed; angular_sw_bin3.
- piv_d256 / tflow_reference / seed 8925: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- piv_d256 / tflow_reference / seed 77395: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- piv_d256 / tflow_reference / seed 65457: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- piv_d64 / tflow_reference / seed 8925: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2.
- piv_d64 / tflow_reference / seed 77395: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2.
- piv_d64 / tflow_reference / seed 65457: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2.
- student_t_d128_df3.0_cor / tflow_reference / seed 8925: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- student_t_d128_df3.0_cor / tflow_reference / seed 77395: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- student_t_d128_df3.0_cor / tflow_reference / seed 65457: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- student_t_d256_df3.0_cor / tflow_reference / seed 8925: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- student_t_d256_df3.0_cor / tflow_reference / seed 77395: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- student_t_d256_df3.0_cor / tflow_reference / seed 65457: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- student_t_d2_df3.0_cor / A / seed 65457: failed; Nonfinite samples in batch starting at 0.
- student_t_d64_df3.0_cor / tflow_reference / seed 8925: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2.
- student_t_d64_df3.0_cor / tflow_reference / seed 77395: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2.
- student_t_d64_df3.0_cor / tflow_reference / seed 65457: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2.
- toy_radial_angular / A / seed 77395: failed; Nonfinite samples in batch starting at 0.
- toy_radial_angular / A / seed 65457: failed; Nonfinite samples in batch starting at 0.
- toy_radial_angular / B / seed 8925: failed; Nonfinite samples in batch starting at 0.
- toy_radial_angular / B / seed 65457: failed; Nonfinite samples in batch starting at 0.
- toy_radial_angular / C / seed 8925: failed; Nonfinite samples in batch starting at 0.
- toy_radial_angular / C / seed 77395: failed; Nonfinite samples in batch starting at 0.
- weather_au_wind / tflow_reference / seed 8925: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- weather_au_wind / tflow_reference / seed 77395: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.
- weather_au_wind / tflow_reference / seed 65457: failed; angular_sw_bin0, angular_sw_bin1, angular_sw_bin2, angular_sw_bin3, angular_sw_mean.

## Interpretation limits

- New explicitly seeded synthetic caches are shared by the new methods; they are not recovered historical realizations.
- Three seeds and differences in observed means alone do not establish statistical significance.
- Measured runtime comparisons require the same recorded hardware/software/precision and retain timing-scope limitations.
- The existing t-Flow adaptation has documented failures and sampler/backbone limitations; it is a qualified reference, not an ABC completion requirement.
- The inherited RAFM near-antipodal path routine can produce excessively large finite angular targets. Toy sampling failures remain scientific failures; the audit identifies a numerical defect but does not attribute each failed run to a specific unsaved training example or integration step.
- [Failure analysis: rafm_toy_numerical_failure_analysis.md](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/docs/rafm_toy_numerical_failure_analysis.md>).
- [Failure analysis: tflow_matched_backbone_failure_analysis.md](</mnt/vast01/users/fouad.oubari/msgm/rafm-additions/docs/tflow_matched_backbone_failure_analysis.md>).
- Image reference protocol is preserved as authorized: 3925 PNGs, overlap with generator train/validation/test = 2339/775/811. This is not a disjoint held-out image reference. No reference regeneration or protocol correction was applied.
- imagenette_dcae: Data .pt/.npz files are gitignored and carry no commit stamp. Script commits below are the LATEST commit touching each script, NOT a verified proof that the on-disk artifacts were produced by that exact commit. The exact extraction/training commit is UNVERIFIED.
- piv_d32: Data .pt/.npz files are gitignored and carry no commit stamp. Script commits below are the LATEST commit touching each script, NOT a verified proof that the on-disk artifacts were produced by that exact commit. The exact extraction/training commit is UNVERIFIED.

The fixed-spherical + empirical-gain AudioMNIST result remains a separately verified checkpoint reference in the JSON/audio artifacts. It does not replace arm A or the RAFM-Ang versus RAFM-Vel ablation.
