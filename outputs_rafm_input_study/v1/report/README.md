# RAFM-Ang input parameterization and t-Flow study

Status: **incomplete**. 0/336 final seeds verified; 0/112 three-seed method groups complete.

Three-seed means use population standard deviation. Missing, failed and incompatible groups have no mean. Training directories do not establish that a scheduler job is currently running.

| Condition | Comparison | A | B | C | t-Flow | Blockers |
|---|---|---|---|---|---|---|
| aniso_k1 | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| aniso_k10 | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| aniso_k100 | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| aniso_k3 | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| aniso_k30 | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| aniso_k300 | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| audiomnist_stft | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| finance_ff49 | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| gaussian_aniso_d16_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| imagenette_dcae | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | image_split_conflict, historical_eval_provenance |
| piv_d16 | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| piv_d256 | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| piv_d32 | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | piv32_identity_missing |
| piv_d64 | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d128_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d16_df1.5_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d16_df10.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d16_df2.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d16_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d16_df5.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d16_df50.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d256_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d2_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d32_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d64_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| student_t_d8_df3.0_cor | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| toy_radial_angular | new_matched_realization | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |
| weather_au_wind | new_matched_runs_on_verified_existing_cache | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | 0/3 incomplete | — |

## Failures and compatibility issues

No final-run failure records are present in this snapshot. This does not imply missing runs passed.

## Preserved implementation and startup failures

- Implementation-check job 765729: 46 passed / 4 failed; Sampler called CUDA peak-memory reset for CPU unit fixtures. Resolution: Guard CUDA peak reset by device.type; CPU peak_memory stays None. Benchmark results affected: False.
- Earlier t-Flow attempt /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v1: cancelled_after_startup_failure; 5 startup candidate failures, 0 observed optimizer updates, 0 checkpoints. These cancelled attempts are separate from current scientific seed outcomes. Resolution: Explicit GPU initialization in common orchestration entrypoint; retain runtime/config bytes and validate actual public trainer in fresh processes before restarting to v2.

Current check results and dated worker/scheduler snapshots are preserved in report.json; this collector makes no live scheduler queries.

## Measured comparison

| Condition | Method | Primary metric | Mean ± population SD | Angular SW | Train seconds | Sample seconds |
|---|---|---|---|---|---|---|
| aniso_k1 | A | sliced_w1 | — | — | — | — |
| aniso_k1 | B | sliced_w1 | — | — | — | — |
| aniso_k1 | C | sliced_w1 | — | — | — | — |
| aniso_k1 | tflow | sliced_w1 | — | — | — | — |
| aniso_k10 | A | sliced_w1 | — | — | — | — |
| aniso_k10 | B | sliced_w1 | — | — | — | — |
| aniso_k10 | C | sliced_w1 | — | — | — | — |
| aniso_k10 | tflow | sliced_w1 | — | — | — | — |
| aniso_k100 | A | sliced_w1 | — | — | — | — |
| aniso_k100 | B | sliced_w1 | — | — | — | — |
| aniso_k100 | C | sliced_w1 | — | — | — | — |
| aniso_k100 | tflow | sliced_w1 | — | — | — | — |
| aniso_k3 | A | sliced_w1 | — | — | — | — |
| aniso_k3 | B | sliced_w1 | — | — | — | — |
| aniso_k3 | C | sliced_w1 | — | — | — | — |
| aniso_k3 | tflow | sliced_w1 | — | — | — | — |
| aniso_k30 | A | sliced_w1 | — | — | — | — |
| aniso_k30 | B | sliced_w1 | — | — | — | — |
| aniso_k30 | C | sliced_w1 | — | — | — | — |
| aniso_k30 | tflow | sliced_w1 | — | — | — | — |
| aniso_k300 | A | sliced_w1 | — | — | — | — |
| aniso_k300 | B | sliced_w1 | — | — | — | — |
| aniso_k300 | C | sliced_w1 | — | — | — | — |
| aniso_k300 | tflow | sliced_w1 | — | — | — | — |
| audiomnist_stft | A | digit_acc | — | — | — | — |
| audiomnist_stft | B | digit_acc | — | — | — | — |
| audiomnist_stft | C | digit_acc | — | — | — | — |
| audiomnist_stft | tflow | digit_acc | — | — | — | — |
| finance_ff49 | A | sliced_w1 | — | — | — | — |
| finance_ff49 | B | sliced_w1 | — | — | — | — |
| finance_ff49 | C | sliced_w1 | — | — | — | — |
| finance_ff49 | tflow | sliced_w1 | — | — | — | — |
| gaussian_aniso_d16_cor | A | sliced_w1 | — | — | — | — |
| gaussian_aniso_d16_cor | B | sliced_w1 | — | — | — | — |
| gaussian_aniso_d16_cor | C | sliced_w1 | — | — | — | — |
| gaussian_aniso_d16_cor | tflow | sliced_w1 | — | — | — | — |
| imagenette_dcae | A | fid | — | — | — | — |
| imagenette_dcae | B | fid | — | — | — | — |
| imagenette_dcae | C | fid | — | — | — | — |
| imagenette_dcae | tflow | fid | — | — | — | — |
| piv_d16 | A | sliced_w1 | — | — | — | — |
| piv_d16 | B | sliced_w1 | — | — | — | — |
| piv_d16 | C | sliced_w1 | — | — | — | — |
| piv_d16 | tflow | sliced_w1 | — | — | — | — |
| piv_d256 | A | sliced_w1 | — | — | — | — |
| piv_d256 | B | sliced_w1 | — | — | — | — |
| piv_d256 | C | sliced_w1 | — | — | — | — |
| piv_d256 | tflow | sliced_w1 | — | — | — | — |
| piv_d32 | A | sliced_w1 | — | — | — | — |
| piv_d32 | B | sliced_w1 | — | — | — | — |
| piv_d32 | C | sliced_w1 | — | — | — | — |
| piv_d32 | tflow | sliced_w1 | — | — | — | — |
| piv_d64 | A | sliced_w1 | — | — | — | — |
| piv_d64 | B | sliced_w1 | — | — | — | — |
| piv_d64 | C | sliced_w1 | — | — | — | — |
| piv_d64 | tflow | sliced_w1 | — | — | — | — |
| student_t_d128_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d128_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d128_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d128_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df1.5_cor | A | sliced_w1 | — | — | — | — |
| student_t_d16_df1.5_cor | B | sliced_w1 | — | — | — | — |
| student_t_d16_df1.5_cor | C | sliced_w1 | — | — | — | — |
| student_t_d16_df1.5_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df10.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d16_df10.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d16_df10.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d16_df10.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df2.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d16_df2.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d16_df2.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d16_df2.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d16_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d16_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d16_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df5.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d16_df5.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d16_df5.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d16_df5.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df50.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d16_df50.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d16_df50.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d16_df50.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d256_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d256_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d256_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d256_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d2_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d2_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d2_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d2_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d32_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d32_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d32_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d32_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d64_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d64_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d64_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d64_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d8_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d8_df3.0_cor | B | sliced_w1 | — | — | — | — |
| student_t_d8_df3.0_cor | C | sliced_w1 | — | — | — | — |
| student_t_d8_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| toy_radial_angular | A | sliced_w1 | — | — | — | — |
| toy_radial_angular | B | sliced_w1 | — | — | — | — |
| toy_radial_angular | C | sliced_w1 | — | — | — | — |
| toy_radial_angular | tflow | sliced_w1 | — | — | — | — |
| weather_au_wind | A | sliced_w1 | — | — | — | — |
| weather_au_wind | B | sliced_w1 | — | — | — | — |
| weather_au_wind | C | sliced_w1 | — | — | — | — |
| weather_au_wind | tflow | sliced_w1 | — | — | — | — |

## Fixed-spherical + gain audio reference

Compatibility: **verified_for_prepared_protocol**. Measured fixed-spherical accuracy is about 0.8067, not an exact reproduction of the reported 0.810 ± 0.013. The original baseline mismatch is preserved.

Measured reference accuracy 0.8066667 ± 0.0073522 (population SD), energy KS 0.0218333; 0 changed digit predictions across 6,000 paired outputs. New-run execution compatibility remains separately listed in report.json.

| Audio method | Role / status | Digit accuracy | Energy KS | Coverage > q95 | Coverage > q99 |
|---|---|---|---|---|---|
| A | new matched study arm / incomplete | — | — | — | — |
| B | new matched study arm / incomplete | — | — | — | — |
| C | new matched study arm / incomplete | — | — | — | — |
| tflow | new matched study arm / incomplete | — | — | — | — |
| fixed_spherical_empirical_gain_reference | completed checkpoint-based reference, not new A or historical paper value / verified_checkpoint_reference | 0.8066667 ± 0.0073522 | 0.0218333 ± 0.0000000 | 0.0500000 ± 0.0000000 | 0.0145000 ± 0.0000000 |

## Findings status

1. Whether normalized input improves original RAFM-Ang: use the paired B−A deltas for complete conditions; no full-suite conclusion before completion.
2. Whether radius conditioning helps: use paired B−C deltas, retaining identical parameter counts. Audio gains were constructed independently of content; a benefit is not assumed.
3. Consistency: each paired metric records all three deltas and the number favoring B. No significance claim follows from three seeds alone.
4. Cost: parameter, conditioning overhead, measured training/sampling time and recorded memory are reported per seed. Missing memory fields remain absent; historical timing is not treated as matched.

See report.json for all errors, source/config fingerprints, dataset identities, hardware, tuning/sanity records, radius drift, and paired deltas.
