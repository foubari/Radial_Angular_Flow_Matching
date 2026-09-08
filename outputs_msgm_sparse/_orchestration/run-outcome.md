# Tabular run outcome

All seven configuration jobs have terminated. **20 of 21 seeds passed validation; Student-t D32 seed 65457 was rejected for nonfinite generated-result metrics.** The failed configuration was stopped without a retry or settings change. The other configurations completed.

The raw transfer bundle preserves all 21 original `metrics.json` files byte-for-byte, including the rejected file with NaN tokens. This is an incomplete result set, not a complete finite-results aggregate.

- [Raw results ZIP, including failure](../msgm_sparse_tabular_raw_results_with_failure.zip)
- [Raw manifest: incomplete_nonfinite](../raw_results_manifest.json)
- [Failure details](studentt-d32-failure.md)
- [CPU packaging and byte-identity validation log](raw-packaging.log)
- [Final Slurm accounting](final-job-accounting.txt)

Packaging ran as one CPU-only step in existing Slurm allocation 758467 with no GPU. It verified all 21 ZIP members against the original bytes and metadata; it performed no scientific recomputation. For the 20 valid seeds, generated tensor finiteness and exact 10,000-step completion were verified by their worker supervisors. The strict all-results collector 758576 was cancelled after its failed dependency, and its complete-result manifest, summary and archive remain absent.

| Configuration | Seed | Result | Original metrics JSON |
|---|---:|---|---|
| toy2d | 8925 | valid | [metrics.json](../exp1_main_benchmark/toy_radial_angular/msgm_sparse/seed_8925/metrics.json) |
| toy2d | 77395 | valid | [metrics.json](../exp1_main_benchmark/toy_radial_angular/msgm_sparse/seed_77395/metrics.json) |
| toy2d | 65457 | valid | [metrics.json](../exp1_main_benchmark/toy_radial_angular/msgm_sparse/seed_65457/metrics.json) |
| gaussian_d16 | 8925 | valid | [metrics.json](../exp1_main_benchmark/gaussian_aniso_d16_cor/msgm_sparse/seed_8925/metrics.json) |
| gaussian_d16 | 77395 | valid | [metrics.json](../exp1_main_benchmark/gaussian_aniso_d16_cor/msgm_sparse/seed_77395/metrics.json) |
| gaussian_d16 | 65457 | valid | [metrics.json](../exp1_main_benchmark/gaussian_aniso_d16_cor/msgm_sparse/seed_65457/metrics.json) |
| studentt_d16 | 8925 | valid | [metrics.json](../exp1_main_benchmark/student_t_d16_df3.0_cor/msgm_sparse/seed_8925/metrics.json) |
| studentt_d16 | 77395 | valid | [metrics.json](../exp1_main_benchmark/student_t_d16_df3.0_cor/msgm_sparse/seed_77395/metrics.json) |
| studentt_d16 | 65457 | valid | [metrics.json](../exp1_main_benchmark/student_t_d16_df3.0_cor/msgm_sparse/seed_65457/metrics.json) |
| studentt_d32 | 8925 | valid | [metrics.json](../exp1_main_benchmark/student_t_d32_df3.0_cor/msgm_sparse/seed_8925/metrics.json) |
| studentt_d32 | 77395 | valid | [metrics.json](../exp1_main_benchmark/student_t_d32_df3.0_cor/msgm_sparse/seed_77395/metrics.json) |
| studentt_d32 | 65457 | rejected_nonfinite | [metrics.json](../exp1_main_benchmark/student_t_d32_df3.0_cor/msgm_sparse/seed_65457/metrics.json) |
| piv_d16 | 8925 | valid | [metrics.json](../exp1_main_benchmark/piv_d16/msgm_sparse/seed_8925/metrics.json) |
| piv_d16 | 77395 | valid | [metrics.json](../exp1_main_benchmark/piv_d16/msgm_sparse/seed_77395/metrics.json) |
| piv_d16 | 65457 | valid | [metrics.json](../exp1_main_benchmark/piv_d16/msgm_sparse/seed_65457/metrics.json) |
| piv_d64 | 8925 | valid | [metrics.json](../exp1_main_benchmark/piv_d64/msgm_sparse/seed_8925/metrics.json) |
| piv_d64 | 77395 | valid | [metrics.json](../exp1_main_benchmark/piv_d64/msgm_sparse/seed_77395/metrics.json) |
| piv_d64 | 65457 | valid | [metrics.json](../exp1_main_benchmark/piv_d64/msgm_sparse/seed_65457/metrics.json) |
| piv_d256 | 8925 | valid | [metrics.json](../exp1_main_benchmark/piv_d256/msgm_sparse/seed_8925/metrics.json) |
| piv_d256 | 77395 | valid | [metrics.json](../exp1_main_benchmark/piv_d256/msgm_sparse/seed_77395/metrics.json) |
| piv_d256 | 65457 | valid | [metrics.json](../exp1_main_benchmark/piv_d256/msgm_sparse/seed_65457/metrics.json) |

Archive SHA-256: `73a562b074ab5de1b6a24a198789bf799a40b3909fb346787395fe048d261b20`.

Source commit: `eb80c6b5af1f3f47d35bf0f4d9ac0c83be9bce5a`. Scientific source, configs, seeds, data protocol, training budgets and sampling settings were preserved.

Unchanged exp1 generates synthetic data before model seeding; independent invocations may use different draws. Sparse SDE and historical dense implementations come from different codebases.
