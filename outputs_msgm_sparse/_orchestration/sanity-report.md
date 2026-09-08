# Toy sanity result

Completed 2026-09-08T15:43:50.901669+00:00 in Slurm job 758516 on one MI210 on `auh7-4b-gpu-180`.

All three seeds completed exactly 10,000 training steps. The independent supervisor validated all 19 required metrics as finite, all generated samples with shape `(10000, 2)` as finite, and all logged losses as finite. Every seed has zero NaN, exploding-norm and invalid-sample rates.

| Seed | Radial W1 | KS | Sliced W1 | MMD | Angular SW mean | Training (s) | Sampling (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8925 | 0.04503344744 | 0.01730000973 | 0.04551431909 | 0.0003046989441 | 0.03549454268 | 720.2569711 | 0.2499141693 |
| 77395 | 0.05700438842 | 0.01729995012 | 0.04562595114 | 0.00031042099 | 0.03789264336 | 716.9019752 | 0.2416563034 |
| 65457 | 0.03733160347 | 0.009899973869 | 0.05077065155 | 0.0003747940063 | 0.03954268713 | 717.4566584 | 0.239584446 |

Original per-seed JSONs contain all 19 metrics:

- [Seed 8925: metrics.json](../exp1_main_benchmark/toy_radial_angular/msgm_sparse/seed_8925/metrics.json)
- [Seed 77395: metrics.json](../exp1_main_benchmark/toy_radial_angular/msgm_sparse/seed_77395/metrics.json)
- [Seed 65457: metrics.json](../exp1_main_benchmark/toy_radial_angular/msgm_sparse/seed_65457/metrics.json)

[Independent validation report](toy2d-validation.json).

Protocol remains the unchanged exp1 command and configuration at source commit `eb80c6b5af1f3f47d35bf0f4d9ac0c83be9bce5a`: seeds 8925/77395/65457; 10,000 steps; batch 256; Adam learning rate 0.001; hidden width 128 and three hidden layers; 10,000 generated samples; RK4 nfe 128 (512 network evaluations, as in tabular FM). No scientific source or configuration was modified.

Unchanged exp1 generates synthetic data before model seeding; independent invocations may use different draws. The sparse SDE implementation and historical dense implementation come from different codebases; this run establishes the configured protocol and finite outputs, not byte identity of the historical dense implementation.

The toy seeds count toward the full 21-result sweep. The remaining six configurations will each use one GPU and run their three seeds sequentially. Their exact training budgets remain unchanged.
