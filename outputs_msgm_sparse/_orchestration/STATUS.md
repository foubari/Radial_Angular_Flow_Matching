# MSGM sparse tabular experiment

**Finished with one rejected result: 20 valid seeds and one nonfinite seed out of 21.** All seven configuration jobs have terminated. There are no remaining tabular training jobs.

Student-t D32 seed 65457 completed 10,000 training steps, then produced NaN values in six result metrics and a recorded NaN/invalid rate of approximately one in 10,000 generated samples. The supervisor stopped only that configuration. No retry, seed replacement, resampling, parameter change or scientific source change was made. The other six configurations completed all three seeds.

[Run outcome and all 21 per-seed JSON links](run-outcome.md). [Failure report](studentt-d32-failure.md). [Final Slurm accounting](final-job-accounting.txt).

The [raw results ZIP](../msgm_sparse_tabular_raw_results_with_failure.zip) preserves all 21 original metrics JSON files byte-for-byte, including the rejected JSON with NaN tokens. Its [raw manifest](../raw_results_manifest.json) explicitly has status `incomplete_nonfinite`, 20 valid and one rejected result. A CPU-only Slurm step verified byte identity. This is a raw transfer bundle, not a complete finite-results aggregate.

The strict CPU collector job 758576 was automatically cancelled after Student-t D32 job 758572 failed. Its complete-result manifest, summary means and archive remain absent.

| Config | Job | Final state |
|---|---:|---|
| toy2d | 758516 | COMPLETED |
| gaussian_d16 | 758570 | COMPLETED |
| studentt_d16 | 758571 | COMPLETED |
| studentt_d32 | 758572 | FAILED: seed 65457 nonfinite |
| piv_d16 | 758573 | COMPLETED |
| piv_d64 | 758574 | COMPLETED |
| piv_d256 | 758575 | COMPLETED |
| Strict CPU collection | 758576 | CANCELLED: failed dependency |

Each training configuration used one MI210 with three sequential seeds; the six remaining configurations ran on `auh7-4b-gpu-055` after toy sanity was reported. The original DC-AE/AudioMNIST jobs and CPU observer were not cancelled or modified. Raw packaging used one short CPU-only step in the existing observer allocation, with all writes under this tabular output tree.

Source remains pinned to `eb80c6b5af1f3f47d35bf0f4d9ac0c83be9bce5a`. The unchanged exp1 protocol uses seeds 8925/77395/65457, 10,000 steps, batch 256, Adam learning rate 0.001, hidden width 128 and three hidden layers, 10,000 generated samples, RK4 nfe 128 (512 network evaluations, as in tabular FM), and split seed 0 with 60/20/20 splits. PIV uses the committed tensors read-only.

Unchanged exp1 generates synthetic data before model seeding; independent invocations may use different draws. Sparse SDE and historical dense implementations come from different codebases. See [source audit](source-audit.md) and [toy sanity report](sanity-report.md).
