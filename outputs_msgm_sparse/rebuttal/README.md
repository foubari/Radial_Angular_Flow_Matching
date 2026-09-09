# Finance and Weather: MSGM sparse results

Both datasets and all three seeds passed final artifact validation. No failed seed was filtered, retried, resampled, or repaired.

Scientific source: `104f9118b690bb17f6ef7cfb871364a1ed6b8c56`. Seeds: 8925, 77395, 65457.

Each independent job uses one GPU; the three seeds run sequentially within its unchanged dataset CLI. The table gives the actual chronological split sizes and generated-sample counts.

| Dataset | Dimension | Train / validation / test | Generated per seed | GPU / PyTorch |
|---|---:|---|---:|---|
| finance_ff49 | 49 | 8609 / 2869 / 2871 | 2871 | AMD Instinct MI210 / 2.7.1+rocm6.3 |
| weather_au_wind | 96 | 3508 / 1169 / 1170 | 1170 | AMD Instinct MI210 / 2.7.1+rocm6.3 |

The unchanged effective configuration is 10,000 training steps, batch 4,096, Adam learning rate 0.001, hidden width 128, checkpoint interval 1,000, and 128 RK4 sampling steps (512 network evaluations). The split is chronological, with split seed 0 recorded by preflight. Sparse logging stays at every 200 steps; the dense adapter logs every 500 steps, so logging overhead is not identical.

[summary.json](summary.json) reports every original metric: 14 distribution/radial/angular values, three stability values, and `sample_time_s` / `total_train_time_s`. Means and **population standard deviations (`ddof=0`)** use all three original numeric values without rounding or filtering. This matches the repository's aggregate convention; no inference or metric evaluation was rerun.

[results_manifest.json](results_manifest.json) records exact commands, input hashes, GPU/runtime metadata, completion-report hashes, and per-seed JSON/sample/training-log hashes. The [raw JSON archive](msgm_sparse_real_results.zip) preserves all six original metric files byte-for-byte. The original per-seed artifacts remain in their existing directories.

| Dataset | Seed | Original metrics | Training log |
|---|---:|---|---|
| finance_ff49 | 8925 | [JSON](E_finance/finance_ff49/msgm_sparse/seed_8925/metrics.json) | [CSV](E_finance/finance_ff49/msgm_sparse/seed_8925/train_log.csv) |
| finance_ff49 | 77395 | [JSON](E_finance/finance_ff49/msgm_sparse/seed_77395/metrics.json) | [CSV](E_finance/finance_ff49/msgm_sparse/seed_77395/train_log.csv) |
| finance_ff49 | 65457 | [JSON](E_finance/finance_ff49/msgm_sparse/seed_65457/metrics.json) | [CSV](E_finance/finance_ff49/msgm_sparse/seed_65457/train_log.csv) |
| weather_au_wind | 8925 | [JSON](E_weather/weather_au_wind/msgm_sparse/seed_8925/metrics.json) | [CSV](E_weather/weather_au_wind/msgm_sparse/seed_8925/train_log.csv) |
| weather_au_wind | 77395 | [JSON](E_weather/weather_au_wind/msgm_sparse/seed_77395/metrics.json) | [CSV](E_weather/weather_au_wind/msgm_sparse/seed_77395/train_log.csv) |
| weather_au_wind | 65457 | [JSON](E_weather/weather_au_wind/msgm_sparse/seed_65457/metrics.json) | [CSV](E_weather/weather_au_wind/msgm_sparse/seed_65457/train_log.csv) |

Validation requires the pinned scientific source, unchanged committed input bytes, passed preflight, both successful worker reports, all 19 finite metrics, finite real samples of exactly the prescribed shape, and complete finite training logs ending at step 10,000. Logged losses do not cover every unlogged minibatch or gradient, and finite artifacts do not establish good generation quality. A failed or missing seed prevents publication of this complete summary.

The sparse and historical dense adapters use different SDE implementations. Matched CLI configuration does not establish bit-identical dense/sparse stacks. All scientific settings and saved inputs were preserved.
