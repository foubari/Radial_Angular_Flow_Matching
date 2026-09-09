# Finance and Weather result tables

All six prescribed runs completed and passed final validation. Mean ± population standard deviation (ddof=0), over seeds 8925, 77395, 65457. This table formats the existing values in [summary.json](summary.json); it does not recompute metrics.

| Metric | Finance FF49 | Weather AU wind |
|---|---:|---:|
| `radial_w1` | 1.48314 ± 0.04458034 | 0.1178591 ± 0.009620961 |
| `ks_stat` | 0.2052711 ± 0.005809827 | 0.03846155 ± 0.001395745 |
| `q950_err` | 0.2073839 ± 0.02258102 | 0.02717519 ± 0.009057577 |
| `q990_err` | 0.2593744 ± 0.01577394 | 0.02041065 ± 0.007845343 |
| `q995_err` | 0.3596646 ± 0.001277434 | 0.02308946 ± 0.01894671 |
| `tail_exc_95` | 0.02840474 ± 0.002327869 | 0.01324786 ± 0.003626188 |
| `tail_exc_99` | 0.006632997 ± 0.001149367 | 0.005384616 ± 0.001846365 |
| `sliced_w1` | 0.1816247 ± 0.002095871 | 0.1263942 ± 0.004208354 |
| `mmd` | 0.01181487 ± 0.0008560923 | 0.005310734 ± 0.0001734808 |
| `angular_sw_bin0` | 0.01350901 ± 0.0007691924 | 0.0185585 ± 0.0006703857 |
| `angular_sw_bin1` | 0.01593227 ± 0.0004271658 | 0.01644415 ± 0.0007420406 |
| `angular_sw_bin2` | 0.01710886 ± 0.0004574414 | 0.01817539 ± 3.217902e-05 |
| `angular_sw_bin3` | 0.0255465 ± 0.0007564554 | 0.02617921 ± 0.001815201 |
| `angular_sw_mean` | 0.01802416 ± 0.0005177068 | 0.01983931 ± 0.0003971039 |
| `nan_rate` | 0 ± 0 | 0 ± 0 |
| `exploding_norm_rate` | 0 ± 0 | 0 ± 0 |
| `invalid_rate` | 0 ± 0 | 0 ± 0 |
| `sample_time_s` | 0.2396732 ± 0.0004797685 | 0.2452176 ± 0.006075676 |
| `total_train_time_s` | 3037.009 ± 26.28094 | 3317.704 ± 20.57517 |

Timings are seconds per seed; the reported sampling time excludes latent initialization, following the unchanged adapter. The complete train/eval jobs took 2:36:46 for Finance and 2:52:29 for Weather, running concurrently.

| Dataset | Dimension | Chronological train / validation / test | Generated per seed | GPU |
|---|---:|---|---:|---|
| finance_ff49 | 49 | 8609 / 2869 / 2871 | 2871 | AMD Instinct MI210, device 0 on auh7-4b-gpu-055 |
| weather_au_wind | 96 | 3508 / 1169 / 1170 | 1170 | AMD Instinct MI210, device 1 on auh7-4b-gpu-055 |

Exact protocol: chronological 60/20/20 split (split seed 0 recorded), batch 4096, 10,000 steps, Adam LR 0.001, hidden width 128, checkpoint interval 1000, seeds 8925/77395/65457, n_gen=min(10000,n_test). Sampling uses the unchanged nfe=128 RK4-step setting (512 network evaluations). Each job used one MI210 with 68,702,699,520 bytes VRAM and PyTorch 2.7.1+rocm6.3.

Sparse and dense drivers produced identical configs in preflight, and committed input byte hashes matched. Existing sparse logging is every 200 steps versus dense every 500, so logging overhead differs. The sparse and dense SDE implementations come from different codebases. No settings were changed and no run was retried.

[All six original JSONs and training logs](README.md) · [Raw JSON ZIP](msgm_sparse_real_results.zip) · [Full provenance and hashes](results_manifest.json) · [Validation records](validation/)

Source commit: `104f9118b690bb17f6ef7cfb871364a1ed6b8c56`.
