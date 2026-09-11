# RAFM-Ang input parameterization and t-Flow study

Status: **incomplete**. 310/336 final seeds verified; 101/112 three-seed method groups complete.

Three-seed means use population standard deviation. Missing, failed and incompatible groups have no mean. Training directories do not establish that a scheduler job is currently running.

| Condition | Comparison | A | B | C | t-Flow | Blockers |
|---|---|---|---|---|---|---|
| aniso_k1 | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| aniso_k10 | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 2/3 incomplete | — |
| aniso_k100 | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| aniso_k3 | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| aniso_k30 | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| aniso_k300 | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| audiomnist_stft | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| finance_ff49 | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| gaussian_aniso_d16_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| imagenette_dcae | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| piv_d16 | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| piv_d256 | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 0/3 incomplete | — |
| piv_d32 | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| piv_d64 | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 0/3 incomplete | — |
| student_t_d128_df3.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 0/3 incomplete | — |
| student_t_d16_df1.5_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d16_df10.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d16_df2.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d16_df3.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d16_df5.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d16_df50.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d256_df3.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 0/3 incomplete | — |
| student_t_d2_df3.0_cor | new_matched_realization | 2/3 incomplete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d32_df3.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| student_t_d64_df3.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 0/3 incomplete | — |
| student_t_d8_df3.0_cor | new_matched_realization | 3/3 complete | 3/3 complete | 3/3 complete | 3/3 complete | — |
| toy_radial_angular | new_matched_realization | 1/3 incomplete | 1/3 incomplete | 1/3 incomplete | 3/3 complete | — |
| weather_au_wind | new_matched_runs_on_verified_existing_cache | 3/3 complete | 3/3 complete | 3/3 complete | 0/3 incomplete | — |

## Failures and compatibility issues

- aniso_k10 / tflow / seed 77395: failed; ["angular_sw_bin3"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/aniso_k10/seed_77395/result.json
- piv_d256 / tflow / seed 8925: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d256/seed_8925/result.json
- piv_d256 / tflow / seed 77395: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d256/seed_77395/result.json
- piv_d256 / tflow / seed 65457: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d256/seed_65457/result.json
- piv_d64 / tflow / seed 8925: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d64/seed_8925/result.json
- piv_d64 / tflow / seed 77395: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d64/seed_77395/result.json
- piv_d64 / tflow / seed 65457: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/piv_d64/seed_65457/result.json
- student_t_d128_df3.0_cor / tflow / seed 8925: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d128_df3.0_cor/seed_8925/result.json
- student_t_d128_df3.0_cor / tflow / seed 77395: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d128_df3.0_cor/seed_77395/result.json
- student_t_d128_df3.0_cor / tflow / seed 65457: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d128_df3.0_cor/seed_65457/result.json
- student_t_d256_df3.0_cor / tflow / seed 8925: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d256_df3.0_cor/seed_8925/result.json
- student_t_d256_df3.0_cor / tflow / seed 77395: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d256_df3.0_cor/seed_77395/result.json
- student_t_d256_df3.0_cor / tflow / seed 65457: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d256_df3.0_cor/seed_65457/result.json
- student_t_d2_df3.0_cor / A / seed 65457: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/student_t_d2_df3.0_cor/A/seed_65457/result.json
- student_t_d64_df3.0_cor / tflow / seed 8925: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d64_df3.0_cor/seed_8925/result.json
- student_t_d64_df3.0_cor / tflow / seed 77395: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d64_df3.0_cor/seed_77395/result.json
- student_t_d64_df3.0_cor / tflow / seed 65457: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/student_t_d64_df3.0_cor/seed_65457/result.json
- toy_radial_angular / A / seed 77395: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/A/seed_77395/result.json
- toy_radial_angular / A / seed 65457: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/A/seed_65457/result.json
- toy_radial_angular / B / seed 8925: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/B/seed_8925/result.json
- toy_radial_angular / B / seed 65457: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/B/seed_65457/result.json
- toy_radial_angular / C / seed 8925: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/C/seed_8925/result.json
- toy_radial_angular / C / seed 77395: failed; {"type": "FloatingPointError", "message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n"}; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_rafm_input_study/v1/final/toy_radial_angular/C/seed_77395/result.json
- weather_au_wind / tflow / seed 8925: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/weather_au_wind/seed_8925/result.json
- weather_au_wind / tflow / seed 77395: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/weather_au_wind/seed_77395/result.json
- weather_au_wind / tflow / seed 65457: failed; ["angular_sw_bin0", "angular_sw_bin1", "angular_sw_bin2", "angular_sw_bin3", "angular_sw_mean"]; /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v2/final/weather_au_wind/seed_65457/result.json

## Preserved implementation and startup failures

- Implementation-check job 765729: 46 passed / 4 failed; Sampler called CUDA peak-memory reset for CPU unit fixtures. Resolution: Guard CUDA peak reset by device.type; CPU peak_memory stays None. Benchmark results affected: False.
- Implementation-check job 765756: 0 passed / 1 failed; Saved-output classifier check aborted before predictions: MIOpen SQLite kernel cache was read-only. Resolution: Explicit writable per-job MIOpen cache paths in scheduler wrapper; evaluator and settings unchanged. Benchmark results affected: False.
- Earlier t-Flow attempt /mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_tflow_full/v1: cancelled_after_startup_failure; 5 startup candidate failures, 0 observed optimizer updates, 0 checkpoints. These cancelled attempts are separate from current scientific seed outcomes. Resolution: Explicit GPU initialization in common orchestration entrypoint; retain runtime/config bytes and validate actual public trainer in fresh processes before restarting to v2.

Current check results and dated worker/scheduler snapshots are preserved in report.json; this collector makes no live scheduler queries.

## Measured comparison

| Condition | Method | Primary metric | Mean ± population SD | Angular SW | Train seconds | Sample seconds |
|---|---|---|---|---|---|---|
| aniso_k1 | A | sliced_w1 | 0.0211472 ± 0.000229 | 0.00815005 ± 6.18e-05 | 29.9516 ± 1.4 | 0.189125 ± 0.00188 |
| aniso_k1 | B | sliced_w1 | 0.0211655 ± 0.000372 | 0.00822313 ± 2.82e-05 | 29.199 ± 0.13 | 0.28631 ± 0.00548 |
| aniso_k1 | C | sliced_w1 | 0.0211521 ± 0.000376 | 0.00805455 ± 5e-05 | 29.8209 ± 0.733 | 0.252535 ± 0.00709 |
| aniso_k1 | tflow | sliced_w1 | 0.504725 ± 0.047 | 0.0881023 ± 0.00804 | 33.2556 ± 0.761 | 0.420272 ± 0.002 |
| aniso_k10 | A | sliced_w1 | 0.134339 ± 0.0134 | 0.00907498 ± 0.000525 | 28.4954 ± 1.85 | 0.192833 ± 0.00799 |
| aniso_k10 | B | sliced_w1 | 0.154051 ± 0.0335 | 0.00974563 ± 0.00104 | 29.6053 ± 4.02 | 0.282808 ± 0.000816 |
| aniso_k10 | C | sliced_w1 | 0.134463 ± 0.00985 | 0.00977954 ± 0.000297 | 28.9598 ± 0.25 | 0.253448 ± 0.00342 |
| aniso_k10 | tflow | sliced_w1 | — | — | — | — |
| aniso_k100 | A | sliced_w1 | 1.47619 ± 0.206 | 0.0102071 ± 0.0009 | 28.7543 ± 1.87 | 0.187184 ± 0.000213 |
| aniso_k100 | B | sliced_w1 | 1.20148 ± 0.148 | 0.00904 ± 0.000423 | 27.2852 ± 0.639 | 0.284107 ± 0.00202 |
| aniso_k100 | C | sliced_w1 | 1.25819 ± 0.0936 | 0.0105899 ± 9.27e-05 | 26.8849 ± 0.177 | 0.249511 ± 0.00318 |
| aniso_k100 | tflow | sliced_w1 | 13.6145 ± 0.467 | 0.0889079 ± 0.00333 | 32.3423 ± 1.5 | 0.418299 ± 0.00214 |
| aniso_k3 | A | sliced_w1 | 0.0469288 ± 0.00187 | 0.00822523 ± 0.000137 | 28.6249 ± 1.95 | 0.191957 ± 0.00625 |
| aniso_k3 | B | sliced_w1 | 0.0490563 ± 0.00372 | 0.00859133 ± 0.000104 | 29.5231 ± 1.07 | 0.290025 ± 0.00779 |
| aniso_k3 | C | sliced_w1 | 0.0477054 ± 0.00104 | 0.00834798 ± 0.000161 | 29.1922 ± 1.78 | 0.254722 ± 0.00798 |
| aniso_k3 | tflow | sliced_w1 | 0.516602 ± 0.152 | 0.0406922 ± 0.0107 | 33.7746 ± 0.987 | 0.420259 ± 0.000811 |
| aniso_k30 | A | sliced_w1 | 0.34036 ± 0.036 | 0.00827177 ± 0.000365 | 27.8225 ± 2.01 | 0.187382 ± 0.000226 |
| aniso_k30 | B | sliced_w1 | 0.300452 ± 0.012 | 0.00790487 ± 0.00012 | 29.1311 ± 0.551 | 0.29306 ± 0.00152 |
| aniso_k30 | C | sliced_w1 | 0.35292 ± 0.0311 | 0.00953367 ± 0.000266 | 29.3881 ± 2.06 | 0.255487 ± 0.00671 |
| aniso_k30 | tflow | sliced_w1 | 4.83029 ± 0.464 | 0.0865656 ± 0.0134 | 31.9576 ± 0.892 | 0.421519 ± 0.00305 |
| aniso_k300 | A | sliced_w1 | 6.04078 ± 0.522 | 0.0133089 ± 0.00171 | 27.991 ± 2.27 | 0.198728 ± 0.0077 |
| aniso_k300 | B | sliced_w1 | 2.5119 ± 0.343 | 0.00756519 ± 0.000244 | 28.229 ± 1.15 | 0.282241 ± 0.00053 |
| aniso_k300 | C | sliced_w1 | 3.74025 ± 0.406 | 0.0108924 ± 0.000286 | 29.4933 ± 1.43 | 0.251041 ± 0.00516 |
| aniso_k300 | tflow | sliced_w1 | 191.466 ± 20.3 | 0.0689496 ± 0.0053 | 32.3962 ± 1.36 | 0.419076 ± 0.00221 |
| audiomnist_stft | A | digit_acc | 0.788333 ± 0.0113 | 0.000732767 ± 2.34e-06 | 8903.6 ± 19.5 | 873.544 ± 3.28 |
| audiomnist_stft | B | digit_acc | 0.7765 ± 0.0135 | 0.000729383 ± 3.04e-06 | 8921.08 ± 13.5 | 876.744 ± 7.97 |
| audiomnist_stft | C | digit_acc | 0.786833 ± 0.00821 | 0.000730646 ± 2.74e-06 | 8906.96 ± 11.8 | 869.893 ± 2.9 |
| audiomnist_stft | tflow | digit_acc | 0.144833 ± 0.00691 | — | 8938.97 ± 5.78 | 878.412 ± 0.442 |
| finance_ff49 | A | sliced_w1 | 0.181095 ± 0.00316 | 0.0179667 ± 0.00074 | 30.5516 ± 0.274 | 0.148687 ± 0.006 |
| finance_ff49 | B | sliced_w1 | 0.181921 ± 0.00277 | 0.0180951 ± 0.000374 | 31.0265 ± 2.63 | 0.228026 ± 0.00749 |
| finance_ff49 | C | sliced_w1 | 0.192922 ± 0.00323 | 0.0209264 ± 0.000863 | 31.8723 ± 1.47 | 0.192296 ± 0.00489 |
| finance_ff49 | tflow | sliced_w1 | 5.95169 ± 0.306 | 0.0602774 ± 0.00408 | 33.7253 ± 0.862 | 0.38924 ± 0.00203 |
| gaussian_aniso_d16_cor | A | sliced_w1 | 0.122398 ± 0.014 | 0.0128373 ± 0.000442 | 28.0462 ± 1.1 | 0.171548 ± 0.000541 |
| gaussian_aniso_d16_cor | B | sliced_w1 | 0.139822 ± 0.0198 | 0.0139663 ± 0.00118 | 28.8855 ± 1.24 | 0.248022 ± 0.00383 |
| gaussian_aniso_d16_cor | C | sliced_w1 | 0.140565 ± 0.0201 | 0.0149026 ± 0.000773 | 29.5518 ± 1.5 | 0.225154 ± 0.00191 |
| gaussian_aniso_d16_cor | tflow | sliced_w1 | 1.18918 ± 0.283 | 0.0724261 ± 0.0107 | 33.428 ± 1.33 | 0.414809 ± 0.00146 |
| imagenette_dcae | A | fid | 147.957 ± 0.378 | 0.0020002 ± 1.73e-05 | 1775.4 ± 33.1 | 49.1546 ± 0.755 |
| imagenette_dcae | B | fid | 311.403 ± 11.1 | 0.00577508 ± 0.0023 | 1805.7 ± 41.3 | 49.0762 ± 0.254 |
| imagenette_dcae | C | fid | 227.437 ± 67.7 | 0.00254277 ± 0.000183 | 1785.28 ± 15.5 | 49.11 ± 0.0955 |
| imagenette_dcae | tflow | fid | 223.796 ± 6.27 | — | 1772.39 ± 1.17 | 50.1898 ± 0.121 |
| piv_d16 | A | sliced_w1 | 0.012025 ± 0.00127 | 0.0564074 ± 0.00325 | 30.6702 ± 5.14 | 0.177942 ± 0.00708 |
| piv_d16 | B | sliced_w1 | 0.0126237 ± 0.0011 | 0.0575728 ± 0.00344 | 30.9233 ± 2.11 | 0.254992 ± 0.00206 |
| piv_d16 | C | sliced_w1 | 0.0123003 ± 0.00104 | 0.0609467 ± 0.00265 | 31.0793 ± 1.48 | 0.229063 ± 0.00573 |
| piv_d16 | tflow | sliced_w1 | 0.0645829 ± 0.0214 | 0.144102 ± 0.0179 | 33.2168 ± 0.974 | 0.417756 ± 0.00215 |
| piv_d256 | A | sliced_w1 | 0.0232868 ± 0.00129 | 0.0121693 ± 0.00056 | 29.2802 ± 2.17 | 0.323549 ± 0.00284 |
| piv_d256 | B | sliced_w1 | 0.0243504 ± 0.00153 | 0.0123624 ± 0.0006 | 31.7043 ± 4.29 | 0.442795 ± 0.00615 |
| piv_d256 | C | sliced_w1 | 0.0248361 ± 0.0018 | 0.0127048 ± 0.000652 | 29.9871 ± 1.9 | 0.418333 ± 0.00645 |
| piv_d256 | tflow | sliced_w1 | — | — | — | — |
| piv_d32 | A | sliced_w1 | 0.0202315 ± 0.00157 | 0.0353442 ± 0.00132 | 30.3151 ± 0.525 | 0.18823 ± 0.00482 |
| piv_d32 | B | sliced_w1 | 0.0210898 ± 0.003 | 0.0360158 ± 0.00301 | 30.3454 ± 1.62 | 0.288978 ± 0.00365 |
| piv_d32 | C | sliced_w1 | 0.0206341 ± 0.00125 | 0.0378622 ± 0.00103 | 29.627 ± 1.39 | 0.263008 ± 0.00298 |
| piv_d32 | tflow | sliced_w1 | 0.893375 ± 0.617 | 0.0726497 ± 0.0129 | 33.6042 ± 0.647 | 0.420847 ± 0.00225 |
| piv_d64 | A | sliced_w1 | 0.028788 ± 0.00197 | 0.0232149 ± 0.00084 | 31.7201 ± 4.89 | 0.21524 ± 0.00499 |
| piv_d64 | B | sliced_w1 | 0.0290825 ± 0.00398 | 0.0235927 ± 0.00117 | 29.4505 ± 1.97 | 0.322287 ± 0.00292 |
| piv_d64 | C | sliced_w1 | 0.039156 ± 0.0018 | 0.0286122 ± 0.000858 | 30.6621 ± 0.563 | 0.295359 ± 0.00703 |
| piv_d64 | tflow | sliced_w1 | — | — | — | — |
| student_t_d128_df3.0_cor | A | sliced_w1 | 1.50997 ± 0.084 | 0.00801515 ± 0.000185 | 28.0057 ± 2.15 | 0.25957 ± 0.000994 |
| student_t_d128_df3.0_cor | B | sliced_w1 | 1.11209 ± 0.0896 | 0.00621377 ± 0.000295 | 29.5101 ± 1.73 | 0.412964 ± 0.00519 |
| student_t_d128_df3.0_cor | C | sliced_w1 | 1.10854 ± 0.0918 | 0.00615429 ± 0.000287 | 29.4188 ± 0.698 | 0.366984 ± 0.00739 |
| student_t_d128_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d16_df1.5_cor | A | sliced_w1 | 2.11712 ± 0.121 | 0.0184865 ± 0.00162 | 27.201 ± 1.42 | 0.173043 ± 0.00168 |
| student_t_d16_df1.5_cor | B | sliced_w1 | 1.86505 ± 0.0262 | 0.0154304 ± 0.000649 | 28.4937 ± 1.51 | 0.255521 ± 0.00166 |
| student_t_d16_df1.5_cor | C | sliced_w1 | 1.80441 ± 0.0396 | 0.0152144 ± 0.000614 | 29.6707 ± 0.691 | 0.232372 ± 0.0067 |
| student_t_d16_df1.5_cor | tflow | sliced_w1 | 271.166 ± 39.5 | 0.110802 ± 0.018 | 32.3505 ± 1.28 | 0.413989 ± 0.00438 |
| student_t_d16_df10.0_cor | A | sliced_w1 | 0.158327 ± 0.0152 | 0.0129074 ± 0.000366 | 27.9963 ± 1.4 | 0.182596 ± 0.00937 |
| student_t_d16_df10.0_cor | B | sliced_w1 | 0.185174 ± 0.00553 | 0.0146498 ± 0.000471 | 28.4805 ± 0.471 | 0.253895 ± 0.00094 |
| student_t_d16_df10.0_cor | C | sliced_w1 | 0.184642 ± 0.00476 | 0.0152324 ± 0.000492 | 29.8873 ± 1.02 | 0.226393 ± 0.00341 |
| student_t_d16_df10.0_cor | tflow | sliced_w1 | 2.71041 ± 0.739 | 0.111404 ± 0.0438 | 30.8126 ± 0.818 | 0.411937 ± 0.000706 |
| student_t_d16_df2.0_cor | A | sliced_w1 | 0.627396 ± 0.0182 | 0.0162338 ± 0.000255 | 27.6914 ± 0.909 | 0.177961 ± 0.00866 |
| student_t_d16_df2.0_cor | B | sliced_w1 | 0.575626 ± 0.0517 | 0.0150523 ± 0.000733 | 30.3725 ± 4.88 | 0.254751 ± 0.00159 |
| student_t_d16_df2.0_cor | C | sliced_w1 | 0.570111 ± 0.0527 | 0.0153242 ± 0.000788 | 29.0208 ± 1.63 | 0.227622 ± 0.00436 |
| student_t_d16_df2.0_cor | tflow | sliced_w1 | 89.9565 ± 2.73 | 0.0868404 ± 0.023 | 30.3986 ± 0.839 | 0.410965 ± 0.000273 |
| student_t_d16_df3.0_cor | A | sliced_w1 | 0.310411 ± 0.0313 | 0.0145394 ± 0.000864 | 28.781 ± 1.54 | 0.174132 ± 0.00215 |
| student_t_d16_df3.0_cor | B | sliced_w1 | 0.323474 ± 0.0355 | 0.0153067 ± 0.00117 | 30.2524 ± 1.69 | 0.265351 ± 0.00612 |
| student_t_d16_df3.0_cor | C | sliced_w1 | 0.319628 ± 0.0385 | 0.0152969 ± 0.00135 | 29.9041 ± 1.28 | 0.227943 ± 0.0055 |
| student_t_d16_df3.0_cor | tflow | sliced_w1 | 4.80983 ± 1.31 | 0.123259 ± 0.0505 | 29.8389 ± 0.324 | 0.413588 ± 0.00173 |
| student_t_d16_df5.0_cor | A | sliced_w1 | 0.20445 ± 0.02 | 0.0130805 ± 0.000653 | 28.0019 ± 1.51 | 0.172241 ± 0.00145 |
| student_t_d16_df5.0_cor | B | sliced_w1 | 0.213148 ± 0.0314 | 0.0140404 ± 0.00121 | 28.1324 ± 1.1 | 0.258299 ± 0.00427 |
| student_t_d16_df5.0_cor | C | sliced_w1 | 0.229948 ± 0.0144 | 0.0147437 ± 0.000486 | 28.6862 ± 0.63 | 0.228391 ± 0.0019 |
| student_t_d16_df5.0_cor | tflow | sliced_w1 | 3.73209 ± 1.79 | 0.118765 ± 0.0505 | 30.1966 ± 0.402 | 0.412208 ± 0.000655 |
| student_t_d16_df50.0_cor | A | sliced_w1 | 0.1811 ± 0.0455 | 0.0146002 ± 0.00241 | 27.1946 ± 0.87 | 0.172532 ± 0.00038 |
| student_t_d16_df50.0_cor | B | sliced_w1 | 0.200706 ± 0.0493 | 0.0161616 ± 0.00256 | 28.8886 ± 1.65 | 0.2548 ± 0.00092 |
| student_t_d16_df50.0_cor | C | sliced_w1 | 0.207581 ± 0.0458 | 0.0170444 ± 0.00251 | 30.2752 ± 2.95 | 0.223911 ± 0.00101 |
| student_t_d16_df50.0_cor | tflow | sliced_w1 | 3.1381 ± 1.39 | 0.105101 ± 0.0372 | 32.1953 ± 1.63 | 0.415962 ± 0.00512 |
| student_t_d256_df3.0_cor | A | sliced_w1 | 1.76484 ± 0.0796 | 0.0047651 ± 0.000178 | 27.3535 ± 0.891 | 0.326177 ± 0.00694 |
| student_t_d256_df3.0_cor | B | sliced_w1 | 1.21185 ± 0.051 | 0.00382652 ± 0.000204 | 29.6656 ± 1.2 | 0.447702 ± 0.000674 |
| student_t_d256_df3.0_cor | C | sliced_w1 | 1.20176 ± 0.0425 | 0.0037501 ± 0.000186 | 27.8212 ± 1.29 | 0.411618 ± 0.000261 |
| student_t_d256_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d2_df3.0_cor | A | sliced_w1 | — | — | — | — |
| student_t_d2_df3.0_cor | B | sliced_w1 | 0.435777 ± 0.0471 | 0.744695 ± 0.109 | 30.1976 ± 1.3 | 0.248265 ± 0.00792 |
| student_t_d2_df3.0_cor | C | sliced_w1 | 0.472682 ± 0.00828 | 0.856542 ± 0.0228 | 29.5117 ± 1.88 | 0.219524 ± 0.00695 |
| student_t_d2_df3.0_cor | tflow | sliced_w1 | 0.819154 ± 0.112 | 0.444696 ± 0.0519 | 32.8428 ± 1.3 | 0.443122 ± 0.00465 |
| student_t_d32_df3.0_cor | A | sliced_w1 | 0.468922 ± 0.0366 | 0.0114199 ± 0.000433 | 30.5915 ± 2.76 | 0.187334 ± 0.000721 |
| student_t_d32_df3.0_cor | B | sliced_w1 | 0.418465 ± 0.0194 | 0.0106911 ± 0.000325 | 28.2699 ± 1.48 | 0.276602 ± 0.00324 |
| student_t_d32_df3.0_cor | C | sliced_w1 | 0.41523 ± 0.0253 | 0.010687 ± 0.000381 | 30.6231 ± 0.383 | 0.248724 ± 0.000156 |
| student_t_d32_df3.0_cor | tflow | sliced_w1 | 7.95471 ± 1.64 | 0.126164 ± 0.0138 | 31.8306 ± 0.852 | 0.417842 ± 0.00137 |
| student_t_d64_df3.0_cor | A | sliced_w1 | 0.91761 ± 0.0666 | 0.00936634 ± 0.000597 | 29.4447 ± 2.68 | 0.214585 ± 0.000493 |
| student_t_d64_df3.0_cor | B | sliced_w1 | 0.733907 ± 0.116 | 0.00822849 ± 0.000719 | 29.8089 ± 1.53 | 0.336745 ± 0.00468 |
| student_t_d64_df3.0_cor | C | sliced_w1 | 0.728216 ± 0.114 | 0.00807921 ± 0.00071 | 30.2207 ± 4.89 | 0.296558 ± 0.00382 |
| student_t_d64_df3.0_cor | tflow | sliced_w1 | — | — | — | — |
| student_t_d8_df3.0_cor | A | sliced_w1 | 0.197703 ± 0.0269 | 0.0178546 ± 0.00152 | 28.7421 ± 1.19 | 0.170762 ± 0.000557 |
| student_t_d8_df3.0_cor | B | sliced_w1 | 0.182934 ± 0.013 | 0.0177453 ± 0.000377 | 29.2805 ± 2.01 | 0.247496 ± 0.00782 |
| student_t_d8_df3.0_cor | C | sliced_w1 | 0.242447 ± 0.0191 | 0.0211377 ± 0.000394 | 30.9332 ± 3.15 | 0.216585 ± 0.000221 |
| student_t_d8_df3.0_cor | tflow | sliced_w1 | 5.25255 ± 1.12 | 0.173225 ± 0.0112 | 32.1988 ± 0.67 | 0.436964 ± 0.00193 |
| toy_radial_angular | A | sliced_w1 | — | — | — | — |
| toy_radial_angular | B | sliced_w1 | — | — | — | — |
| toy_radial_angular | C | sliced_w1 | — | — | — | — |
| toy_radial_angular | tflow | sliced_w1 | 0.296139 ± 0.11 | 0.143094 ± 0.0478 | 32.7018 ± 0.849 | 0.437334 ± 0.00161 |
| weather_au_wind | A | sliced_w1 | 0.0766653 ± 0.000749 | 0.0144459 ± 0.000456 | 28.7054 ± 0.489 | 0.14456 ± 0.00103 |
| weather_au_wind | B | sliced_w1 | 0.0819225 ± 0.00282 | 0.0146738 ± 0.00021 | 29.2762 ± 2 | 0.215265 ± 0.00449 |
| weather_au_wind | C | sliced_w1 | 0.0813396 ± 0.00158 | 0.017798 ± 0.000616 | 30.1516 ± 2.63 | 0.183938 ± 0.00287 |
| weather_au_wind | tflow | sliced_w1 | — | — | — | — |

## Fixed-spherical + gain audio reference

Compatibility: **verified_for_prepared_protocol**. Measured fixed-spherical accuracy is about 0.8067, not an exact reproduction of the reported 0.810 ± 0.013. The original baseline mismatch is preserved.

Measured reference accuracy 0.8066667 ± 0.0073522 (population SD), energy KS 0.0218333; 0 changed digit predictions across 6,000 paired outputs. New-run execution compatibility remains separately listed in report.json.

Raw new-evaluator audit status: **compatibility_discrepancy**. Current measured reference: **measured_under_current_backend_with_recorded_differences**.

These are newly measured metrics for the same saved Y/X under the current study backend. Archived metrics and raw audit status remain unchanged. Digit predictions, accuracy, energy KS and coverage rates are identical; PIT, radial W1, logits and near-fixed-radius energy-bin assignments can differ. Only these current-backend values enter new comparisons; no historical training-time comparison.

Measured current-minus-archived differences (each seed/version):
- Seed 8925 / baseline: {"radial_w1": -2.8125445039606234e-07}; energy-bin accuracies identical: False.
- Seed 8925 / posthoc: {"PIT": -5.00000000069889e-07, "radial_w1": 1.3611813386299465e-06}; energy-bin accuracies identical: True.
- Seed 1234 / baseline: {"radial_w1": -1.9796689332274298e-07}; energy-bin accuracies identical: False.
- Seed 1234 / posthoc: {"PIT": -5.000000000143778e-07, "radial_w1": 1.3399173816042165e-06}; energy-bin accuracies identical: True.
- Seed 7 / baseline: {"radial_w1": -3.0561288189012714e-07}; energy-bin accuracies identical: False.
- Seed 7 / posthoc: {"PIT": -5.00000000069889e-07, "radial_w1": 1.4123320579567666e-06}; energy-bin accuracies identical: True.

| Audio method | Role / status | Digit accuracy | Energy KS | Coverage > q95 | Coverage > q99 |
|---|---|---|---|---|---|
| A | new matched study arm / complete | 0.7883333 ± 0.0113309 | 0.0218333 ± 0.0000000 | 0.0500000 ± 0.0000000 | 0.0145000 ± 0.0000000 |
| B | new matched study arm / complete | 0.7765000 ± 0.0134969 | 0.0218333 ± 0.0000000 | 0.0500000 ± 0.0000000 | 0.0145000 ± 0.0000000 |
| C | new matched study arm / complete | 0.7868333 ± 0.0082091 | 0.0218333 ± 0.0000000 | 0.0500000 ± 0.0000000 | 0.0145000 ± 0.0000000 |
| tflow | new matched study arm / complete | 0.1448333 ± 0.0069081 | 0.8322778 ± 0.0095939 | 0.8755000 ± 0.0109848 | 0.7640000 ± 0.0178466 |
| fixed_spherical_empirical_gain_reference | checkpoint reference re-evaluated under current backend, not new A or historical paper value / measured_under_current_backend_with_recorded_differences | 0.8066667 ± 0.0073522 | 0.0218333 ± 0.0000000 | 0.0500000 ± 0.0000000 | 0.0145000 ± 0.0000000 |

## Findings status

1. Whether normalized input improves original RAFM-Ang: use the paired B−A deltas for complete conditions; no full-suite conclusion before completion.
2. Whether radius conditioning helps: use paired B−C deltas, retaining identical parameter counts. Audio gains were constructed independently of content; a benefit is not assumed.
3. Consistency: each paired metric records all three deltas and the number favoring B. No significance claim follows from three seeds alone.
4. Cost: parameter, conditioning overhead, measured training/sampling time and recorded memory are reported per seed. Missing memory fields remain absent; historical timing is not treated as matched.

See report.json for all errors, source/config fingerprints, dataset identities, hardware, tuning/sanity records, radius drift, and paired deltas.
