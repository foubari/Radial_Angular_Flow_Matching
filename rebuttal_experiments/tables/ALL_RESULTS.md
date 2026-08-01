# ALL_RESULTS — every aggregated results table in one file

_Consolidation of all `tables/*.md`. Raw per-run metrics live under `raw_results/.../metrics.json`; per-run NFE/solver sweeps in `raw_results/E4_nfe_solver/*.csv`; directional raw in `raw_results/E3_angular/*.csv`. See `FINAL_REPORT.md` for interpretation._

## Contents

- 0. MASTER — all benchmarks × all methods (radial/sliced/ks/train)  (`tables/MASTER_results.md`)
- 1. E1 reproduction vs paper  (`tables/E1_repro_vs_paper.md`)
- 2a. E2 attribution — Student-t d16  (`tables/E2_attribution_student_t_d16.md`)
- 2b. E2 attribution — Student-t d32  (`tables/E2_attribution_student_t_d32.md`)
- 2c. E2 attribution — Gaussian d16  (`tables/E2_attribution_gaussian_d16.md`)
- 3. E3 directional — common-radial cr_sliced_w1 (mean±std)  (`tables/E3_cr_sliced_w1.md`)
- 4. E4 — NFE×solver + radius drift (synthetic d16)  (`tables/E4_summary.md`)
- 6a. E6 dimension scaling — radial_w1 by dim (mean±std)  (`tables/E6_radialW1_by_dim.md`)
- 6b. E6 dimension scaling — full metrics pivot  (`tables/E6_dim_scaling.md`)
- 8. E8 tail-heaviness scaling (df 1.5→50)  (`tables/E8_tail_scaling.md`)
- 9. E9 anisotropy scaling (cond 1→10^5)  (`tables/E9_aniso.md`)
- 7. E7 sample-size scaling (n 500→50k)  (`tables/E7_sample_efficiency.md`)
- F1. Finance d49 (chronological)  (`tables/E_finance_ff49.md`)
- F2. Finance d49 (random split ablation)  (`tables/E_finance_randomsplit.md`)
- W1. Weather d96 (MLP 3×128)  (`tables/E_weather_au_wind.md`)
- W2. Weather d96 (residual MLP 256×4)  (`tables/E_weather_bigmodel.md`)
- P. PIV d64 (real)  (`tables/E10_piv_d64.md`)
- T. Toy2D d2 (all methods incl. MSGM & no_proj)  (`tables/EXISTING_toy2d.md`)
- S. Student-t d16 (existing, incl. MSGM & no_proj)  (`tables/EXISTING_studentt_d16_full.md`)

---


## 0. MASTER — all benchmarks × all methods (radial/sliced/ks/train)

# MASTER results — all benchmark experiments × all methods

Per-method mean±pstd over seeds. radial_w1 = radial Wasserstein-1; sliced_w1 = global sliced-W1; ks = radial KS; train_s = training seconds/seed. Lower is better (except train_s = cost). Scaling sweeps (dim, tail, anisotropy, sample-size) are parameter curves — see `tables/E6_dim_scaling.md`, `E8_tail_scaling.md`, `E9_aniso.md`, `E7_sample_efficiency.md`.

| dataset / setting | method | radial_w1 | sliced_w1 | ks | train_s |
|---|---|---|---|---|---|
| Student-t d16 (repro, b4096) | gaussian_fm | 1.731±0.398 | 0.420±0.020 | 0.073±0.024 | 17.220±0.141 |
|  | source_only_oracle | 0.443±0.136 | 0.338±0.043 | 0.017±0.004 | 18.201±0.074 |
|  | source_only_empirical | 0.398±0.052 | 0.309±0.035 | 0.018±0.004 | 18.623±0.408 |
|  | rafm_oracle | 0.287±0.071 | 0.276±0.039 | 0.013±0.002 | 35.162±0.183 |
|  | rafm_empirical | 0.261±0.008 | 0.268±0.039 | 0.013±0.001 | 35.166±0.305 |
| Student-t d32 (repro, b4096) | gaussian_fm | 9.415±0.418 | 1.437±0.022 | 0.337±0.015 | 20.391±0.862 |
|  | source_only_oracle | 1.077±0.204 | 0.791±0.123 | 0.035±0.002 | 21.748±0.661 |
|  | source_only_empirical | 0.954±0.044 | 0.770±0.090 | 0.036±0.002 | 19.358±0.376 |
|  | rafm_oracle | 0.370±0.112 | 0.429±0.083 | 0.011±0.002 | 34.516±0.298 |
|  | rafm_empirical | 0.352±0.054 | 0.440±0.094 | 0.012±0.002 | 33.199±0.300 |
| Gaussian-aniso d16 (control) | gaussian_fm | 0.083±0.034 | 0.147±0.034 | 0.013±0.003 | 17.544±0.092 |
|  | source_only_oracle | 0.130±0.002 | 0.147±0.021 | 0.017±0.001 | 18.197±0.017 |
|  | source_only_empirical | 0.113±0.019 | 0.149±0.018 | 0.018±0.003 | 18.454±0.294 |
|  | rafm_oracle | 0.072±0.005 | 0.104±0.008 | 0.012±0.002 | 32.740±0.188 |
|  | rafm_empirical | 0.055±0.009 | 0.104±0.007 | 0.012±0.002 | 32.848±0.102 |
| Toy2D d2 (existing) | gaussian_fm | 0.082±0.043 | 0.080±0.013 | 0.056±0.007 | 18.451±0.085 |
|  | msgm | 0.049±0.009 | 0.045±0.004 | 0.020±0.003 | 3012 |
|  | source_only_oracle | 0.051±0.015 | 0.077±0.017 | 0.037±0.008 | 19.400±0.042 |
|  | source_only_empirical | 0.069±0.025 | 0.089±0.024 | 0.033±0.009 | 19.409±0.123 |
|  | rafm_oracle | 0.024±0.006 | 0.336±0.049 | 0.033±0.027 | 38.298±0.179 |
|  | rafm_empirical | 0.023±0.005 | 0.337±0.042 | 0.029±0.029 | 38.467±0.178 |
|  | rafm_empirical_no_proj | 26.700±25.658 | 17.006±16.027 | 0.614±0.279 | — |
| PIV d64 (real) | gaussian_fm | 0.192±0.014 | 0.035±0.001 | 0.163±0.004 | 17.062±0.297 |
|  | source_only_empirical | 0.118±0.012 | 0.025±0.003 | 0.116±0.012 | 18.067±0.128 |
|  | rafm_empirical | 0.048±0.002 | 0.025±0.001 | 0.047±0.003 | 32.891±0.294 |
| Finance d49 (real, chrono) | gaussian_fm | 1.418±0.113 | 0.194±0.018 | 0.218±0.009 | 17.605±0.078 |
|  | msgm | 1.417±0.033 | 0.182±0.003 | 0.197±0.006 | 18658 |
|  | source_only_empirical | 1.707±0.040 | 0.208±0.004 | 0.246±0.001 | 18.877±0.120 |
|  | rafm_empirical | 1.423±0.040 | 0.174±0.003 | 0.200±0.003 | 34.688±0.071 |
| Finance d49 (real, random) | gaussian_fm | 0.252±0.038 | 0.069±0.004 | 0.034±0.002 | 17.037±0.112 |
|  | source_only_empirical | 0.351±0.024 | 0.069±0.006 | 0.066±0.008 | 17.911±0.052 |
|  | rafm_empirical | 0.175±0.014 | 0.057±0.006 | 0.020±0.001 | 38.912±0.093 |
| Weather d96 (real, MLP3x128) | gaussian_fm | 0.179±0.031 | 0.085±0.007 | 0.044±0.010 | 17.633±0.070 |
|  | msgm | 0.099±0.028 | 0.120±0.003 | 0.033±0.009 | 20085 |
|  | source_only_empirical | 0.380±0.037 | 0.094±0.007 | 0.084±0.018 | 18.591±0.045 |
|  | rafm_empirical | 0.087±0.005 | 0.075±0.004 | 0.033±0.002 | 36.143±1.766 |
| Weather d96 (real, resMLP) | gaussian_fm | 0.311±0.211 | 0.105±0.008 | 0.091±0.044 | 34.424±0.364 |
|  | source_only_empirical | 0.185±0.113 | 0.092±0.010 | 0.049±0.024 | 34.774±0.187 |
|  | rafm_empirical | 0.087±0.005 | 0.081±0.009 | 0.033±0.002 | 59.884±0.982 |


## 1. E1 reproduction vs paper

# E1 — reproduced vs paper (RESULTS_SUMMARY.md)

Reproduced = 3-seed mean±std from this run (batch 4096, matched settings). Paper = value from RESULTS_SUMMARY.md.


## student_t_d16

| method | metric | paper | reproduced (mean±std) | within 1 std? |
|---|---|---|---|---|
| gaussian_fm | radial_w1 | 1.500 | 1.731+/-0.398 | yes |
| gaussian_fm | sliced_w1 | 0.453 | 0.420+/-0.020 | yes |
| source_only_oracle | radial_w1 | 0.569 | 0.443+/-0.136 | yes |
| source_only_oracle | sliced_w1 | 0.350 | 0.338+/-0.043 | yes |
| source_only_empirical | radial_w1 | 0.412 | 0.398+/-0.052 | yes |
| source_only_empirical | sliced_w1 | 0.345 | 0.309+/-0.035 | yes |
| rafm_oracle | radial_w1 | 0.372 | 0.287+/-0.071 | yes |
| rafm_oracle | sliced_w1 | 0.266 | 0.276+/-0.039 | yes |
| rafm_empirical | radial_w1 | 0.329 | 0.261+/-0.008 | yes |
| rafm_empirical | sliced_w1 | 0.263 | 0.268+/-0.039 | yes |

## student_t_d32

| method | metric | paper | reproduced (mean±std) | within 1 std? |
|---|---|---|---|---|
| gaussian_fm | radial_w1 | 9.696 | 9.415+/-0.418 | yes |
| gaussian_fm | sliced_w1 | 1.388 | 1.437+/-0.022 | yes |
| source_only_oracle | radial_w1 | 0.744 | 1.077+/-0.204 | check |
| source_only_oracle | sliced_w1 | 0.573 | 0.791+/-0.123 | check |
| rafm_empirical | radial_w1 | 0.406 | 0.352+/-0.054 | yes |
| rafm_empirical | sliced_w1 | 0.440 | 0.440+/-0.094 | yes |

## gaussian_aniso_d16

| method | metric | paper | reproduced (mean±std) | within 1 std? |
|---|---|---|---|---|
| gaussian_fm | radial_w1 | 0.128 | 0.083+/-0.034 | check |
| gaussian_fm | sliced_w1 | 0.159 | 0.147+/-0.034 | yes |
| rafm_empirical | radial_w1 | 0.114 | 0.055+/-0.009 | check |
| rafm_empirical | sliced_w1 | 0.108 | 0.104+/-0.007 | yes |


## 2a. E2 attribution — Student-t d16

# E2 attribution — student_t_d16

Lower is better for all metrics. Delta<0 = improvement from that component. Uncertainty = quadrature of per-method stds (3 seeds).

## Absolute (mean±std)
| metric | gaussian_fm | source_only_empirical | rafm_empirical |
|---|---|---|---|
| radial_w1 | 1.731+/-0.4 | 0.3981+/-0.052 | 0.261+/-0.008 |
| ks_stat | 0.07323+/-0.024 | 0.01813+/-0.0036 | 0.0127+/-0.00062 |
| sliced_w1 | 0.4203+/-0.02 | 0.309+/-0.035 | 0.2685+/-0.039 |
| mmd | 0.001135+/-0.00021 | 0.0007889+/-0.00018 | 0.000573+/-0.00019 |
| angular_sw_mean | 0.01469+/-0.0013 | 0.01514+/-0.00045 | 0.01343+/-0.00071 |
| q995_err | 0.2714+/-0.02 | 0.0525+/-0.011 | 0.05511+/-0.013 |
| tail_exc_99 | 0.008467+/-0.00019 | 0.001167+/-0.00026 | 0.0005333+/-0.00026 |

## Decomposition (delta per component)
| metric | source correction (Gaussian->eCDF source) | spherical path (Euclid->slerp) | total |
|---|---|---|---|
| radial_w1 | -1.333+/-0.4 | -0.1371+/-0.053 | -1.47 |
| ks_stat | -0.0551+/-0.025 | -0.005433+/-0.0037 | -0.06053 |
| sliced_w1 | -0.1113+/-0.041 | -0.04055+/-0.052 | -0.1519 |
| mmd | -0.0003459+/-0.00028 | -0.0002159+/-0.00026 | -0.0005618 |
| angular_sw_mean | +0.0004489+/-0.0014 | -0.00171+/-0.00084 | -0.001261 |
| q995_err | -0.2189+/-0.023 | +0.002608+/-0.017 | -0.2163 |
| tail_exc_99 | -0.0073+/-0.00032 | -0.0006333+/-0.00037 | -0.007933 |


## 2b. E2 attribution — Student-t d32

# E2 attribution — student_t_d32

Lower is better for all metrics. Delta<0 = improvement from that component. Uncertainty = quadrature of per-method stds (3 seeds).

## Absolute (mean±std)
| metric | gaussian_fm | source_only_empirical | rafm_empirical |
|---|---|---|---|
| radial_w1 | 9.415+/-0.42 | 0.9544+/-0.044 | 0.3523+/-0.054 |
| ks_stat | 0.3373+/-0.015 | 0.03567+/-0.0021 | 0.0123+/-0.0019 |
| sliced_w1 | 1.437+/-0.022 | 0.7696+/-0.09 | 0.4401+/-0.094 |
| mmd | 0.007109+/-0.00061 | 0.002646+/-0.00076 | 0.0007305+/-0.00035 |
| angular_sw_mean | 0.02166+/-0.0011 | 0.01603+/-0.0016 | 0.01054+/-0.0013 |
| q995_err | 0.4064+/-0.0047 | 0.03428+/-0.032 | 0.03463+/-0.035 |
| tail_exc_99 | 0.009967+/-4.7e-05 | 0.0009333+/-0.00061 | 0.001767+/-0.00033 |

## Decomposition (delta per component)
| metric | source correction (Gaussian->eCDF source) | spherical path (Euclid->slerp) | total |
|---|---|---|---|
| radial_w1 | -8.461+/-0.42 | -0.602+/-0.07 | -9.063 |
| ks_stat | -0.3017+/-0.015 | -0.02337+/-0.0028 | -0.325 |
| sliced_w1 | -0.667+/-0.093 | -0.3295+/-0.13 | -0.9965 |
| mmd | -0.004463+/-0.00097 | -0.001915+/-0.00083 | -0.006378 |
| angular_sw_mean | -0.005635+/-0.0019 | -0.00549+/-0.0021 | -0.01112 |
| q995_err | -0.3721+/-0.032 | +0.0003465+/-0.047 | -0.3718 |
| tail_exc_99 | -0.009033+/-0.00061 | +0.0008333+/-0.0007 | -0.0082 |


## 2c. E2 attribution — Gaussian d16

# E2 attribution — gaussian_aniso_d16

Lower is better for all metrics. Delta<0 = improvement from that component. Uncertainty = quadrature of per-method stds (3 seeds).

## Absolute (mean±std)
| metric | gaussian_fm | source_only_empirical | rafm_empirical |
|---|---|---|---|
| radial_w1 | 0.08257+/-0.034 | 0.1134+/-0.019 | 0.05494+/-0.0089 |
| ks_stat | 0.01297+/-0.0034 | 0.01783+/-0.0025 | 0.01237+/-0.0019 |
| sliced_w1 | 0.1473+/-0.034 | 0.1495+/-0.018 | 0.104+/-0.0075 |
| mmd | 0.0005244+/-0.00023 | 0.0005052+/-0.00015 | 0.0002222+/-2.9e-05 |
| angular_sw_mean | 0.01378+/-0.0017 | 0.01439+/-0.0015 | 0.01199+/-0.00052 |
| q995_err | 0.008788+/-0.0053 | 0.01422+/-0.0049 | 0.01056+/-0.0045 |
| tail_exc_99 | 0.0009333+/-0.00074 | 0.002+/-0.00094 | 0.0007333+/-0.00041 |

## Decomposition (delta per component)
| metric | source correction (Gaussian->eCDF source) | spherical path (Euclid->slerp) | total |
|---|---|---|---|
| radial_w1 | +0.03088+/-0.039 | -0.05851+/-0.021 | -0.02763 |
| ks_stat | +0.004867+/-0.0042 | -0.005467+/-0.0032 | -0.0006 |
| sliced_w1 | +0.002238+/-0.038 | -0.04554+/-0.019 | -0.0433 |
| mmd | -1.919e-05+/-0.00027 | -0.000283+/-0.00015 | -0.0003022 |
| angular_sw_mean | +0.0006053+/-0.0023 | -0.0024+/-0.0015 | -0.001794 |
| q995_err | +0.005433+/-0.0073 | -0.003658+/-0.0067 | +0.001775 |
| tail_exc_99 | +0.001067+/-0.0012 | -0.001267+/-0.001 | -0.0002 |


## 3. E3 directional — common-radial cr_sliced_w1 (mean±std)

# E3 common-radial diagnostic — cr_sliced_w1 (mean±std, Student-t d16, 3 seeds)

| method | cr_sliced_w1 (mean±std) | n |
|---|---|---|
| gaussian_fm | 0.508±0.122 | 3 |
| source_only_empirical | 0.387±0.032 | 3 |
| msgm | 0.463±0.024 | 3 |
| rafm_empirical | 0.296±0.028 | 3 |
| rafm_oracle | 0.295±0.029 | 3 |
| source_only_oracle | 0.387±0.032 | 3 |
| rafm_empirical_no_proj | 0.295±0.016 | 3 |


## 4. E4 — NFE×solver + radius drift (synthetic d16)

# E4 — NFE x solver x radius-drift summary (fixed checkpoint)


## RAFM empirical d16 (spherical, projection ON vs OFF)

| solver | NFE | proj | radial_w1 | sliced_w1 | drift_final_mean | nan |
|---|---|---|---|---|---|---|
| rk4 | 4 | True | 1.600 | 0.400 | 1.76e+00 | 0.000 |
| rk4 | 4 | False | 0.766 | 0.304 | 1.62e+00 | 0.000 |
| rk4 | 4 | True | 1.600 | 0.400 | 1.76e+00 | 0.000 |
| rk4 | 4 | False | 0.766 | 0.304 | 1.62e+00 | 0.000 |
| rk4 | 4 | True | 1.600 | 0.400 | 1.76e+00 | 0.000 |
| rk4 | 4 | False | 0.766 | 0.304 | 1.62e+00 | 0.000 |
| rk4 | 8 | True | 0.211 | 0.262 | 6.71e-02 | 0.000 |
| rk4 | 8 | False | 0.464 | 0.280 | 1.30e+00 | 0.000 |
| rk4 | 20 | True | 0.246 | 0.262 | 7.02e-03 | 0.000 |
| rk4 | 20 | False | 0.442 | 0.279 | 1.28e+00 | 0.000 |
| rk4 | 48 | True | 0.251 | 0.262 | 3.86e-04 | 0.000 |
| rk4 | 48 | False | 0.437 | 0.279 | 1.28e+00 | 0.000 |
| rk4 | 100 | True | 0.251 | 0.262 | 1.74e-05 | 0.000 |
| rk4 | 100 | False | 0.437 | 0.279 | 1.28e+00 | 0.000 |

## Gaussian FM d16 (euclidean, no projection)

| solver | NFE | proj | radial_w1 | sliced_w1 | drift_final_mean | nan |
|---|---|---|---|---|---|---|
| rk4 | 4 | False | 9.382 | 1.906 | 1.09e+01 | 0.000 |
| rk4 | 4 | False | 9.382 | 1.906 | 1.09e+01 | 0.000 |
| rk4 | 4 | False | 9.382 | 1.906 | 1.09e+01 | 0.000 |
| rk4 | 8 | False | 4.073 | 0.851 | 1.62e+01 | 0.000 |
| rk4 | 20 | False | 2.929 | 0.593 | 2.23e+01 | 0.000 |
| rk4 | 48 | False | 1.812 | 0.419 | 2.11e+01 | 0.000 |
| rk4 | 100 | False | 1.898 | 0.434 | 2.12e+01 | 0.000 |


## 6a. E6 dimension scaling — radial_w1 by dim (mean±std)

# E6 radial_w1 by dimension (mean±std, 3 seeds)

| method | d2 | d8 | d16 | d32 | d64 | d128 | d256 |
|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.013±0.002 | 0.411±0.118 | 1.749±0.744 | 9.834±0.093 | 10.646±0.639 | 124.260±1.425 | 347.687±0.623 |
| source_only_empirical | 0.019±0.002 | 0.184±0.013 | 0.494±0.070 | 1.282±0.308 | 9.218±0.178 | 18.683±0.184 | 51.053±2.413 |
| rafm_oracle | 0.132±0.169 | 0.179±0.028 | 0.253±0.079 | 0.383±0.031 | 0.929±0.095 | 1.098±0.263 | 2.676±0.177 |
| rafm_empirical | 0.134±0.170 | 0.150±0.007 | 0.190±0.018 | 0.435±0.065 | 0.982±0.121 | 0.850±0.040 | 2.703±0.138 |


## 6b. E6 dimension scaling — full metrics pivot

# Scaling (dim) — E6_dim_scaling


## radial_w1 vs dim

| method | dim=2 | dim=8 | dim=16 | dim=32 | dim=64 | dim=128 | dim=256 |
|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.0132 | 0.4109 | 1.749 | 9.834 | 10.65 | 124.3 | 347.7 |
| rafm_empirical | 0.134 | 0.1504 | 0.1904 | 0.4353 | 0.9821 | 0.8503 | 2.703 |
| rafm_oracle | 0.1321 | 0.179 | 0.2535 | 0.3829 | 0.9288 | 1.098 | 2.676 |
| source_only_empirical | 0.01864 | 0.1839 | 0.4935 | 1.282 | 9.218 | 18.68 | 51.05 |

## sliced_w1 vs dim

| method | dim=2 | dim=8 | dim=16 | dim=32 | dim=64 | dim=128 | dim=256 |
|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.01719 | 0.2054 | 0.6902 | 1.44 | 1.54 | 9.017 | 17.81 |
| rafm_empirical | 0.4667 | 0.1715 | 0.2454 | 0.4297 | 0.8401 | 1.007 | 1.241 |
| rafm_oracle | 0.4684 | 0.168 | 0.2424 | 0.4149 | 0.8442 | 1.016 | 1.221 |
| source_only_empirical | 0.02305 | 0.1765 | 0.2779 | 0.5613 | 1.428 | 1.682 | 2.879 |

## angular_sw_mean vs dim

| method | dim=2 | dim=8 | dim=16 | dim=32 | dim=64 | dim=128 | dim=256 |
|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.03458 | 0.01865 | 0.02514 | 0.02192 | 0.01481 | 0.03315 | - |
| rafm_empirical | 0.8086 | 0.0163 | 0.01289 | 0.01083 | 0.009863 | 0.006028 | 0.003749 |
| rafm_oracle | 0.808 | 0.01729 | 0.01401 | 0.01055 | 0.009729 | 0.005875 | 0.003722 |
| source_only_empirical | 0.05578 | 0.01585 | 0.01438 | 0.01321 | 0.01185 | 0.006273 | 0.004392 |

## nan_rate vs dim

| method | dim=2 | dim=8 | dim=16 | dim=32 | dim=64 | dim=128 | dim=256 |
|---|---|---|---|---|---|---|---|
| gaussian_fm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rafm_empirical | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rafm_oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| source_only_empirical | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## total_train_time_s vs dim

| method | dim=2 | dim=8 | dim=16 | dim=32 | dim=64 | dim=128 | dim=256 |
|---|---|---|---|---|---|---|---|
| gaussian_fm | 17.2 | 17.14 | 17.08 | 17.26 | 17.35 | 18.11 | 17.68 |
| rafm_empirical | 65.09 | 32.82 | 32.75 | 33.17 | 32.72 | 32.69 | 33.9 |
| rafm_oracle | 66.84 | 32.78 | 32.81 | 33.75 | 32.87 | 32.95 | 34.27 |
| source_only_empirical | 18.22 | 18.07 | 18.03 | 18.5 | 18.14 | 18.86 | 18.55 |


## 8. E8 tail-heaviness scaling (df 1.5→50)

# Scaling (df) — E8_tail_scaling


## radial_w1 vs df

| method | df=1.5 | df=2 | df=3 | df=5 | df=10 | df=50 |
|---|---|---|---|---|---|---|
| gaussian_fm | 44.27 | 19.12 | 1.821 | 0.4701 | 0.1912 | 0.08155 |
| rafm_empirical | 4.021 | 1.069 | 0.2378 | 0.09994 | 0.1563 | 0.0726 |
| rafm_oracle | 4.191 | 1.272 | 0.2604 | 0.1173 | 0.0805 | 0.07347 |
| source_only_empirical | 8.651 | 3.345 | 0.354 | 0.203 | 0.2119 | 0.1396 |

## sliced_w1 vs df

| method | df=1.5 | df=2 | df=3 | df=5 | df=10 | df=50 |
|---|---|---|---|---|---|---|
| gaussian_fm | 11.17 | 3.953 | 0.4454 | 0.179 | 0.1742 | 0.1481 |
| rafm_empirical | 2.858 | 0.8843 | 0.2209 | 0.1306 | 0.1203 | 0.1044 |
| rafm_oracle | 2.907 | 0.8878 | 0.2287 | 0.1275 | 0.1155 | 0.104 |
| source_only_empirical | 3.386 | 1.227 | 0.3942 | 0.2014 | 0.1413 | 0.1261 |

## angular_sw_mean vs df

| method | df=1.5 | df=2 | df=3 | df=5 | df=10 | df=50 |
|---|---|---|---|---|---|---|
| gaussian_fm | 0.109 | 0.044 | 0.01556 | 0.01259 | 0.01344 | 0.01372 |
| rafm_empirical | 0.03598 | 0.02278 | 0.0127 | 0.011 | 0.0118 | 0.01124 |
| rafm_oracle | 0.03659 | 0.02185 | 0.01306 | 0.01136 | 0.0118 | 0.0124 |
| source_only_empirical | 0.0327 | 0.0271 | 0.01789 | 0.0135 | 0.01267 | 0.01243 |

## nan_rate vs df

| method | df=1.5 | df=2 | df=3 | df=5 | df=10 | df=50 |
|---|---|---|---|---|---|---|
| gaussian_fm | 0 | 0 | 0 | 0 | 0 | 0 |
| rafm_empirical | 0 | 0 | 0 | 0 | 0 | 0 |
| rafm_oracle | 0 | 0 | 0 | 0 | 0 | 0 |
| source_only_empirical | 0 | 0 | 0 | 0 | 0 | 0 |

## total_train_time_s vs df

| method | df=1.5 | df=2 | df=3 | df=5 | df=10 | df=50 |
|---|---|---|---|---|---|---|
| gaussian_fm | 17.22 | 17.22 | 17.1 | 17.04 | 16.96 | 17.08 |
| rafm_empirical | 32.86 | 32.38 | 32.73 | 32.36 | 32.74 | 32.73 |
| rafm_oracle | 32.82 | 32.41 | 32.39 | 32.32 | 32.57 | 32.43 |
| source_only_empirical | 18.13 | 18.28 | 18.02 | 17.88 | 17.98 | 17.91 |


## 9. E9 anisotropy scaling (cond 1→10^5)

# Scaling (kappa) — E9_aniso


## radial_w1 vs kappa

| method | kappa=1 | kappa=3 | kappa=10 | kappa=30 | kappa=100 | kappa=300 |
|---|---|---|---|---|---|---|
| gaussian_fm | 0.01821 | 0.5845 | 0.3821 | 1.959 | 8.444 | 37.37 |
| rafm_empirical | 0.01017 | 0.03729 | 0.07427 | 0.2287 | 0.9802 | 3.247 |
| source_only_empirical | 0.01515 | 0.08795 | 0.4381 | 1.428 | 3.828 | 24.04 |

## sliced_w1 vs kappa

| method | kappa=1 | kappa=3 | kappa=10 | kappa=30 | kappa=100 | kappa=300 |
|---|---|---|---|---|---|---|
| gaussian_fm | 0.03055 | 0.104 | 0.2409 | 1.005 | 2.781 | 13.78 |
| rafm_empirical | 0.02148 | 0.0371 | 0.1336 | 0.4491 | 1.096 | 4.203 |
| source_only_empirical | 0.03032 | 0.06335 | 0.2216 | 0.5493 | 1.716 | 5.74 |

## angular_sw_mean vs kappa

| method | kappa=1 | kappa=3 | kappa=10 | kappa=30 | kappa=100 | kappa=300 |
|---|---|---|---|---|---|---|
| gaussian_fm | 0.008836 | 0.009474 | 0.01119 | 0.01488 | 0.0161 | 0.02982 |
| rafm_empirical | 0.008273 | 0.008351 | 0.008281 | 0.009669 | 0.009725 | 0.01202 |
| source_only_empirical | 0.008955 | 0.009457 | 0.01064 | 0.01103 | 0.01147 | 0.0104 |

## nan_rate vs kappa

| method | kappa=1 | kappa=3 | kappa=10 | kappa=30 | kappa=100 | kappa=300 |
|---|---|---|---|---|---|---|
| gaussian_fm | 0 | 0 | 0 | 0 | 0 | 0 |
| rafm_empirical | 0 | 0 | 0 | 0 | 0 | 0 |
| source_only_empirical | 0 | 0 | 0 | 0 | 0 | 0 |

## total_train_time_s vs kappa

| method | kappa=1 | kappa=3 | kappa=10 | kappa=30 | kappa=100 | kappa=300 |
|---|---|---|---|---|---|---|
| gaussian_fm | 17.99 | 17.67 | 17.71 | 17.84 | 17.99 | 17.7 |
| rafm_empirical | 35.63 | 34.91 | 34.5 | 34.81 | 34.75 | 34.64 |
| source_only_empirical | 18.92 | 18.7 | 18.69 | 18.79 | 18.9 | 18.95 |


## 7. E7 sample-size scaling (n 500→50k)

# E7 sample-size scaling (existing repo exp2, Student-t d16; batch 256)


## radial_w1 vs n_train
| method | n=500 | n=1000 | n=5000 | n=20000 | n=50000 |
|---|---|---|---|---|---|
| gaussian_fm | 0.9395 | 1.123 | 1.463 | 1.541 | 1.551 |
| rafm_empirical_ecdf | 0.8845 | 0.9625 | 0.4004 | 0.3891 | 0.3711 |
| rafm_oracle | 1.408 | 0.9136 | 0.4158 | 0.3675 | 0.376 |
| source_only_empirical_ecdf | 0.7592 | 0.9746 | 0.3542 | 0.323 | 0.278 |
| source_only_empirical_log | 0.7566 | 0.9767 | 0.3543 | 0.323 | 0.278 |
| source_only_oracle | 0.9638 | 0.6944 | 0.3135 | 0.3148 | 0.3135 |

## sliced_w1 vs n_train
| method | n=500 | n=1000 | n=5000 | n=20000 | n=50000 |
|---|---|---|---|---|---|
| gaussian_fm | 0.4829 | 0.4111 | 0.3485 | 0.4095 | 0.5264 |
| rafm_empirical_ecdf | 0.4978 | 0.3886 | 0.2471 | 0.2256 | 0.257 |
| rafm_oracle | 0.5174 | 0.4126 | 0.281 | 0.2216 | 0.2701 |
| source_only_empirical_ecdf | 0.4819 | 0.4201 | 0.2896 | 0.3408 | 0.3276 |
| source_only_empirical_log | 0.4814 | 0.4202 | 0.2896 | 0.3408 | 0.3276 |
| source_only_oracle | 0.4829 | 0.4265 | 0.2989 | 0.3279 | 0.355 |


## F1. Finance d49 (chronological)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 1.418+/-0.11 | 0.2184+/-0.009 | 0.1941+/-0.018 | 0.0117+/-0.0019 | 0.02356+/-0.0025 | 0.1903+/-0.056 | 0.3108+/-0.039 | 0.006169+/-0.0013 | 0 | 17.61+/-0.078 | 0.1642+/-0.01 |
| msgm | 1.417+/-0.033 | 0.1975+/-0.0059 | 0.182+/-0.0029 | 0.01243+/-0.00046 | 0.01947+/-0.00041 | 0.255+/-0.02 | 0.3751+/-0.018 | 0.00733+/-0.00016 | 0 | 1.866e+04+/-4.6e+02 | 1.483+/-0.01 |
| rafm_empirical | 1.423+/-0.04 | 0.2003+/-0.0028 | 0.1741+/-0.0026 | 0.009796+/-0.0006 | 0.01726+/-0.00033 | 0.2759+/-0.0065 | 0.3812+/-0.021 | 0.007214+/-0.00075 | 0 | 34.69+/-0.071 | 0.2643+/-0.019 |
| source_only_empirical | 1.707+/-0.04 | 0.246+/-0.00091 | 0.2076+/-0.0045 | 0.01312+/-0.00042 | 0.02054+/-0.00061 | 0.26+/-0.022 | 0.3498+/-0.023 | 0.006981+/-0.00043 | 0 | 18.88+/-0.12 | 0.1743+/-0.0087 |


## F2. Finance d49 (random split ablation)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.2516+/-0.038 | 0.03355+/-0.0018 | 0.06901+/-0.0045 | 0.001092+/-0.00049 | 0.01389+/-0.0018 | 0.05941+/-0.027 | 0.2194+/-0.078 | 0.002423+/-0.0011 | 0 | 17.04+/-0.11 | 0.1404+/-0.017 |
| rafm_empirical | 0.1748+/-0.014 | 0.0202+/-0.001 | 0.05703+/-0.0057 | 0.0006429+/-0.0003 | 0.01296+/-8.1e-05 | 0.09495+/-0.01 | 0.04371+/-0.03 | 0.002569+/-0.00072 | 0 | 38.91+/-0.093 | 0.2329+/-0.014 |
| source_only_empirical | 0.3512+/-0.024 | 0.06583+/-0.0077 | 0.06881+/-0.0058 | 0.001136+/-0.00025 | 0.0134+/-0.001 | 0.09628+/-0.021 | 0.031+/-0.026 | 0.001989+/-0.00075 | 0 | 17.91+/-0.052 | 0.1622+/-0.0054 |


## W1. Weather d96 (MLP 3×128)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.1786+/-0.031 | 0.04416+/-0.01 | 0.08542+/-0.0075 | 0.00267+/-0.00067 | 0.01496+/-0.00015 | 0.1016+/-0.019 | 0.09711+/-0.042 | 0.02077+/-0.0012 | 0 | 17.63+/-0.07 | 0.1608+/-0.0059 |
| msgm | 0.09868+/-0.028 | 0.03305+/-0.0093 | 0.1199+/-0.0029 | 0.004187+/-0.00017 | 0.0187+/-0.00031 | 0.01904+/-0.0055 | 0.01301+/-0.0091 | 0.00339+/-0.0021 | 0 | 2.009e+04+/-5.6e+02 | 2.516+/-0.044 |
| rafm_empirical | 0.08727+/-0.0048 | 0.03305+/-0.0018 | 0.07456+/-0.0045 | 0.001909+/-0.00038 | 0.01342+/-0.00054 | 0.02769+/-0.0042 | 0.00828+/-0.0024 | 0.007949+/-0.0007 | 0 | 36.14+/-1.8 | 0.2633+/-0.022 |
| source_only_empirical | 0.3797+/-0.037 | 0.08376+/-0.018 | 0.09444+/-0.007 | 0.003005+/-0.00048 | 0.01499+/-0.00064 | 0.148+/-0.02 | 0.1689+/-0.044 | 0.03729+/-0.0053 | 0 | 18.59+/-0.045 | 0.1545+/-0.0058 |


## W2. Weather d96 (residual MLP 256×4)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.3107+/-0.21 | 0.09145+/-0.044 | 0.1049+/-0.0081 | 0.004356+/-0.00063 | 0.01604+/-0.00074 | 0.02125+/-0.015 | 0.02041+/-0.018 | 0.002308+/-0.0012 | 0 | 34.42+/-0.36 | 0.3612+/-0.021 |
| rafm_empirical | 0.08727+/-0.0048 | 0.03305+/-0.0018 | 0.08111+/-0.0088 | 0.002345+/-0.00057 | 0.01369+/-0.00093 | 0.02769+/-0.0042 | 0.00828+/-0.0024 | 0.007949+/-0.0007 | 0 | 59.88+/-0.98 | 0.4597+/-0.035 |
| source_only_empirical | 0.1847+/-0.11 | 0.049+/-0.024 | 0.09241+/-0.0097 | 0.003297+/-0.00074 | 0.01457+/-0.00029 | 0.01429+/-0.0092 | 0.01594+/-0.02 | 0.002479+/-0.0011 | 0 | 34.77+/-0.19 | 0.361+/-0.012 |


## P. PIV d64 (real)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.1916+/-0.014 | 0.1631+/-0.0045 | 0.0353+/-0.0014 | 0.007417+/-0.00082 | 0.02508+/-0.0011 | 0.03023+/-0.0081 | 0.02313+/-0.016 | 0.002333+/-0.00033 | 0 | 17.06+/-0.3 | 0.1619+/-0.017 |
| rafm_empirical | 0.04822+/-0.0019 | 0.04691+/-0.0026 | 0.0251+/-0.00056 | 0.003702+/-0.00034 | 0.02266+/-0.00016 | 0.02526+/-0.0035 | 0.05633+/-0.0096 | 0.0008+/-0.00029 | 0 | 32.89+/-0.29 | 0.2911+/-0.023 |
| source_only_empirical | 0.1177+/-0.012 | 0.1158+/-0.012 | 0.0252+/-0.0028 | 0.004136+/-0.0015 | 0.02329+/-0.0017 | 0.02837+/-0.009 | 0.0725+/-0.012 | 0.0018+/-0.00054 | 0 | 18.07+/-0.13 | 0.1767+/-0.0075 |


## T. Toy2D d2 (all methods incl. MSGM & no_proj)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 0.08214+/-0.043 | 0.05627+/-0.0068 | 0.07999+/-0.013 | 0.001288+/-0.00049 | 0.05016+/-0.0051 | 0.04126+/-0.026 | 0.08661+/-0.018 | 0.001233+/-0.00074 | 0 | 18.45+/-0.085 | 0.2633+/-0.038 |
| msgm | 0.04879+/-0.0087 | 0.0203+/-0.0026 | 0.04492+/-0.0038 | 0.0002869+/-3e-05 | 0.03475+/-0.0011 | 0.03694+/-0.024 | 0.02741+/-0.016 | 0.0006+/-0.00037 | 0 | 3012+/-17 | 0.7715+/-0.027 |
| rafm_empirical | 0.02327+/-0.0049 | 0.0291+/-0.029 | 0.3371+/-0.042 | 0.04076+/-0.015 | 0.4695+/-0.099 | 0.01332+/-0.0059 | 0.04631+/-0.034 | 0.0004+/-0.00024 | 0 | 38.47+/-0.18 | 0.378+/-0.042 |
| rafm_empirical_no_proj | 26.7+/-26 | 0.6144+/-0.28 | 17.01+/-16 | 0.1654+/-0.043 | 0.3573+/-0.045 | 170.2+/-1.7e+02 | 190.1+/-1.9e+02 | 0.2233+/-0.23 | 0.1477+/-0.21 | - | 0.416+/-0.057 |
| rafm_oracle | 0.02406+/-0.0063 | 0.0328+/-0.027 | 0.3358+/-0.049 | 0.04234+/-0.015 | 0.4668+/-0.097 | 0.02605+/-0.011 | 0.02702+/-0.0066 | 0.0006667+/-0.00045 | 0 | 38.3+/-0.18 | 0.3753+/-0.0097 |
| source_only_empirical | 0.06879+/-0.025 | 0.03303+/-0.009 | 0.08885+/-0.024 | 0.002198+/-0.0013 | 0.09802+/-0.038 | 0.0731+/-0.046 | 0.142+/-0.17 | 0.0017+/-0.00078 | 0 | 19.41+/-0.12 | 0.2142+/-0.0081 |
| source_only_oracle | 0.05071+/-0.015 | 0.0369+/-0.0079 | 0.07669+/-0.017 | 0.002145+/-0.0012 | 0.09143+/-0.038 | 0.03295+/-0.019 | 0.05868+/-0.065 | 0.001133+/-0.00049 | 0 | 19.4+/-0.042 | 0.2542+/-0.018 |


## S. Student-t d16 (existing, incl. MSGM & no_proj)

| method | radial_w1 | ks_stat | sliced_w1 | mmd | angular_sw_mean | q990_err | q995_err | tail_exc_99 | nan_rate | total_train_time_s | sample_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_fm | 3.341+/-0.91 | 0.1745+/-0.063 | 0.7595+/-0.23 | 0.004407+/-0.0029 | 0.02344+/-0.003 | 0.2927+/-0.025 | 0.3929+/-0.024 | 0.01 | 0 | 18.34+/-0.16 | 0.2402+/-0.034 |
| msgm | 0.376+/-0.065 | 0.01423+/-0.0036 | 0.4864+/-0.028 | 0.001844+/-0.0002 | 0.01987+/-0.001 | 0.07963+/-0.013 | 0.07771+/-0.019 | 0.0027+/-0.00071 | 0 | 2956+/-15 | 0.898+/-0.043 |
| rafm_empirical | 0.2264+/-0.048 | 0.0119+/-0.0029 | 0.3316+/-0.03 | 0.0007329+/-0.00012 | 0.01597+/-0.0016 | 0.03429+/-0.01 | 0.0516+/-0.034 | 0.0011+/-0.00029 | 0 | 35.25+/-0.068 | 0.399+/-0.082 |
| rafm_empirical_no_proj | 0.5317+/-0.051 | 0.02753+/-0.0027 | 0.3368+/-0.036 | 0.0007042+/-0.00016 | 0.01601+/-0.0011 | 0.05405+/-0.024 | 0.05382+/-0.026 | 0.001733+/-0.00092 | 0 | - | 0.3235+/-0.089 |
| rafm_oracle | 0.2377+/-0.056 | 0.01333+/-0.0037 | 0.3195+/-0.024 | 0.0007304+/-0.00015 | 0.01575+/-0.00086 | 0.02508+/-0.0085 | 0.06383+/-0.024 | 0.001+/-0.00016 | 0 | 35.01+/-0.17 | 0.3627+/-0.04 |
| source_only_empirical | 0.3986+/-0.044 | 0.0207+/-0.0055 | 0.4379+/-0.064 | 0.001411+/-0.00042 | 0.01873+/-0.0015 | 0.02901+/-0.023 | 0.04474+/-0.039 | 0.0009667+/-0.00094 | 0 | 18.96+/-0.34 | 0.2429+/-0.057 |
| source_only_oracle | 0.5083+/-0.13 | 0.0271+/-0.0048 | 0.4332+/-0.06 | 0.001411+/-0.00039 | 0.01867+/-0.0021 | 0.04824+/-0.0035 | 0.1108+/-0.01 | 0.0022+/-0.00028 | 0 | 19.22+/-0.035 | 0.2764+/-0.017 |
