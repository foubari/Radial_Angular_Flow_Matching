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
