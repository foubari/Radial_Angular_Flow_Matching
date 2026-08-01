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
