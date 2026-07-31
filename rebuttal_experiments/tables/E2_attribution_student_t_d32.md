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
