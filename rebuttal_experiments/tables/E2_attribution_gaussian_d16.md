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
