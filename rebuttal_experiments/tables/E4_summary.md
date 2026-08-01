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
