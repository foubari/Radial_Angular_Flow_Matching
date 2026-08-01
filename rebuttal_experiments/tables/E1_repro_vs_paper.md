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
