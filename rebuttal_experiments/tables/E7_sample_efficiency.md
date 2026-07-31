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
