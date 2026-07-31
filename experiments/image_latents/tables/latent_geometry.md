# Phase 2 — RAE/DINOv2-B Stage-2 latent geometry (Imagenette val, N=2000)

| quantity | mean | std | CoV | skew | kurt | q01 | q99 |
|---|---|---|---|---|---|---|---|
| global norm ||z|| (196608-d) | 504.986 | 8.181 | 0.0162 | -0.13 | -0.17 | 486.23 | 523.13 |
| per-token norm (768-d) | 31.542 | 1.214 | 0.0385 | 0.14 | 0.21 | 28.81 | 34.52 |

- between-class radius std (global norm): 5.744 ; within-class mean std: 5.675
- mean per-image token-norm CoV: 0.0346

## Decoder radius-scaling sensitivity (PSNR vs unscaled reconstruction)
| scale | 0.5 | 0.75 | 0.9 | 1.0 | 1.1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| PSNR dB | 20.0 | 27.7 | 35.9 | 120.0 | 36.6 | 30.7 | 27.3 |
