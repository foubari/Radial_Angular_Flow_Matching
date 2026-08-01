# Phase 5 — DC-AE latent geometry (Imagenette val, N=2000, latent [32,8,8]=2048-d)

| quantity | mean | std | CoV | skew | kurt | q01 | q99 |
|---|---|---|---|---|---|---|---|
| global norm | 103.24 | 12.79 | 0.1239 | 0.79 | 1.28 | 78.1 | 143.0 |
| per-token norm | 12.74 | 2.63 | 0.2064 | 0.57 | 0.60 | 7.5 | 20.1 |

- between-class radius std 5.421 vs within-class 11.390

## Decoder radius sensitivity (PSNR vs unscaled, dB)
| scale | 0.5 | 0.75 | 0.9 | 1.0 | 1.1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| PSNR | 16.7 | 21.9 | 28.9 | 126.0 | 28.9 | 21.9 | 17.2 |
