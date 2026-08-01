# Phase 5 — small-sample FID sanity check (DC-AE decoded), NOT FID-50k

torchvision-InceptionV3 features, N≈2000/method, real ref = Imagenette val. Relative across methods only.

| method | small-FID (lower=better) |
|---|---|
| gaussian_euclidean | 300.69 |
| matched_euclidean | 322.08 |
| fixed_spherical | 309.7 |
| rafm_empirical | 294.24 |
| _decoder rFID floor_ | 21.34 |
