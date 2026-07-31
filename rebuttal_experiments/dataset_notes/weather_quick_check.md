# Weather quick check — WeatherBench2 ERA5, 10m wind (u,v) over Australia

- Source: https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr
- License: ERA5/Copernicus (research, attribution) via WeatherBench2 public bucket.
- Representation: 10m u+v wind, Australia box lat(-45, -10) lon(110, 155), flattened -> d=96.
- Time subsample: every 16th 6h step; N=5847, 1959-01-01..2023-01-09.
- Split (chronological): train[0:3508] val[3508:4677] test[4677:5847]; train-only standardization.

## Norms & tails (standardized)
- data norm q: q0.5=9.49, q0.9=11.94, q0.99=14.26, q0.999=15.57
- gaussian-source norm q: q0.5=9.77, q0.9=10.69, q0.99=11.43, q0.999=12.06
- data norm mean=9.63 vs gaussian mean=9.78 (mismatch=0.15)
- mean marginal excess kurtosis=-0.09; radial excess kurtosis=0.08

## Anisotropy
- cov eigenvalues top5: [17.6, 14.78, 12.99, 6.65, 5.73]; condition number=1530.5

## Regime (test)
- high-wind threshold (‖x‖ q90)=12.14; ordinary mean ‖x‖=9.26; extreme mean ‖x‖=13.05

## Verdict
Real gridded weather; spatially correlated wind fields (anisotropic), moderate tails. Higher dimension than WeatherAUS and physically natural norm. Suitable.
