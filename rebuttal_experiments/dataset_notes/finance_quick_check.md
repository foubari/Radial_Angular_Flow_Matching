# Finance quick check — Ken French 49 Industry (daily, value-weighted)

- Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip
- License: Kenneth French Data Library (free for research).
- N=14349 daily obs, d=49 industries, dates 19690701–20260529.
- Sentinel (-99.99) rate pre-clean: 5.2278%; rows dropped for NaN: 11904.
- Split (chronological): train[0:8609] val[8609:11478] test[11478:14349].
- Standardization: train-only per-dim mean/std.

## Norms & tails (standardized data)
- data ‖x‖ quantiles q50.0=5.41, q90.0=10.90, q95.0=13.65, q99.0=22.62, q99.9=45.10
- Gaussian-source ‖x‖ quantiles q50.0=6.96, q90.0=7.88, q95.0=8.14, q99.0=8.69, q99.9=9.20
- data norm mean=6.572 vs Gaussian mean=6.972 (mismatch=0.400)
- mean marginal excess kurtosis (per-dim) = 11.18 (0=Gaussian; >0 heavy-tailed)
- radial excess kurtosis = 41.32

## Anisotropy
- covariance eigenvalue spectrum (top5): [25.926, 1.768, 1.355, 1.014, 0.921]
- condition number (λmax/λmin) = 196.5

## Regime split (test)
- high-vol threshold (‖x‖ q90 of test) = 12.04; high-vol frac by construction ~10%.
- ordinary-period mean ‖x‖ = 6.326; high-vol mean ‖x‖ = 17.450

## Verdict
Real, heavy-tailed (excess kurtosis > 0), anisotropic (condition number >> 1), d=49 — usable; norm mismatch vs Gaussian source is the quantity RAFM targets.
