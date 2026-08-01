"""Prepare a gridded weather dataset from WeatherBench2 ERA5 (public GCS, no auth).

Source: gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr
        (accessed over public HTTPS via fsspec+zarr). License: ERA5/Copernicus, free for
        research with attribution; WeatherBench2 redistributes it publicly.

Representation: 10m wind components (u,v) over an Australia region box, flattened and
concatenated -> vector of dimension d = 2 * (nlat*nlon). Time-subsampled to keep the
download manageable. CHRONOLOGICAL 60/20/20 split, TRAIN-ONLY standardization.

Outputs:
  rebuttal_experiments/data/weather/weather_au_wind.pt   (N, d) standardized
  rebuttal_experiments/data/weather/weather_meta.json
  rebuttal_experiments/dataset_notes/weather_quick_check.md
"""
import json, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")

URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
LAT = (-45, -10); LON = (110, 155)   # Australia box
TIME_STEP = 16                        # every 16th 6h step (~4 days) -> ~5.8k samples
DATA = Path("rebuttal_experiments/data/weather"); NOTES = Path("rebuttal_experiments/dataset_notes")


def main():
    import xarray as xr, fsspec
    ds = xr.open_zarr(fsspec.get_mapper(URL), consolidated=True)
    sub = ds[VARS].isel(time=slice(0, None, TIME_STEP))
    sub = sub.sel(latitude=slice(LAT[0], LAT[1]), longitude=slice(LON[0], LON[1]))
    print("selecting region; sizes:", dict(sub.sizes), "downloading...", flush=True)
    arr = np.stack([sub[v].values for v in VARS], axis=1)  # (N, 2, nlon, nlat) or (N,2,nlat,nlon)
    # dims order is (time, longitude, latitude) per store; flatten last dims
    N = arr.shape[0]
    X = arr.reshape(N, -1).astype(np.float64)  # (N, d)
    times = sub.time.values
    d = X.shape[1]
    print(f"loaded X {X.shape} from {str(times[0])[:10]}..{str(times[-1])[:10]}", flush=True)

    # drop any rows with NaN (should be none for winds)
    keep = ~np.isnan(X).any(axis=1)
    X = X[keep]; times = times[keep]; N = X.shape[0]

    n_tr = int(N * 0.6); n_va = int(N * 0.2)
    tr = slice(0, n_tr)
    mean = X[tr].mean(axis=0); std = X[tr].std(axis=0) + 1e-12
    Z = (X - mean) / std

    DATA.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(Z, dtype=torch.float32), DATA / "weather_au_wind.pt")
    meta = {"dim": d, "N": int(N), "vars": VARS, "region_lat": LAT, "region_lon": LON,
            "time_step": TIME_STEP, "date_start": str(times[0])[:10], "date_end": str(times[-1])[:10],
            "split": {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, N]},
            "standardize": "train-only per-dim mean/std", "source": URL,
            "license": "ERA5/Copernicus (research use, attribution); via WeatherBench2 public bucket"}
    (DATA / "weather_meta.json").write_text(json.dumps(meta, indent=2))

    # quick check
    from scipy.stats import kurtosis
    r = np.linalg.norm(Z, axis=1)
    cov = np.cov(Z[:n_tr].T); eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    g = np.random.default_rng(0).standard_normal((10000, d)); rg = np.linalg.norm(g, axis=1)
    te = slice(n_tr + n_va, N); rte = r[te]; thr = np.quantile(rte, 0.9)
    qs = [0.5, 0.9, 0.99, 0.999]
    lines = [f"# Weather quick check — WeatherBench2 ERA5, 10m wind (u,v) over Australia\n",
             f"- Source: {URL}",
             f"- License: ERA5/Copernicus (research, attribution) via WeatherBench2 public bucket.",
             f"- Representation: 10m u+v wind, Australia box lat{LAT} lon{LON}, flattened -> d={d}.",
             f"- Time subsample: every {TIME_STEP}th 6h step; N={N}, {str(times[0])[:10]}..{str(times[-1])[:10]}.",
             f"- Split (chronological): train[0:{n_tr}] val[{n_tr}:{n_tr+n_va}] test[{n_tr+n_va}:{N}]; train-only standardization.\n",
             "## Norms & tails (standardized)",
             "- data norm q: " + ", ".join(f"q{q}={np.quantile(r,q):.2f}" for q in qs),
             "- gaussian-source norm q: " + ", ".join(f"q{q}={np.quantile(rg,q):.2f}" for q in qs),
             f"- data norm mean={r.mean():.2f} vs gaussian mean={rg.mean():.2f} (mismatch={abs(r.mean()-rg.mean()):.2f})",
             f"- mean marginal excess kurtosis={float(np.mean(kurtosis(Z,axis=0))):.2f}; radial excess kurtosis={float(kurtosis(r)):.2f}\n",
             "## Anisotropy",
             f"- cov eigenvalues top5: {np.round(eig[:5],2).tolist()}; condition number={float(eig[0]/eig[-1]):.1f}\n",
             "## Regime (test)",
             f"- high-wind threshold (‖x‖ q90)={thr:.2f}; ordinary mean ‖x‖={rte[rte<=thr].mean():.2f}; extreme mean ‖x‖={rte[rte>thr].mean():.2f}\n",
             "## Verdict",
             "Real gridded weather; spatially correlated wind fields (anisotropic), moderate tails. "
             "Higher dimension than WeatherAUS and physically natural norm. Suitable.",
             ]
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "weather_quick_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(l for l in lines if "‖" not in l))
    print("Saved weather tensor + meta + quick check.")


if __name__ == "__main__":
    main()
