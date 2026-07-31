"""Prepare the Kenneth French 49 Industry Portfolios (daily, value-weighted) dataset.

Downloads the public ZIP from the Dartmouth data library, parses the first returns
block (Average Value Weighted Returns -- Daily), handles -99.99/-999 sentinels,
builds daily industry-return vectors (d=49), makes CHRONOLOGICAL 60/20/20 splits
(no leakage), standardizes using TRAIN-ONLY statistics, saves the processed tensor,
split indices, standardization stats, and a quick diagnostic markdown.

Source (public): https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip
License: free for research use (Kenneth R. French Data Library).

Outputs:
  rebuttal_experiments/data/finance/49_Industry_Portfolios_daily_CSV.zip  (raw)
  rebuttal_experiments/data/finance/finance_returns_raw.pt   (N,49) standardized, all periods
  rebuttal_experiments/data/finance/finance_meta.json        (dates, split idx, mean/std, sentinels)
  rebuttal_experiments/dataset_notes/finance_quick_check.md
"""
import io, json, zipfile, urllib.request, ssl, socket
from pathlib import Path
import numpy as np
import torch

ROOT = Path("rebuttal_experiments")
DATA = ROOT / "data/finance"
NOTES = ROOT / "dataset_notes"
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / "49_Industry_Portfolios_daily_CSV.zip"
    if zpath.exists():
        return zpath.read_bytes()
    socket.setdefaulttimeout(60)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context()) as r:
        data = r.read()
    zpath.write_bytes(data)
    return data


def parse_first_block(csv_text):
    """Return (dates[list int], names[list], values[np.ndarray N x d])."""
    lines = csv_text.splitlines()
    dates, rows, names = [], [], None
    started = False
    for ln in lines:
        s = ln.strip()
        if not s:
            if started:
                break  # blank line ends the first block
            continue
        parts = [p.strip() for p in ln.split(",")]
        # header row: first cell empty, rest are industry names (non-numeric)
        if not started and parts[0] == "" and len(parts) > 10:
            names = [p for p in parts[1:] if p]
            started = True
            continue
        if started:
            # data row: first cell is an 8-digit date
            if len(parts[0]) == 8 and parts[0].isdigit():
                dates.append(int(parts[0]))
                rows.append([float(x) for x in parts[1:1 + len(names)]])
            else:
                break
    vals = np.array(rows, dtype=np.float64)
    return dates, names, vals


def main():
    raw = download()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    csv_text = zf.read(csv_name).decode("latin-1")
    dates, names, vals = parse_first_block(csv_text)
    print(f"parsed block: {vals.shape} names={len(names)} dates {dates[0]}..{dates[-1]}")

    # sentinels -99.99 / -999 -> NaN
    vals[vals <= -99.99] = np.nan
    sentinel_rate = float(np.isnan(vals).mean())
    # drop rows with any NaN (rare after full history); returns already in percent -> to fraction
    keep = ~np.isnan(vals).any(axis=1)
    vals = vals[keep] / 100.0
    dates = [d for d, k in zip(dates, keep) if k]
    N, d = vals.shape

    # chronological split 60/20/20 (train first in time)
    n_tr = int(N * 0.6); n_va = int(N * 0.2)
    tr = np.arange(0, n_tr); va = np.arange(n_tr, n_tr + n_va); te = np.arange(n_tr + n_va, N)
    mean = vals[tr].mean(axis=0); std = vals[tr].std(axis=0) + 1e-12
    Z = (vals - mean) / std  # standardize with TRAIN-only stats

    DATA.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(Z, dtype=torch.float32), DATA / "finance_returns_raw.pt")
    meta = {"dim": d, "N": N, "names": names,
            "date_start": dates[0], "date_end": dates[-1],
            "sentinel_rate_preclean": sentinel_rate,
            "split": {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, N]},
            "standardize": "train-only per-dim mean/std", "returns_unit": "fraction (raw/100)",
            "source": URL, "license": "Kenneth French Data Library (research use)"}
    (DATA / "finance_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- quick diagnostic ----
    def norms(x): return np.linalg.norm(x, axis=1)
    r_all = norms(Z)
    from scipy.stats import kurtosis
    cov = np.cov(Z[tr].T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    aniso = float(eig[0] / eig[-1])
    # Gaussian-source norm reference (chi with d dof for standardized iso gaussian)
    g = np.random.default_rng(0).standard_normal((10000, d))
    rg = norms(g)
    # high-vol regime: top-10% rows by |return| L2 norm in test
    te_norms = r_all[te]
    thr = np.quantile(te_norms, 0.9)
    qs = [0.5, 0.9, 0.95, 0.99, 0.999]
    lines = [f"# Finance quick check — Ken French 49 Industry (daily, value-weighted)\n",
             f"- Source: {URL}",
             f"- License: Kenneth French Data Library (free for research).",
             f"- N={N} daily obs, d={d} industries, dates {dates[0]}–{dates[-1]}.",
             f"- Sentinel (-99.99) rate pre-clean: {sentinel_rate:.4%}; rows dropped for NaN: {int((~keep).sum())}.",
             f"- Split (chronological): train[0:{n_tr}] val[{n_tr}:{n_tr+n_va}] test[{n_tr+n_va}:{N}].",
             f"- Standardization: train-only per-dim mean/std.\n",
             "## Norms & tails (standardized data)",
             f"- data ‖x‖ quantiles " + ", ".join(f"q{int(q*1000)/10}={np.quantile(r_all,q):.2f}" for q in qs),
             f"- Gaussian-source ‖x‖ quantiles " + ", ".join(f"q{int(q*1000)/10}={np.quantile(rg,q):.2f}" for q in qs),
             f"- data norm mean={r_all.mean():.3f} vs Gaussian mean={rg.mean():.3f} (mismatch={abs(r_all.mean()-rg.mean()):.3f})",
             f"- mean marginal excess kurtosis (per-dim) = {float(np.mean(kurtosis(Z, axis=0))):.2f} (0=Gaussian; >0 heavy-tailed)",
             f"- radial excess kurtosis = {float(kurtosis(r_all)):.2f}\n",
             "## Anisotropy",
             f"- covariance eigenvalue spectrum (top5): {np.round(eig[:5],3).tolist()}",
             f"- condition number (λmax/λmin) = {aniso:.1f}\n",
             "## Regime split (test)",
             f"- high-vol threshold (‖x‖ q90 of test) = {thr:.2f}; high-vol frac by construction ~10%.",
             f"- ordinary-period mean ‖x‖ = {te_norms[te_norms<=thr].mean():.3f}; high-vol mean ‖x‖ = {te_norms[te_norms>thr].mean():.3f}",
             "\n## Verdict",
             "Real, heavy-tailed (excess kurtosis > 0), anisotropic (condition number >> 1), d=49 — usable; norm mismatch vs Gaussian source is the quantity RAFM targets.",
             ]
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "finance_quick_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved processed tensor + meta + quick check.")


if __name__ == "__main__":
    main()
