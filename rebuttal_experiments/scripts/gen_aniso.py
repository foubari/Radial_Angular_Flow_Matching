"""Generate controlled anisotropy-sweep datasets (E9) as .pt files for run_real.py.

X = z @ A^T with z ~ N(0, I_d) (LIGHT-tailed base, so anisotropy is the ONLY varying
factor), and A = U diag(s) V^T with singular values s logspaced from 1 to kappa
(condition number = kappa). d=32, N=50000. Saved standardized? No — kept raw (the
radial law is exactly chi scaled by A's spectrum). run_real does chronological split of
iid data (order already random), no leakage concern.
"""
from pathlib import Path
import numpy as np, torch, json

OUT = Path("rebuttal_experiments/data/aniso"); OUT.mkdir(parents=True, exist_ok=True)
D = 32; N = 50000
KAPPAS = [1, 3, 10, 30, 100, 300]


def main():
    rng = np.random.default_rng(42)
    # fixed random orthonormal U, V (shared across kappa so only the spectrum changes)
    U, _ = np.linalg.qr(rng.standard_normal((D, D)))
    V, _ = np.linalg.qr(rng.standard_normal((D, D)))
    for kappa in KAPPAS:
        s = np.logspace(0, np.log10(kappa), D)  # condition number = kappa
        A = U @ np.diag(s) @ V.T
        z = rng.standard_normal((N, D))
        X = z @ A.T
        torch.save(torch.tensor(X, dtype=torch.float32), OUT / f"aniso_gauss_d32_k{kappa}.pt")
        cov = np.cov(X.T); eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        print(f"kappa={kappa:4d} saved; empirical cond={eig[0]/eig[-1]:.1f}")
    (OUT / "meta.json").write_text(json.dumps(
        {"base": "gaussian (light-tailed)", "dim": D, "N": N, "kappas": KAPPAS,
         "A": "U diag(logspace(1,kappa)) V^T, fixed U,V (seed 42)"}, indent=2))


if __name__ == "__main__":
    main()
