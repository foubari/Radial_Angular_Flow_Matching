#!/usr/bin/env python3
"""Theory-to-experiment diagnostics for the scale-free angular target.

Central claim to support empirically (NOT 'the angular norm is constant / equal to pi/2'):
  full-velocity target ‖u_t‖ grows with radial magnitude ‖x_t‖, whereas the
  angular target A = u_t/‖x_t‖ is scale-free w.r.t. the radius and bounded by pi
  along the spherical path. On the coupled sphere ‖x_t‖=R, so ‖u_t‖/‖x_t‖ equals the
  geodesic angle theta = angle(u0,u1) ∈ [0, pi]. In HIGH DIMENSION random directions are
  near-orthogonal, so theta CONCENTRATES near pi/2 — an empirical high-D phenomenon here,
  not a universal value (in low d it spreads across [0, pi]).

Pure geometry: needs only (x0, x1, path, t) — no trained model. x0 is the radial
coupling R·u0 (R=‖x1‖), path = spherical geodesic (same as RAFM/Angular training).
Reconstructed from data alone.

Settings: heavy-tailed Student-t d16 (df3) and AudioMNIST STFT (energy=radius, heavy gain).
Outputs: fig_theory_scalefree.{pdf,png}, theory_stats.json, table_theory.tex
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "rebuttal_experiments"))
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.utils.sphere import uniform_on_sphere

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18,
})
C_FULL, C_ANG = "#0072B2", "#E76F51"
N_PAIRS = 40000


def sample_quantities(x1, device, n=N_PAIRS, seed=0, chunk=2000):
    """Return radius ‖x_t‖, full-velocity norm ‖u_t‖, angular norm ‖u_t‖/‖x_t‖ for n random (x0,x1,t).

    Chunked to bound memory for high-D data (e.g. AudioMNIST D=16254)."""
    torch.manual_seed(seed)
    N, D = x1.shape
    path = SphericalGeodesicPath()
    rt_a, un_a, ang_a = [], [], []
    for s in range(0, n, chunk):
        b = min(chunk, n - s)
        idx = torch.randint(N, (b,))
        xb = x1[idx].to(device)
        R = xb.norm(dim=1, keepdim=True)
        u0 = uniform_on_sphere(b, D, device=device)
        x0 = R * u0
        t = torch.rand(b, device=device)
        xt = path.sample_path(x0, xb, t)
        ut = path.conditional_vector_field(x0, xb, t)
        rt = xt.norm(dim=1).clamp(min=1e-8)
        un = ut.norm(dim=1)
        rt_a.append(rt.cpu().numpy()); un_a.append(un.cpu().numpy()); ang_a.append((un / rt).cpu().numpy())
        del xb, R, u0, x0, t, xt, ut, rt, un
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(rt_a), np.concatenate(un_a), np.concatenate(ang_a)


def stats(name, r, un, ang):
    def cov(a): return float(np.std(a) / (np.mean(a) + 1e-12))
    def qs(a): return {q: float(np.quantile(a, q)) for q in (0.5, 0.9, 0.99)}
    return {
        "dataset": name, "n": len(r),
        "radius": {"mean": float(r.mean()), "median": float(np.median(r)), "max": float(r.max())},
        "full_velocity_norm": {"mean": float(un.mean()), "std": float(un.std()), "CoV": cov(un), "quantiles": qs(un)},
        "angular_norm": {"mean": float(ang.mean()), "std": float(ang.std()), "CoV": cov(ang), "quantiles": qs(ang)},
        "corr_fullnorm_radius": float(np.corrcoef(un, r)[0, 1]),
        "corr_angularnorm_radius": float(np.corrcoef(ang, r)[0, 1]),
    }


def binned(ax, r, y, color, label):
    """Median + 10-90% band of y across radius bins (log-spaced)."""
    edges = np.quantile(r, np.linspace(0, 1, 13))
    edges = np.unique(edges)
    cx, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() < 20:
            continue
        cx.append(np.sqrt(a * b) if a > 0 else b / 2)
        med.append(np.median(y[m])); lo.append(np.quantile(y[m], .1)); hi.append(np.quantile(y[m], .9))
    cx, med, lo, hi = map(np.array, (cx, med, lo, hi))
    ax.plot(cx, med, color=color, marker="o", ms=3.5, label=label, zorder=4)
    ax.fill_between(cx, lo, hi, color=color, alpha=0.15, zorder=2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- heavy-tail synthetic ---
    import yaml
    from experiments.exp0_source_diagnostics import DATASETS_FACTORY
    y = yaml.safe_load(open(REPO / "rebuttal_experiments/configs/E1_studentt_d16.yaml"))
    ds = DATASETS_FACTORY[y["dataset"]["name"]](y["dataset"])
    x1_syn = ds.get_train_data()
    # --- audiomnist ---
    d = torch.load(REPO / "experiments/poc_audio/data/audiomnist_stft_train.pt", map_location="cpu")
    x1_aud = d["x"].reshape(d["x"].shape[0], -1).float()

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0))
    for row, (name, x1, ttl) in enumerate([
            ("student_t_d16_df3", x1_syn, r"Student-$t$ $d{=}16$ (heavy tail)"),
            ("audiomnist_stft", x1_aud, "AudioMNIST STFT (energy=radius)")]):
        r, un, ang = sample_quantities(x1, device)
        results[name] = stats(name, r, un, ang)
        # left: full-velocity norm vs radius (grows)
        binned(axes[row][0], r, un, C_FULL, r"full velocity $\|\dot X_t\|$")
        axes[row][0].set_ylabel(ttl + "\n" + r"$\|\dot X_t\|$", fontsize=8)
        axes[row][0].set_xscale("log"); axes[row][0].set_yscale("log")
        rr = results[name]["corr_fullnorm_radius"]
        axes[row][0].set_title(f"full-velocity target — corr w/ radius = {rr:.2f}", fontsize=8)
        # right: angular norm vs radius (flat / bounded)
        binned(axes[row][1], r, ang, C_ANG, r"angular $\|\dot X_t\|/\|X_t\|$")
        axes[row][1].set_xscale("log")
        ra = results[name]["corr_angularnorm_radius"]
        axes[row][1].set_title(f"angular target — corr w/ radius = {ra:.2f}", fontsize=8)
        axes[row][1].set_ylabel(r"$\|\dot X_t\|/\|X_t\|=\theta$", fontsize=8)
        # theta = geodesic angle in [0, pi]; pi/2 = high-D concentration (not universal)
        axes[row][1].axhline(np.pi / 2, color="#666", ls="--", lw=0.8)
        axes[row][1].annotate(r"$\pi/2$ (high-$d$ concentration)", xy=(axes[row][1].get_xlim()[0], np.pi / 2),
                              fontsize=6.5, color="#666", va="bottom")
        for c in (0, 1):
            axes[row][c].set_xlabel(r"radius $\|X_t\|$")
    axes[0][0].legend(); axes[0][1].legend()
    fig.suptitle(r"Full-velocity target scales with radius; angular target $=\theta\in[0,\pi]$ is "
                 r"scale-free (concentrates near $\pi/2$ in high $d$)",
                 y=1.01, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "fig_theory_scalefree.pdf"); fig.savefig(HERE / "fig_theory_scalefree.png", dpi=300)
    plt.close(fig)

    (HERE / "theory_stats.json").write_text(json.dumps(results, indent=2))

    # LaTeX table
    tex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Dataset & Target & Mean & Std & CoV & corr(radius) & $q_{99}$ \\", r"\midrule"]
    disp = {"student_t_d16_df3": r"Student-$t$ $d{=}16$", "audiomnist_stft": "AudioMNIST"}
    for name, s in results.items():
        fv, an = s["full_velocity_norm"], s["angular_norm"]
        tex.append(f"{disp[name]} & full-vel $\\|\\dot X_t\\|$ & {fv['mean']:.3f} & {fv['std']:.3f} & "
                   f"{fv['CoV']:.2f} & {s['corr_fullnorm_radius']:.2f} & {fv['quantiles'][0.99]:.3f} \\\\")
        tex.append(f" & angular $\\|\\dot X_t\\|/\\|X_t\\|$ & {an['mean']:.3f} & {an['std']:.3f} & "
                   f"{an['CoV']:.2f} & {s['corr_angularnorm_radius']:.2f} & {an['quantiles'][0.99]:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "rebuttal_experiments/paper_assets/tables/table_theory.tex").write_text("\n".join(tex))

    print("wrote fig_theory_scalefree.{pdf,png}, theory_stats.json, table_theory.tex")
    for name, s in results.items():
        print(f"  {name}: full-vel CoV={s['full_velocity_norm']['CoV']:.2f} corr={s['corr_fullnorm_radius']:.2f} | "
              f"angular CoV={s['angular_norm']['CoV']:.2f} corr={s['corr_angularnorm_radius']:.2f}")


if __name__ == "__main__":
    main()
