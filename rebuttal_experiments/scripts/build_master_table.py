"""Build one consolidated results table across all benchmark-style experiments x all methods.

For each dataset/experiment (a dir containing <method>/seed_*/metrics.json), compute the
per-method mean (±pstd) of the headline metrics. Emits tables/MASTER_results.md.
Scaling sweeps (dim/tail/aniso/sample-size) are parameter curves — pointed to, not flattened.
"""
import json, statistics as st, math
from pathlib import Path

ROOT = Path("C:/Users/Shadow/Desktop/Radial_Angular_FM")
# (label, results_root)  — benchmark-style (method/seed_*/metrics.json)
DATASETS = [
    ("Student-t d16 (repro, b4096)", "rebuttal_experiments/raw_results/E1_reproduction/student_t_d16_df3.0_cor"),
    ("Student-t d32 (repro, b4096)", "rebuttal_experiments/raw_results/E1_reproduction/student_t_d32_df3.0_cor"),
    ("Gaussian-aniso d16 (control)", "rebuttal_experiments/raw_results/E1_reproduction/gaussian_aniso_d16_cor"),
    ("Toy2D d2 (existing)",          "outputs/exp1_main_benchmark/toy_radial_angular"),
    ("PIV d64 (real)",              "rebuttal_experiments/raw_results/E10_piv/piv_d64"),
    ("Finance d49 (real, chrono)",  "rebuttal_experiments/raw_results/E_finance/finance_ff49"),
    ("Finance d49 (real, random)",  "rebuttal_experiments/raw_results/E_finance_randomsplit/finance_ff49"),
    ("Weather d96 (real, MLP3x128)","rebuttal_experiments/raw_results/E_weather/weather_au_wind"),
    ("Weather d96 (real, resMLP)",  "rebuttal_experiments/raw_results/E_weather_bigmodel/weather_au_wind"),
]
METHOD_ORDER = ["gaussian_fm", "msgm", "source_only_oracle", "source_only_empirical",
                "rafm_oracle", "rafm_empirical", "rafm_empirical_no_proj"]
METRICS = ["radial_w1", "sliced_w1", "ks_stat", "total_train_time_s"]


def agg_method(mdir: Path):
    out = {}
    for k in METRICS:
        vals = []
        for sd in mdir.glob("seed_*"):
            mp = sd / "metrics.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                if k in m and isinstance(m[k], (int, float)) and math.isfinite(m[k]):
                    vals.append(float(m[k]))
        if vals:
            out[k] = (sum(vals)/len(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
    return out


def fmt(t):
    if t is None:
        return "—"
    mean, sd, n = t
    if mean >= 1000:
        return f"{mean:.0f}"
    return f"{mean:.3f}±{sd:.3f}" if sd else f"{mean:.3f}"


def main():
    lines = ["# MASTER results — all benchmark experiments × all methods\n",
             "Per-method mean±pstd over seeds. radial_w1 = radial Wasserstein-1; sliced_w1 = global sliced-W1; "
             "ks = radial KS; train_s = training seconds/seed. Lower is better (except train_s = cost). "
             "Scaling sweeps (dim, tail, anisotropy, sample-size) are parameter curves — see "
             "`tables/E6_dim_scaling.md`, `E8_tail_scaling.md`, `E9_aniso.md`, `E7_sample_efficiency.md`.\n"]
    lines.append("| dataset / setting | method | radial_w1 | sliced_w1 | ks | train_s |")
    lines.append("|---|---|---|---|---|---|")
    for label, root in DATASETS:
        rp = ROOT / root
        if not rp.exists():
            continue
        methods = {m.name: agg_method(m) for m in rp.iterdir() if m.is_dir()}
        ordered = [m for m in METHOD_ORDER if m in methods] + \
                  [m for m in methods if m not in METHOD_ORDER]
        first = True
        for meth in ordered:
            a = methods[meth]
            if not a:
                continue
            ds_cell = label if first else ""
            first = False
            lines.append(f"| {ds_cell} | {meth} | {fmt(a.get('radial_w1'))} | "
                         f"{fmt(a.get('sliced_w1'))} | {fmt(a.get('ks_stat'))} | "
                         f"{fmt(a.get('total_train_time_s'))} |")
    out = ROOT / "rebuttal_experiments/tables/MASTER_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
