"""Aggregate a scaling experiment (E6 dim, E8 tail) whose results live under
<exp_root>/<dataset_dir>/<method>/seed_*/metrics.json, where <dataset_dir> encodes
the swept parameter (e.g. student_t_d64_df3.0_cor -> dim=64; student_t_d16_df1.5_cor -> df=1.5).

Writes a long CSV: param, method, metric, mean, std, n_seeds  and a compact markdown
pivot of a chosen metric (default radial_w1 and sliced_w1) vs param per method.

Usage:
  python rebuttal_experiments/scripts/aggregate_scaling.py --exp_root rebuttal_experiments/raw_results/E6_dim_scaling --param dim --out rebuttal_experiments/tables/E6_dim_scaling
  python rebuttal_experiments/scripts/aggregate_scaling.py --exp_root rebuttal_experiments/raw_results/E8_tail_scaling --param df --out rebuttal_experiments/tables/E8_tail_scaling
"""
import argparse, csv, json, re, math
from pathlib import Path
import statistics as st

PIVOT_METRICS = ["radial_w1", "sliced_w1", "angular_sw_mean", "nan_rate", "total_train_time_s"]


def parse_param(dirname, param):
    if param == "dim":
        m = re.search(r"_d(\d+)_", dirname)
    elif param == "kappa":
        m = re.search(r"_k(\d+)$", dirname) or re.search(r"_k(\d+)", dirname)
    else:  # df
        m = re.search(r"_df([0-9.]+)", dirname)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--param", choices=["dim", "df", "kappa"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.exp_root)

    rows = []  # (param, method, metric, mean, std, n)
    pivot = {}  # (metric) -> {method -> {param -> mean}}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        param = parse_param(ds_dir.name, args.param)
        if param is None:
            continue
        for mdir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            vals = {}
            for sd in sorted(mdir.glob("seed_*")):
                mp = sd / "metrics.json"
                if not mp.exists():
                    continue
                m = json.loads(mp.read_text())
                for k, v in m.items():
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        vals.setdefault(k, []).append(float(v))
            for k, arr in vals.items():
                mean = sum(arr) / len(arr)
                sd_ = st.pstdev(arr) if len(arr) > 1 else 0.0
                rows.append((param, mdir.name, k, mean, sd_, len(arr)))
                if k in PIVOT_METRICS:
                    pivot.setdefault(k, {}).setdefault(mdir.name, {})[param] = mean

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([args.param, "method", "metric", "mean", "std", "n_seeds"])
        for r in sorted(rows):
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6g}", f"{r[4]:.6g}", r[5]])

    # markdown pivots
    lines = [f"# Scaling ({args.param}) — {root.name}\n"]
    params = sorted({r[0] for r in rows})
    for metric in PIVOT_METRICS:
        if metric not in pivot:
            continue
        lines.append(f"\n## {metric} vs {args.param}\n")
        lines.append("| method | " + " | ".join(f"{args.param}={int(p) if p==int(p) else p}" for p in params) + " |")
        lines.append("|" + "---|" * (len(params) + 1))
        for method in sorted(pivot[metric]):
            cells = [method]
            for p in params:
                v = pivot[metric][method].get(p)
                cells.append(f"{v:.4g}" if v is not None else "-")
            lines.append("| " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
