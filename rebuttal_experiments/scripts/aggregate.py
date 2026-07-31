"""Aggregate exp1-style metrics.json across seeds into mean/std tables.

Walks <results_root>/<method>/seed_*/metrics.json, groups by method, writes:
  - <out>.csv   : long form (method, metric, mean, std, n_seeds)
  - <out>.md    : compact markdown table of headline metrics

Usage:
  python rebuttal_experiments/scripts/aggregate.py --results_root <dir> --out <path_without_ext>
"""
import argparse, json, csv, math
from pathlib import Path
import statistics as st

HEADLINE = ["radial_w1", "ks_stat", "sliced_w1", "mmd", "angular_sw_mean",
            "q990_err", "q995_err", "tail_exc_99", "nan_rate",
            "total_train_time_s", "sample_time_s"]


def collect(root: Path):
    data = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        vals = {}
        for sd in sorted(mdir.glob("seed_*")):
            mp = sd / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            for k, v in m.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    vals.setdefault(k, []).append(float(v))
                elif isinstance(v, (int, float)):
                    vals.setdefault(k + "__nan_count", []).append(1.0)
        if vals:
            data[mdir.name] = vals
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.results_root)
    data = collect(root)

    # long-form CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "mean", "std", "n_seeds"])
        for method, vals in data.items():
            for k, arr in sorted(vals.items()):
                mean = sum(arr) / len(arr)
                sd = st.pstdev(arr) if len(arr) > 1 else 0.0
                w.writerow([method, k, f"{mean:.6g}", f"{sd:.6g}", len(arr)])

    # markdown headline table
    lines = ["| method | " + " | ".join(HEADLINE) + " |",
             "|" + "---|" * (len(HEADLINE) + 1)]
    for method, vals in data.items():
        cells = [method]
        for k in HEADLINE:
            if k in vals:
                arr = vals[k]
                mean = sum(arr) / len(arr)
                sd = st.pstdev(arr) if len(arr) > 1 else 0.0
                cells.append(f"{mean:.4g}+/-{sd:.2g}" if sd else f"{mean:.4g}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
