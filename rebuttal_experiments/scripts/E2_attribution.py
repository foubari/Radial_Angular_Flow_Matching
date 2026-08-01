"""E2 — component-ablation attribution (addresses reviewer R2).

Decomposes the gain along the ladder, using the aggregated E1 CSV(s):
  Gaussian FM  --(source correction)-->  source-only (empirical)  --(spherical path)-->  RAFM (empirical)

Reports, per metric, the delta from each step (lower-is-better metrics: negative delta = improvement),
so reviewers see how much of the radial vs angular/global improvement each component buys.
Uncertainty on a delta uses quadrature of the two per-method stds.

Usage:
  python rebuttal_experiments/scripts/E2_attribution.py --agg rebuttal_experiments/tables/E1_studentt_d16.csv \
      --tag student_t_d16 --out rebuttal_experiments/tables/E2_attribution_student_t_d16.md
"""
import argparse, csv, math
from pathlib import Path

LADDER = [("gaussian_fm", "source_only_empirical", "source correction (Gaussian->eCDF source)"),
          ("source_only_empirical", "rafm_empirical", "spherical path (Euclid->slerp)")]
METRICS = ["radial_w1", "ks_stat", "sliced_w1", "mmd", "angular_sw_mean", "q995_err", "tail_exc_99"]


def load(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d.setdefault(row["method"], {})[row["metric"]] = (float(row["mean"]), float(row["std"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = load(args.agg)

    lines = [f"# E2 attribution — {args.tag}",
             "",
             "Lower is better for all metrics. Delta<0 = improvement from that component. "
             "Uncertainty = quadrature of per-method stds (3 seeds).",
             ""]
    # absolute table
    methods = ["gaussian_fm", "source_only_empirical", "rafm_empirical"]
    lines.append("## Absolute (mean±std)")
    lines.append("| metric | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for m in METRICS:
        cells = [m]
        for meth in methods:
            if meth in d and m in d[meth]:
                mu, sd = d[meth][m]; cells.append(f"{mu:.4g}+/-{sd:.2g}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Decomposition (delta per component)")
    lines.append("| metric | " + " | ".join(desc for _, _, desc in LADDER) + " | total |")
    lines.append("|" + "---|" * (len(LADDER) + 2))
    for m in METRICS:
        cells = [m]
        total = 0.0
        ok = True
        for a, b, _ in LADDER:
            if a in d and b in d and m in d[a] and m in d[b]:
                mu_a, sd_a = d[a][m]; mu_b, sd_b = d[b][m]
                delta = mu_b - mu_a
                unc = math.hypot(sd_a, sd_b)
                total += delta
                cells.append(f"{delta:+.4g}+/-{unc:.2g}")
            else:
                cells.append("-"); ok = False
        cells.append(f"{total:+.4g}" if ok else "-")
        lines.append("| " + " | ".join(cells) + " |")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
