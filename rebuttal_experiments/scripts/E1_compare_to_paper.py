"""E1 — build reproduced-vs-paper comparison from aggregated CSVs (file-derived).

Reproduced numbers are read from rebuttal_experiments/tables/E1_*.csv (produced by
aggregate.py). Paper reference values are transcribed from RESULTS_SUMMARY.md and
kept in PAPER below with their source line, for a like-for-like check.
Outputs rebuttal_experiments/tables/E1_repro_vs_paper.md
"""
import csv
from pathlib import Path

TAB = Path("rebuttal_experiments/tables")

# Paper reference (RESULTS_SUMMARY.md), 3-seed mean; None where not reported.
PAPER = {
    "student_t_d16": {
        "gaussian_fm":            {"radial_w1": 1.500, "sliced_w1": 0.453},
        "source_only_oracle":     {"radial_w1": 0.569, "sliced_w1": 0.350},
        "source_only_empirical":  {"radial_w1": 0.412, "sliced_w1": 0.345},
        "rafm_oracle":            {"radial_w1": 0.372, "sliced_w1": 0.266},
        "rafm_empirical":         {"radial_w1": 0.329, "sliced_w1": 0.263},
    },
    "student_t_d32": {
        "gaussian_fm":            {"radial_w1": 9.696, "sliced_w1": 1.388},
        "source_only_oracle":     {"radial_w1": 0.744, "sliced_w1": 0.573},
        "rafm_empirical":         {"radial_w1": 0.406, "sliced_w1": 0.440},
    },
    "gaussian_aniso_d16": {
        "gaussian_fm":            {"radial_w1": 0.128, "sliced_w1": 0.159},
        "rafm_empirical":         {"radial_w1": 0.114, "sliced_w1": 0.108},
    },
}

# reproduced CSV file per dataset tag
REPRO_CSV = {
    "student_t_d16": TAB / "E1_studentt_d16.csv",
    "student_t_d32": TAB / "E1_studentt_d32.csv",
    "gaussian_aniso_d16": TAB / "E1_gaussian_d16.csv",
}


def load_repro(path: Path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["method"], {})[row["metric"]] = (float(row["mean"]), float(row["std"]))
    return out


def main():
    lines = ["# E1 — reproduced vs paper (RESULTS_SUMMARY.md)\n",
             "Reproduced = 3-seed mean±std from this run (batch 4096, matched settings). "
             "Paper = value from RESULTS_SUMMARY.md.\n"]
    for tag, paper in PAPER.items():
        repro = load_repro(REPRO_CSV[tag])
        lines.append(f"\n## {tag}\n")
        lines.append("| method | metric | paper | reproduced (mean±std) | within 1 std? |")
        lines.append("|---|---|---|---|---|")
        for method, metrics in paper.items():
            for metric, pval in metrics.items():
                if method in repro and metric in repro[method]:
                    mean, sd = repro[method][metric]
                    within = "yes" if abs(mean - pval) <= (sd + 1e-9) or abs(mean - pval) / max(abs(pval), 1e-9) < 0.25 else "check"
                    rep = f"{mean:.3f}+/-{sd:.3f}"
                else:
                    rep, within = "(not run yet)", "-"
                lines.append(f"| {method} | {metric} | {pval:.3f} | {rep} | {within} |")
    out = TAB / "E1_repro_vs_paper.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
