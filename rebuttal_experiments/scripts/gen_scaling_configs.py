"""Generate self-contained configs for scaling experiments E6 (dimension), E8 (tail df),
E9 (anisotropy via df fixed, dims fixed but n_samples) — writes to rebuttal_experiments/configs/genN.

Methods kept: gaussian_fm, source_only_empirical, rafm_empirical, rafm_oracle
(MSGM excluded from scaling sweeps — ~50 min/run makes it infeasible at this breadth;
run separately for the main benchmark only).
"""
from pathlib import Path

BASE = dict(
    hidden_dim=128, n_layers=3, premodule="null",
    lr=0.001, batch_size=4096, n_train_steps=10000, log_every=200, ckpt_every=5000,
    solver="rk4", nfe=128, n_gen_samples=10000,
    n_seeds=3, base_seed=42, device="auto",
    split_seed=0, train_frac=0.6, val_frac=0.2,
    n_projections_sw=500, n_projections_angular=200, n_angular_bins=4,
    norm_exploding_factor=100,
)
METHODS = """methods:
  - name: gaussian_fm
    source: gaussian
    path: euclidean
  - name: source_only_empirical
    source: radial_empirical_ecdf
    path: euclidean
  - name: rafm_oracle
    source: radial_oracle
    path: spherical_geodesic
  - name: rafm_empirical
    source: radial_empirical_ecdf
    path: spherical_geodesic
"""

CFG_DIR = Path("rebuttal_experiments/configs")


def header(experiment, output_dir):
    lines = [f"experiment: {experiment}", f"output_dir: {output_dir}"]
    for k, v in BASE.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def dataset_block(name, dim, df=None, n_samples=50000):
    b = ["dataset:", f"  name: {name}", f"  dim: {dim}"]
    if df is not None:
        b.append(f"  df: {df}")
    b.append(f"  n_samples: {n_samples}")
    b.append("  correlated: true")
    return "\n".join(b) + "\n"


def write(path, experiment, output_dir, ds):
    txt = header(experiment, output_dir) + ds + METHODS
    path.write_text(txt, encoding="utf-8")
    print("wrote", path)


def main():
    # E6 dimension scaling — Student-t df=3, dims
    for d in [2, 8, 16, 32, 64, 128, 256]:
        write(CFG_DIR / f"E6_dim{d}.yaml", "E6_dim_scaling",
              "rebuttal_experiments/raw_results",
              dataset_block("student_t", d, df=3.0))
    # E8 tail-heaviness — Student-t d=16, varying df (heavy->light)
    for df in [1.5, 2.0, 3.0, 5.0, 10.0, 50.0]:
        write(CFG_DIR / f"E8_df{df}.yaml", "E8_tail_scaling",
              "rebuttal_experiments/raw_results",
              dataset_block("student_t", 16, df=df))


if __name__ == "__main__":
    main()
