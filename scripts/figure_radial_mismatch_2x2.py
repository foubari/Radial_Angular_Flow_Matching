"""Figure: 2×2 Radial mismatch & generated fidelity.

Top row: source radial laws (survival function, log-scale)
Bottom row: generated radial distributions (survival function, log-scale)
Columns: Student-t d=32, PIV d=256

Usage:
    python scripts/figure_radial_mismatch_2x2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch

from rafm.data.student_t import StudentT
from rafm.data.piv import PIVDataset
from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource


def survival(norms: np.ndarray, n_points: int = 500):
    """Return (r, 1-F(r)) for smooth survival curve."""
    s = np.sort(norms)
    sf = 1.0 - np.arange(1, len(s) + 1) / len(s)
    # Subsample for clean plotting
    idx = np.linspace(0, len(s) - 1, n_points).astype(int)
    return s[idx], sf[idx]


# ── Data loaders ──

def load_student_t_d32():
    ds = StudentT(dim=32, df=3.0, correlated=True)
    return ds.get_train_data(), ds.get_test_data(), 32

def load_piv_d256():
    ds = PIVDataset(data_root="data/piv", dim=256)
    return ds.get_train_data(), ds.get_test_data(), 256


def load_generated_samples(dataset_name: str, method: str):
    """Load all seed samples and concatenate."""
    base = Path(f"outputs/exp1_main_benchmark/{dataset_name}/{method}")
    if not base.exists():
        return None
    all_samples = []
    for seed_dir in sorted(base.iterdir()):
        sp = seed_dir / "samples.pt"
        if sp.exists():
            all_samples.append(torch.load(sp, map_location='cpu'))
    if not all_samples:
        return None
    return torch.cat(all_samples, dim=0)


# ── Colors ──

COLORS = {
    "test":        ("Test data",    "black",   "-",  2.0),
    "gaussian":    ("Gaussian $\\mathcal{N}(0,I)$", "#228B22", "--", 1.5),
    "empirical":   ("Empirical source", "#1f5faa", "-.", 1.5),
    "gaussian_fm": ("Gaussian FM",  "#228B22", "-",  1.5),
    "source_only": ("Source-only",  "#999999", "-",  1.5),
    "rafm":        ("RAFM",         "#1f5faa", "-",  1.8),
    "msgm":        ("MSGM",         "#e67e22", "-",  1.5),
}


def plot_survival(ax, norms, key, zorder=2):
    label, color, ls, lw = COLORS[key]
    r, sf = survival(norms)
    # Clip zeros for log scale
    mask = sf > 0
    ax.plot(r[mask], sf[mask], color=color, linestyle=ls, linewidth=lw,
            label=label, zorder=zorder)


def style_ax(ax, xlabel=True, ylabel=True, title=None):
    ax.set_yscale('log')
    ax.grid(True, alpha=0.15, which='both')
    if xlabel:
        ax.set_xlabel(r"$\|x\|$", fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel("Tail probability", fontsize=14, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=10)


def make_figure(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
    })

    # Load datasets
    train_st, test_st, dim_st = load_student_t_d32()
    train_piv, test_piv, dim_piv = load_piv_d256()

    R_test_st = torch.norm(test_st, dim=-1).numpy()
    R_test_piv = torch.norm(test_piv, dim=-1).numpy()
    R_train_st = torch.norm(train_st, dim=-1).numpy()
    R_train_piv = torch.norm(train_piv, dim=-1).numpy()

    # Gaussian source norms
    gauss = GaussianSource()
    n_source = 50000
    R_gauss_st = torch.norm(gauss.sample(n_source, dim_st), dim=-1).numpy()
    R_gauss_piv = torch.norm(gauss.sample(n_source, dim_piv), dim=-1).numpy()

    # Empirical source norms
    emp_st = RadialEmpiricalSource(mode="ecdf").fit(train_st)
    R_emp_st = torch.norm(emp_st.sample(n_source, dim_st), dim=-1).numpy()
    emp_piv = RadialEmpiricalSource(mode="ecdf").fit(train_piv)
    R_emp_piv = torch.norm(emp_piv.sample(n_source, dim_piv), dim=-1).numpy()

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # ═══ TOP ROW: Source mismatch ═══

    # Top-left: Student-t d=32
    ax = axes[0, 0]
    plot_survival(ax, R_test_st, "test", zorder=3)
    plot_survival(ax, R_gauss_st, "gaussian")
    plot_survival(ax, R_emp_st, "empirical")
    style_ax(ax, xlabel=False, title=r"Student-$t$, $d=32$")
    ax.legend(fontsize=10, framealpha=0.8)

    # Top-right: PIV d=256
    ax = axes[0, 1]
    plot_survival(ax, R_test_piv, "test", zorder=3)
    plot_survival(ax, R_gauss_piv, "gaussian")
    plot_survival(ax, R_emp_piv, "empirical")
    style_ax(ax, xlabel=False, ylabel=False, title=r"PIV, $d=256$")
    ax.legend(fontsize=10, framealpha=0.8)

    # ═══ BOTTOM ROW: Generated radial fidelity ═══

    # Bottom-left: Student-t d=32
    ax = axes[1, 0]
    plot_survival(ax, R_test_st, "test", zorder=3)

    methods_st = [
        ("gaussian_fm", "gaussian_fm"),
        ("source_only_empirical", "source_only"),
        ("rafm_empirical", "rafm"),
        ("msgm", "msgm"),
    ]
    for method_dir, color_key in methods_st:
        samples = load_generated_samples("student_t_d32_df3.0_cor", method_dir)
        if samples is not None:
            R_gen = torch.norm(samples, dim=-1).numpy()
            plot_survival(ax, R_gen, color_key)

    style_ax(ax, title=r"Student-$t$, $d=32$ — generated")
    ax.legend(fontsize=9, framealpha=0.8)

    # Bottom-right: PIV d=256
    ax = axes[1, 1]
    plot_survival(ax, R_test_piv, "test", zorder=3)

    methods_piv = [
        ("gaussian_fm", "gaussian_fm"),
        ("source_only_empirical", "source_only"),
        ("rafm_empirical", "rafm"),
        ("msgm", "msgm"),
    ]
    for method_dir, color_key in methods_piv:
        samples = load_generated_samples("piv_d256", method_dir)
        if samples is not None:
            R_gen = torch.norm(samples, dim=-1).numpy()
            plot_survival(ax, R_gen, color_key)

    style_ax(ax, ylabel=False, title=r"PIV, $d=256$ — generated")
    ax.legend(fontsize=9, framealpha=0.8)

    # ── Row labels ──
    fig.text(0.02, 0.73, "Source\nmismatch", fontsize=13, fontweight='bold',
             ha='center', va='center', rotation=90, color='#555555')
    fig.text(0.02, 0.28, "Generated\nfidelity", fontsize=13, fontweight='bold',
             ha='center', va='center', rotation=90, color='#555555')

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    fig.subplots_adjust(hspace=0.25, wspace=0.18)

    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_radial_mismatch_2x2.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"2x2 radial mismatch figure saved to {out_dir}/figure_radial_mismatch_2x2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
