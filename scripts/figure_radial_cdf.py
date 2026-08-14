"""Figure: Radial CDF comparison (Exp 0).

Overlays radial CDFs of Gaussian source, oracle source, empirical source,
and test data for each dataset. Shows the source mismatch visually.

Usage:
    python scripts/figure_radial_cdf.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch

from rafm.data.student_t import StudentT
from rafm.data.gaussian_aniso import GaussianAniso
from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource


def radial_cdf(norms: np.ndarray):
    """Return sorted norms and their empirical CDF values."""
    s = np.sort(norms)
    cdf = np.arange(1, len(s) + 1) / len(s)
    return s, cdf


def make_figure(out_dir: str = "figures/paper_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
    })

    datasets = [
        ("Student-$t$, $d=16$", StudentT(dim=16, df=3.0, correlated=True)),
        ("Student-$t$, $d=32$", StudentT(dim=32, df=3.0, correlated=True)),
        ("Gaussian, $d=16$", GaussianAniso(dim=16, correlated=True)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (title, ds) in zip(axes, datasets):
        test_data = ds.get_test_data()
        train_data = ds.get_train_data()
        dim = ds.dim

        # Test radial CDF
        R_test = torch.norm(test_data, dim=-1).numpy()
        s_test, cdf_test = radial_cdf(R_test)

        # Gaussian source
        gauss = GaussianSource()
        x_gauss = gauss.sample(len(test_data), dim)
        R_gauss = torch.norm(x_gauss, dim=-1).numpy()
        s_gauss, cdf_gauss = radial_cdf(R_gauss)

        # Empirical source
        emp = RadialEmpiricalSource(mode="ecdf").fit(train_data)
        x_emp = emp.sample(len(test_data), dim)
        R_emp = torch.norm(x_emp, dim=-1).numpy()
        s_emp, cdf_emp = radial_cdf(R_emp)

        # Plot
        ax.plot(s_test, cdf_test, color='black', linewidth=2.0,
                label='Test data', zorder=3)
        ax.plot(s_gauss, cdf_gauss, color='#228B22', linewidth=1.5,
                linestyle='--', label='Gaussian $\\mathcal{N}(0,I)$', zorder=2)
        ax.plot(s_emp, cdf_emp, color='#1f5faa', linewidth=1.5,
                linestyle='-.', label='Empirical eCDF', zorder=2)

        ax.set_xlabel(r"$\|x\|$", fontsize=14, fontweight='bold')
        ax.set_ylabel("CDF", fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(out_dir / f"figure_radial_cdf.{ext}"),
                    bbox_inches='tight', dpi=300)
    print(f"Radial CDF figure saved to {out_dir}/figure_radial_cdf.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
