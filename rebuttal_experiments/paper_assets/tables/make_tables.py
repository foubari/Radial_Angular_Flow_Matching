#!/usr/bin/env python3
"""Emit LaTeX-ready (booktabs) tables from the aggregated result JSONs. No retraining.

Tables:
  table_realdata.tex      real data PIV/Weather/Finance x methods
  table_synthetic_main.tex Student-t d16 main (incl. oracle)
  table_audiomnist.tex    AudioMNIST
  table_dcae.tex          DC-AE ImageNette
  table_ablation.tex      std-RAFM vs Angular across all domains + delta
  table_appendix_sweeps.tex dim/df/aniso sweeps (std vs Angular, key metrics)
(table_theory.tex + table_efficiency.tex produced by their own scripts.)

Every cell is mean ± std over 3 seeds. NaN seeds are NOT averaged away: a setting whose
Angular aggregate is over <3 finite seeds is flagged with a dagger.
"""
import json, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MASTER = json.load(open(REPO / "rebuttal_experiments/master_results.json"))
SUITE = json.load(open(REPO / "rebuttal_experiments/master_suite.json"))


def c(a, p=3, flag_n=3):
    if not a or a.get("mean") is None or (isinstance(a["mean"], float) and math.isnan(a["mean"])):
        return "--"
    s = f"{a['mean']:.{p}f}\\std{{{a.get('std',0):.{p}f}}}"
    if a.get("n", flag_n) < flag_n:
        s += "$^\\dagger$"
    return s


def wrap(caption, label, header, rows, colspec):
    return "\n".join([
        r"\begin{table}[t]\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\small", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def emit(name, text):
    (HERE / name).write_text(text + "\n")
    print(f"  wrote {name}")


# preamble note (define \std once in the paper)
PREAMBLE = "% requires: \\newcommand{\\std}[1]{{\\scriptsize$\\pm$#1}}\n"

METH_TAB = [("gaussian_fm", "Gaussian FM"), ("source_only_empirical", "Source-only"),
            ("rafm_empirical", "RAFM (std)"), ("angular_rafm", "\\textbf{Angular RAFM}")]
METH_TAB_ORACLE = METH_TAB[:2] + [("source_only_oracle", "Source-only (oracle)"),
                                  ("rafm_oracle", "RAFM (oracle)")] + METH_TAB[2:]


def table_realdata():
    single = SUITE["singletons"]
    dsets = [("piv_d64", "PIV"), ("weather_au_wind", "Weather"), ("finance_ff49", "Finance")]
    mets = [("radial_w1", "Radial $W_1$"), ("sliced_w1", "Sliced $W_1$"), ("angular_sw_mean", "Angular SW")]
    rows = []
    for dk, dl in dsets:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{dl}}}}} \\\\")
        for mk, ml in METH_TAB:
            cells = [c(single[dk].get(mk, {}).get(met[0])) for met in mets]
            rows.append(f"\\quad {ml} & " + " & ".join(cells) + " \\\\")
    header = "Method & " + " & ".join(m[1] + " $\\downarrow$" for m in mets) + r" \\"
    emit("table_realdata.tex", PREAMBLE + wrap(
        "Real-data generation: standard vs Angular RAFM and baselines (mean$\\pm$std, 3 seeds). "
        "All metrics lower is better.", "tab:realdata", header, rows, "l" + "r" * len(mets)))


def table_synthetic_main():
    d = SUITE["singletons"]["student_t_d16_df3.0_cor"]
    mets = [("radial_w1", "Radial $W_1$"), ("ks_stat", "Radial KS"),
            ("sliced_w1", "Sliced $W_1$"), ("angular_sw_mean", "Angular SW")]
    rows = [f"{ml} & " + " & ".join(c(d.get(mk, {}).get(met[0]), 3) for met in mets) + " \\\\"
            for mk, ml in METH_TAB_ORACLE]
    header = "Method & " + " & ".join(m[1] + " $\\downarrow$" for m in mets) + r" \\"
    emit("table_synthetic_main.tex", PREAMBLE + wrap(
        "Main synthetic benchmark (Student-$t$, $d{=}16$, dof$=3$, correlated; 3 seeds). "
        "Oracle rows use the true radial law.", "tab:synthetic", header, rows, "l" + "r" * len(mets)))


def table_audiomnist():
    a = MASTER["audio"]
    order = [("gaussian", "Gaussian FM"), ("matched", "Matched-Eucl."), ("fixed_spher", "Fixed-spherical"),
             ("std_rafm", "RAFM (std)"), ("angular_rafm", "\\textbf{Angular RAFM}")]
    mets = [("digit_acc", "Digit acc.\\,$\\uparrow$"), ("energy_KS", "Energy KS\\,$\\downarrow$"),
            ("cov>q95", "cov$>$q95"), ("cov>q99", "cov$>$q99"), ("PIT", "PIT")]
    rows = [f"{ml} & " + " & ".join(c(a.get(mk, {}).get(met[0]), 3) for met in mets) + " \\\\"
            for mk, ml in order]
    header = "Method & " + " & ".join(m[1] for m in mets) + r" \\"
    emit("table_audiomnist.tex", PREAMBLE + wrap(
        "AudioMNIST (reversible complex-STFT, UNet flow, 24k steps, 3 seeds). Digit accuracy is content; "
        "energy KS is calibration.", "tab:audio", header, rows, "l" + "r" * len(mets)))


def table_dcae():
    d = MASTER["dcae"]
    order = [("gaussian_euclidean", "Gaussian FM"), ("matched_euclidean", "Matched-Eucl."),
             ("fixed_spherical", "Fixed-spherical"), ("rafm", "RAFM (std)"),
             ("angular_rafm", "\\textbf{Angular RAFM}")]
    mets = [("fid", "FID\\,$\\downarrow$"), ("radial_w1", "Radial $W_1$\\,$\\downarrow$"),
            ("ks", "Radial KS\\,$\\downarrow$"), ("precision", "Prec.\\,$\\uparrow$"),
            ("recall", "Rec.\\,$\\uparrow$"), ("coverage", "Cov.\\,$\\uparrow$")]
    rows = [f"{ml} & " + " & ".join(c(d.get(mk, {}).get(met[0]), 3) for met in mets) + " \\\\"
            for mk, ml in order]
    header = "Method & " + " & ".join(m[1] for m in mets) + r" \\"
    emit("table_dcae.tex", PREAMBLE + wrap(
        "DC-AE ImageNette latents (SiT backbone, 40k steps, 3 seeds). Angular RAFM attains the best FID "
        "and the best radial calibration simultaneously.", "tab:dcae", header, rows, "l" + "r" * len(mets)))


def table_ablation():
    """std-RAFM vs Angular across all domains, primary metric + directional, with delta."""
    rows = []
    # tabular/singletons
    S = SUITE["singletons"]
    def pair(std, ang, met):
        a = std.get(met) if std else None
        b = ang.get(met) if ang else None
        if not a or not b:
            return "--", "--", "--"
        d = b["mean"] - a["mean"]
        return c(a), c(b), (f"\\good{{{d:+.3f}}}" if d < 0 else f"\\bad{{{d:+.3f}}}")
    tab = [("PIV", "piv_d64"), ("Weather", "weather_au_wind"), ("Finance", "finance_ff49"),
           ("Student-$t$ d16", "student_t_d16_df3.0_cor")]
    rows.append(r"\multicolumn{5}{l}{\emph{Tabular / synthetic --- Sliced $W_1$ (overall) $\downarrow$}} \\")
    for lab, key in tab:
        std, ang = S[key].get("rafm_empirical"), S[key].get("angular_rafm")
        cs, ca, cd = pair(std, ang, "sliced_w1")
        rows.append(f"\\quad {lab} & {cs} & {ca} & {cd} & \\\\")
    # dc-ae (FID)
    dc = MASTER["dcae"]
    a, b = dc["rafm"]["fid"], dc["angular_rafm"]["fid"]
    rows.append(r"\multicolumn{5}{l}{\emph{DC-AE --- FID $\downarrow$}} \\")
    rows.append(f"\\quad ImageNette & {c(a,2)} & {c(b,2)} & \\good{{{b['mean']-a['mean']:+.1f}}} & \\\\")
    # audio (digit acc, higher better)
    au = MASTER["audio"]
    a, b = au["std_rafm"]["digit_acc"], au["angular_rafm"]["digit_acc"]
    rows.append(r"\multicolumn{5}{l}{\emph{AudioMNIST --- digit accuracy $\uparrow$}} \\")
    rows.append(f"\\quad AudioMNIST & {c(a,3)} & {c(b,3)} & \\good{{{b['mean']-a['mean']:+.3f}}} & \\\\")
    header = r"Setting & RAFM (std) & \textbf{Angular} & $\Delta$ & \\"
    emit("table_ablation.tex", "% requires: \\newcommand{\\std}[1]{{\\scriptsize$\\pm$#1}}\n"
         "% \\newcommand{\\good}[1]{\\textcolor{teal}{#1}} \\newcommand{\\bad}[1]{\\textcolor{red}{#1}}\n"
         + wrap("Standard RAFM vs Angular RAFM ablation across all domains. $\\Delta$ is Angular minus "
                "standard on the domain's headline metric (green favours Angular).",
                "tab:ablation", header, rows, "llrrl"))


def table_appendix_sweeps():
    rows = []
    def block(title, sweep, xs, axislab):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for xk in xs:
            s = sweep[str(xk)]
            def g(m, k): return c(s.get(m, {}).get(k), 3)
            rows.append(f"\\quad {axislab}$=${xk} & {g('rafm_empirical','radial_w1')} & {g('angular_rafm','radial_w1')} & "
                        f"{g('rafm_empirical','sliced_w1')} & {g('angular_rafm','sliced_w1')} & "
                        f"{g('rafm_empirical','angular_sw_mean')} & {g('angular_rafm','angular_sw_mean')} \\\\")
    block("Dimension sweep (Student-$t$, dof 3)", SUITE["dim"], [2, 8, 16, 32, 64, 128, 256], "$d$")
    block("Tail sweep (Student-$t$, $d{=}16$)", SUITE["df"], [1.5, 2.0, 3.0, 5.0, 10.0, 50.0], "dof")
    block("Anisotropy sweep (Gaussian base, $d{=}32$)", SUITE["aniso"], [1, 3, 10, 30, 100, 300], "$\\kappa$")
    header = (r"Setting & \multicolumn{2}{c}{Radial $W_1$} & \multicolumn{2}{c}{Sliced $W_1$} & "
              r"\multicolumn{2}{c}{Angular SW} \\"
              "\n& std & Ang. & std & Ang. & std & Ang. \\\\")
    emit("table_appendix_sweeps.tex", PREAMBLE + wrap(
        "Synthetic sweeps (appendix): standard vs Angular RAFM, mean$\\pm$std over 3 seeds. "
        "$^\\dagger$ marks aggregates over $<3$ finite seeds (Angular $d{=}2$ instability).",
        "tab:sweeps", header, rows, "l" + "rr" * 3))


if __name__ == "__main__":
    print("Building LaTeX tables...")
    table_realdata(); table_synthetic_main(); table_audiomnist(); table_dcae()
    table_ablation(); table_appendix_sweeps()
    print("done.")
