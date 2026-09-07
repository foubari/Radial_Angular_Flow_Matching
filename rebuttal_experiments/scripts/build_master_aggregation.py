"""Master aggregation of ALL methods x ALL datasets (tabular + DC-AE image + AudioMNIST).

Reads the native per-domain result stores (metric sets differ by domain, so one table per
dataset) and emits:
  * MASTER_RESULTS.md   — per-dataset tables (methods x that domain's metrics), mean +/- std over seeds
  * master_results.json — the same numbers, machine-readable

msgm is left BLANK where it was not run (piv, dc-ae, audiomnist). Angular RAFM is included
everywhere it exists. No experiment is retrained here — pure read + aggregate.

Sources:
  tabular   rebuttal_experiments/raw_results/{E10_piv,E_finance,E_weather}/<name>/<method>/seed_*/metrics.json
  dc-ae     experiments/image_latents/results_phaseB.json  (existing 4 methods, key "40000/<m>")
            experiments/image_latents/dit_sit_s<seed>/eval_std/angular_rafm/eval_40000.json  (angular, 3 seeds)
  audio     experiments/poc_audio/stage2_3seed.json  (key "24000")
"""
import json, glob, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def agg(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": m, "std": s, "vals": vals, "n": len(vals)}


def cell(a, prec=4):
    if a is None:
        return ""            # blank = not run
    return f"{a['mean']:.{prec}f}±{a['std']:.{prec}f}"


# ----------------------------------------------------------------- tabular
STD5 = ["gaussian_fm", "source_only_empirical", "rafm_empirical", "angular_rafm", "msgm"]
TAB = [("piv", "E10_piv", "piv_d64", STD5),
       ("finance", "E_finance", "finance_ff49", STD5),
       ("weather", "E_weather", "weather_au_wind", STD5),
       ("student_t", "E1_reproduction", "student_t_d16_df3.0_cor",
        ["gaussian_fm", "source_only_empirical", "source_only_oracle",
         "rafm_empirical", "rafm_oracle", "angular_rafm", "msgm"])]
TAB_LABEL = {"gaussian_fm": "Gaussian FM", "source_only_empirical": "Source-only (emp.)",
             "source_only_oracle": "Source-only (oracle)", "rafm_empirical": "RAFM (std)",
             "rafm_oracle": "RAFM (oracle)", "angular_rafm": "Angular RAFM", "msgm": "MSGM"}
TAB_METRICS = ["radial_w1", "ks_stat", "sliced_w1", "angular_sw_mean", "nan_rate"]


def load_tabular():
    out = {}
    for ds, edir, name, methods in TAB:
        out[ds] = {}
        for m in methods:
            seed_files = sorted(glob.glob(str(REPO / "rebuttal_experiments/raw_results" / edir / name / m / "seed_*/metrics.json")))
            if not seed_files:
                out[ds][m] = None
                continue
            per = {k: [] for k in TAB_METRICS}
            for f in seed_files:
                d = json.load(open(f))
                for k in TAB_METRICS:
                    if k in d:
                        per[k].append(d[k])
            out[ds][m] = {k: agg(v) for k, v in per.items()}
    return out


# ----------------------------------------------------------------- dc-ae
DCAE_MAP = {"gaussian_euclidean": "Gaussian FM", "matched_euclidean": "Matched-Eucl.",
            "fixed_spherical": "Fixed-spherical", "rafm": "RAFM (std)",
            "angular_rafm": "Angular RAFM", "msgm": "MSGM"}
DCAE_ORDER = ["gaussian_euclidean", "matched_euclidean", "fixed_spherical", "rafm", "angular_rafm", "msgm"]
DCAE_METRICS = ["fid", "radial_w1", "ks", "sliced_w1", "precision", "recall", "coverage"]


def load_dcae():
    res = {}
    pb = REPO / "experiments/image_latents/results_phaseB.json"
    phaseb = json.load(open(pb)) if pb.exists() else {}
    for m in DCAE_ORDER:
        key = f"40000/{m}"
        if key in phaseb:
            e = phaseb[key]
            res[m] = {k: (agg(e[k]["vals"]) if k in e and "vals" in e[k] else None) for k in DCAE_METRICS}
        elif m == "angular_rafm":
            # aggregate the 3 per-seed eval jsons produced by launch_angular_dit.bat
            per = {k: [] for k in DCAE_METRICS}
            found = False
            for seed in (8925, 7, 1234):
                f = REPO / f"experiments/image_latents/dit_sit_s{seed}/eval_std/angular_rafm/eval_40000.json"
                if not f.exists():
                    continue
                found = True
                d = json.load(open(f))
                flat = {**d.get("latent", {}), **d.get("image", {})}
                for k in DCAE_METRICS:
                    if k in flat:
                        per[k].append(flat[k])
            res[m] = {k: agg(v) for k, v in per.items()} if found else None
        else:
            res[m] = None   # msgm: not run
    return res


# ----------------------------------------------------------------- audio
AUD_MAP = {"gaussian": "Gaussian FM", "matched": "Matched-Eucl.", "fixed_spher": "Fixed-spherical",
           "std_rafm": "RAFM (std)", "angular_rafm": "Angular RAFM", "msgm": "MSGM"}
AUD_ORDER = ["gaussian", "matched", "fixed_spher", "std_rafm", "angular_rafm", "msgm"]
AUD_METRICS = ["digit_acc", "energy_KS", "cov>q95", "cov>q99", "PIT"]


def load_audio():
    f = REPO / "experiments/poc_audio/stage2_3seed.json"
    if not f.exists():
        return {}
    d = json.load(open(f))["24000"]
    res = {}
    for m in AUD_ORDER:
        if m in d:
            res[m] = {k: agg(d[m][k]["vals"]) for k in AUD_METRICS if k in d[m]}
        else:
            res[m] = None    # msgm: not run
    return res


# ----------------------------------------------------------------- render
def md_table(title, methods, labels, metrics, data, prec=4, note=""):
    lines = [f"### {title}", ""]
    if note:
        lines += [f"*{note}*", ""]
    hdr = "| Method | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines += [hdr, sep]
    for m in methods:
        row = data.get(m)
        cells = []
        for k in metrics:
            cells.append(cell(row[k], prec) if (row and row.get(k) is not None) else "")
        lines.append(f"| {labels[m]} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    tab = load_tabular()
    dcae = load_dcae()
    audio = load_audio()

    md = ["# Master results — all methods x all datasets",
          "",
          "Mean ± std over 3 seeds. Cells are blank where a method was **not run** (MSGM on piv / DC-AE / "
          "AudioMNIST; run later if duration is reasonable). Metric sets differ by domain, so one table per "
          "dataset. Arrows: radial_w1 / ks / sliced_w1 / angular_sw / energy_KS / FID lower = better; "
          "digit_acc / precision / recall / coverage higher = better.",
          ""]

    md.append("## Tabular (MLP flows, nfe=512)\n")
    for ds, edir, name, methods in TAB:
        md.append(md_table(f"{ds}  ({name})", methods, TAB_LABEL, TAB_METRICS, tab[ds], prec=4))

    md.append("## DC-AE ImageNette latents (SiT backbone, 40k, nfe=25→100 fn-evals)\n")
    md.append(md_table("dc-ae", DCAE_ORDER, DCAE_MAP, DCAE_METRICS, dcae, prec=3))

    md.append("## AudioMNIST (reversible complex-STFT, UNet flow, 24k)\n")
    md.append(md_table("audiomnist", AUD_ORDER, AUD_MAP, AUD_METRICS, audio, prec=3))

    out_md = REPO / "rebuttal_experiments/MASTER_RESULTS.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = REPO / "rebuttal_experiments/master_results.json"
    out_json.write_text(json.dumps({"tabular": tab, "dcae": dcae, "audio": audio}, indent=2))
    print(f"wrote {out_md}\nwrote {out_json}")
    # console preview of headline
    print("\n--- angular_rafm presence ---")
    print("tabular:", {ds: (tab[ds].get("angular_rafm") is not None) for ds, _, _, _ in TAB})
    print("dc-ae  :", dcae.get("angular_rafm") is not None)
    print("audio  :", audio.get("angular_rafm") is not None)


if __name__ == "__main__":
    main()
