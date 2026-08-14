"""Aggregate Phase-B eval JSONs -> mean+-std over seeds, per method x checkpoint. CPU-only.

Reads experiments/image_latents/dit_sit/seed_*/eval_std/<method>/eval_<step>.json (written by
dit_eval_sit.py) and produces:
  * a markdown results table (RESULTS_PHASEB.md) with FID/KID/prdc + latent radial_w1/ks/sliced_w1,
    reported as mean +- std across the available seeds, at each checkpoint;
  * results_phaseB.json with the raw aggregation.
Safe to run any time after some evals exist (missing seeds/steps are simply skipped).
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/"experiments/image_latents/dit_sit"
METHODS=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"]
LAT=["radial_w1","ks","sliced_w1"]; IMG=["fid","kid","precision","recall","density","coverage"]

def ms(v):
    a=np.array(v,float); return (float(a.mean()),float(a.std())) if len(a) else (float("nan"),0.0)

def main():
    # data[(step,method)][metric] = list over seeds
    data=defaultdict(lambda:defaultdict(list)); seeds=defaultdict(set)
    for f in glob.glob(str(ROOT/"seed_*/eval_std/*/eval_*.json")):
        j=json.loads(Path(f).read_text()); step=j["step"]; method=j["method"]
        seed=re.search(r"seed_(\d+)",f).group(1); seeds[(step,method)].add(seed)
        for k in LAT: data[(step,method)][k].append(j["latent"][k])
        for k in IMG: data[(step,method)][k].append(j["image"][k])
    if not data: print("No eval JSONs found yet."); return
    steps=sorted({s for s,_ in data},reverse=True)
    out={}; lines=["# Phase-B results (SiT, scaled DC-AE, Imagenette-10) — mean+-std over seeds",
        "\nStandardized: torch-fidelity FID/KID + prdc, CFG=1 unguided, RK4 nfe25, 3k gen vs 3925 real ref.",
        "Latent metrics vs held-out centered test. FID is limited-sample (n=3000) — relative, not FID-50k.\n"]
    for step in steps:
        lines+=[f"\n## checkpoint {step}\n",
            "| method | seeds | radial_w1 | ks | sliced_w1 | FID | KID | precision | recall |",
            "|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            if (step,m) not in data: continue
            d=data[(step,m)]; ns=len(seeds[(step,m)]); out[f"{step}/{m}"]={"seeds":sorted(seeds[(step,m)])}
            cell={}
            for k in LAT+IMG:
                mu,sd=ms(d[k]); out[f"{step}/{m}"][k]={"mean":mu,"std":sd,"vals":d[k]}
                cell[k]=f"{mu:.3f}±{sd:.3f}" if k in LAT or k in("kid","precision","recall","density","coverage") else f"{mu:.1f}±{sd:.1f}"
            lines.append(f"| {m} | {ns} | {cell['radial_w1']} | {cell['ks']} | {cell['sliced_w1']} | {cell['fid']} | {cell['kid']} | {cell['precision']} | {cell['recall']} |")
    (REPO/"experiments/image_latents/RESULTS_PHASEB.md").write_text("\n".join(lines))
    (REPO/"experiments/image_latents/results_phaseB.json").write_text(json.dumps(out,indent=2))
    print("\n".join(lines)); print("\nWrote RESULTS_PHASEB.md + results_phaseB.json")

if __name__=="__main__": main()
