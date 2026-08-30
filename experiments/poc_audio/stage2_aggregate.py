"""Stage-2 3-seed aggregation: mean +- std for matched / std-RAFM / angular-RAFM at 12/18/24k. CPU.
Seeds: 8925 (base eval dirs) + 1234 + 7."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; EV=HERE/"eval_unet"
# method -> list of 3 seed eval dirs (8925, 1234, 7)
SRC={
 "gaussian":     ["gaussian_euclidean","runs_s1234_gaussian_euclidean","runs_s7_gaussian_euclidean"],
 "matched":      ["matched_euclidean","runs_s1234_matched_euclidean","runs_s7_matched_euclidean"],
 "fixed_spher":  ["fixed_spherical","runs_s1234_fixed_spherical","runs_s7_fixed_spherical"],
 "std_rafm":     ["rafm","runs_s1234_rafm","runs_s7_rafm"],
 "angular_rafm": ["runs_angular_rafm","runs_s1234_ang_rafm","runs_s7_ang_rafm"],
}
STEPS=[12000,18000,24000]
FIELDS=[("digit_acc","content","digit_acc"),("energy_KS","energy","ks"),
        ("cov>q95","energy","cov_gt_q95"),("cov>q99","energy","cov_gt_q99"),
        ("cov<q10","energy","cov_lt_q10"),("PIT","energy","pit_mean")]
def load(d,s):
    f=EV/d/f"step{s}"/"eval.json"; return json.loads(f.read_text()) if f.exists() else None
out={}
for s in STEPS:
    print(f"\n===== checkpoint {s}  (mean +- std over 3 seeds) =====")
    hdr=f"{'method':13}"+"".join(f"{n:>16}" for n,_,_ in FIELDS); print(hdr)
    out[s]={}
    for m,dirs in SRC.items():
        vals={n:[] for n,_,_ in FIELDS}; ns=0
        for d in dirs:
            j=load(d,s)
            if not j: continue
            ns+=1
            for n,sec,key in FIELDS: vals[n].append(j[sec][key])
        row=f"{m:13}"
        out[s][m]={"n_seeds":ns}
        for n,_,_ in FIELDS:
            a=np.array(vals[n],float); mu,sd=(a.mean(),a.std()) if len(a) else (np.nan,0)
            out[s][m][n]={"mean":round(float(mu),4),"std":round(float(sd),4),"vals":[round(float(x),4) for x in a]}
            row+=f"{mu:>9.3f}+-{sd:<5.3f}"
        print(row+f"   (n={ns})")
(HERE/"stage2_3seed.json").write_text(json.dumps(out,indent=2))
print("\nwrote stage2_3seed.json")
