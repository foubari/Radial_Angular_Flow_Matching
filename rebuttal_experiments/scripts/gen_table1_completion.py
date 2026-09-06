"""Generate the RAFM-Ang Table-1 completion results (JSON + Markdown). Read-only over metrics."""
import json, glob, statistics as st, math, hashlib, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from rafm.data.piv import PIVDataset

K = ["radial_w1", "ks_stat", "sliced_w1", "total_train_time_s"]


def dshash(dim):
    d = PIVDataset(dim=dim, data_root=str(REPO / "data/piv"), split_seed=0)
    tr, te = d.get_train_data(), d.get_test_data()
    h = lambda x: hashlib.md5(np.ascontiguousarray(x.numpy())).hexdigest()
    return {"n_train": len(tr), "n_test": len(te), "dim": int(tr.shape[1]),
            "split": "split_seed=0 (exp1 PIVDataset _make_splits)", "train_md5": h(tr), "test_md5": h(te)}


def agg(path, m):
    fs = sorted(glob.glob(f"{path}/{m}/seed_*/metrics.json"))
    seeds = [Path(f).parent.name.replace("seed_", "") for f in fs]
    o = {"per_seed": {}, "mean_std": {}}
    for k in K:
        v = [json.load(open(f)).get(k) for f in fs]
        vv = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
        o["per_seed"][k] = {s: (round(x, 5) if x is not None else None) for s, x in zip(seeds, v)}
        o["mean_std"][k] = ({"mean": round(st.mean(vv), 5), "std": round(st.pstdev(vv), 5), "n": len(vv)} if vv else None)
    o["nfe_stored"] = json.load(open(fs[0])).get("nfe") if fs else None
    return o


res = {
    "protocol": {"batch_size": 256, "train_steps": 10000, "n_gen_samples": 10000,
                 "solver": "rk4", "rk4_steps": 128, "nfe": 512, "seeds": [8925, 77395, 65457]},
    "PIV_d64": {"status": "protocol-matched (rerun)", "dataset": dshash(64),
                "RAFM_Ang": agg(str(REPO / "outputs/exp1_main_benchmark/piv_d64"), "angular_rafm"),
                "RAFM_Vel_ref": agg(str(REPO / "outputs/exp1_main_benchmark/piv_d64"), "rafm_empirical")},
    "PIV_d256": {"status": "protocol-matched (already run; protocol verified)", "dataset": dshash(256),
                 "RAFM_Ang": agg(str(REPO / "outputs/exp1_main_benchmark/piv_d256"), "angular_rafm"),
                 "RAFM_Vel_ref": agg(str(REPO / "outputs/exp1_main_benchmark/piv_d256"), "rafm_empirical")},
    "StudentT_d16_d32": {"status": "BLOCKED - not run",
        "reason": ("Table-1 Student-t realization is not recoverable: StudentT._generate draws z from the "
                   "UNSEEDED global torch RNG (only the mixing matrix A is seeded, matrix_seed=42); "
                   "exp1_main_benchmark does not seed before dataset construction; no dataset .pt was cached. "
                   "Two fresh processes give different test-set md5 (dea8f374... vs 250e7363...), same A "
                   "(eb934f15...). Per instruction, stopped rather than evaluate on an incompatible realization.")},
}
(REPO / "rebuttal_experiments/RAFM_ANG_TABLE1_completion.json").write_text(json.dumps(res, indent=2))


def row(name, a, k):
    ps = a["per_seed"][k]; ms = a["mean_std"][k]
    per = " / ".join(f"{v}" for v in ps.values())
    return f"| {name} | {per} | **{ms['mean']}±{ms['std']}** |" if ms else f"| {name} | -- | -- |"


md = ["# RAFM-Ang — Table-1 completion (protocol-matched)", "",
      "Protocol: batch 256, 10k steps, n_gen 10000, RK4 **128 steps = 512 NFE**, seeds {8925, 77395, 65457}. "
      "Metrics lower = better.", ""]
for ds, blk in [("PIV d=64", res["PIV_d64"]), ("PIV d=256", res["PIV_d256"])]:
    d = blk["dataset"]
    md += [f"## {ds} — {blk['status']}", "",
           f"Dataset: n_train={d['n_train']}, n_test={d['n_test']}, dim={d['dim']}, {d['split']}; "
           f"train md5 `{d['train_md5'][:24]}`, test md5 `{d['test_md5'][:24]}`. nfe stored = {blk['RAFM_Ang']['nfe_stored']}.", ""]
    for k, lab in [("radial_w1", "Radial W1"), ("ks_stat", "Radial KS"), ("sliced_w1", "Sliced W1"), ("total_train_time_s", "Train time (s)")]:
        md += [f"### {lab}", "", "| method | per-seed (8925/77395/65457) | mean±std |", "|---|---|---|",
               row("RAFM-Ang", blk["RAFM_Ang"], k), row("RAFM-Vel (ref)", blk["RAFM_Vel_ref"], k), ""]
md += ["## Student-t ν=3, d={16,32} — BLOCKED (not run)", "", res["StudentT_d16_d32"]["reason"], "",
       "The existing rebuttal Student-t angular runs used batch 4096 **and** a different (also non-recoverable) "
       "realization, so they are not protocol-matched to Table-1 either. No compatible RAFM-Ang Student-t result "
       "can be produced without re-running all baselines on a freshly-cached realization — out of scope per instruction."]
(REPO / "rebuttal_experiments/RAFM_ANG_TABLE1_completion.md").write_text("\n".join(md), encoding="utf-8")
print("wrote RAFM_ANG_TABLE1_completion.{json,md}")
print("PIV d64 RAFM-Ang radial_w1:", res["PIV_d64"]["RAFM_Ang"]["mean_std"]["radial_w1"],
      "| RAFM-Vel:", res["PIV_d64"]["RAFM_Vel_ref"]["mean_std"]["radial_w1"])
