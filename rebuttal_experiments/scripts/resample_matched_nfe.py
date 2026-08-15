"""Re-sample the run_real Angular RAFM checkpoints at the SAME sampling budget as the retained
baselines: RK4 128 steps = 512 model evaluations (baselines stored nfe=512; the original Angular
run_real launches used 512 steps = 2048 evals). Sampler-only, NO retraining.

Scope: the 11 run_real settings only (piv, finance, weather, aniso k1..k300, finance random-split,
weather big-model). Synthetic exp1 runs already used 128 steps (matched) and are untouched.

For each angular_rafm/seed_* run it loads config.yaml + checkpoint.pt, rebuilds the exact dataset
(same split), source and model, samples at nfe=128, recomputes the identical metric suite vs the same
test split, backs up the old metrics.json/samples.pt (as *_nfe2048.*), and overwrites with matched-NFE.
"""
import json, glob, shutil, sys
from pathlib import Path
import torch, yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "rebuttal_experiments"))
from lib.real_dataset import RealTabularDataset
from rafm.models.mlp import MLP
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.flow_matching.sampler import Sampler
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics
from rafm.utils.seeds import set_all_seeds

MATCHED_STEPS = 128    # RK4 128 steps * 4 = 512 evals == retained baselines

# name -> .pt path (run_real datasets)
def pt_for(name):
    if name == "piv_d64":
        return REPO / "data/piv/piv_d64.pt"
    if name == "finance_ff49":
        return REPO / "rebuttal_experiments/data/finance/finance_returns_raw.pt"
    if name == "weather_au_wind":
        return REPO / "rebuttal_experiments/data/weather/weather_au_wind.pt"
    if name.startswith("aniso_k"):
        k = name.split("aniso_k")[1]
        return REPO / f"rebuttal_experiments/data/aniso/aniso_gauss_d32_k{k}.pt"
    raise ValueError(f"no pt mapping for {name}")


RUN_REAL_DIRS = ["E10_piv", "E_finance", "E_weather", "E9_aniso",
                 "E_finance_randomsplit", "E_weather_bigmodel"]


def build_model(cfg, dim, device):
    arch = cfg.get("arch", "mlp")
    if arch == "resmlp":
        from lib.resmlp import ResidualMLP
        m = ResidualMLP(input_dim=dim, hidden_dim=cfg.get("hidden_dim", 128), n_blocks=cfg.get("n_layers", 3))
    else:
        m = MLP(input_dim=dim, hidden_dim=cfg.get("hidden_dim", 128), n_layers=cfg.get("n_layers", 3))
    return m.to(device)


def resample_run(run_dir: Path, device: str):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if int(cfg.get("nfe", 0)) == MATCHED_STEPS:
        print(f"  {run_dir.relative_to(REPO)}: already nfe={MATCHED_STEPS}, skip"); return None
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        print(f"  {run_dir.relative_to(REPO)}: NO checkpoint, skip"); return None

    name = cfg["dataset"]["name"]; dim = cfg["dataset"]["dim"]
    split = cfg.get("split", "chrono")
    seed = int(run_dir.name.split("_")[1])
    ds = RealTabularDataset(str(pt_for(name)), name, split=split)
    train = ds.get_train_data(); test = ds.get_test_data()

    model = build_model(cfg, dim, device)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd); model.eval()

    source = RadialEmpiricalSource(mode="ecdf").fit(train)
    scfg = dict(cfg); scfg["nfe"] = MATCHED_STEPS; scfg["solver"] = "rk4"
    scfg["path"] = cfg.get("path", "spherical_geodesic"); scfg["angular"] = bool(cfg.get("angular", False))
    scfg["device"] = device
    sampler = Sampler(model, source, scfg)

    set_all_seeds(seed)
    n_gen = int(cfg.get("n_gen_samples", min(10000, ds.n_test)))
    gen = sampler.sample(n_gen, dim)
    samples = gen["samples"]

    m = {}
    m.update(radial_metrics(samples, test))
    m.update(distributional_metrics(samples, test, n_projections=cfg.get("n_projections_sw", 500)))
    m.update(angular_metrics(samples, test, n_bins=cfg.get("n_angular_bins", 4), n_projections=200))
    m.update(stability_metrics(samples))
    m["nfe"] = gen["nfe"]; m["sample_time_s"] = gen["sample_time_s"]
    old = json.loads((run_dir / "metrics.json").read_text())
    m["total_train_time_s"] = old.get("total_train_time_s")
    m["n_params"] = old.get("n_params", sum(p.numel() for p in model.parameters()))

    # backup old (nfe2048) then overwrite
    if not (run_dir / "metrics_nfe2048.json").exists():
        shutil.copy(run_dir / "metrics.json", run_dir / "metrics_nfe2048.json")
        if (run_dir / "samples.pt").exists():
            shutil.copy(run_dir / "samples.pt", run_dir / "samples_nfe2048.pt")
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    torch.save(samples, run_dir / "samples.pt")
    return (old.get("radial_w1"), m["radial_w1"], old.get("sliced_w1"), m["sliced_w1"], m["nfe"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirs = []
    for e in RUN_REAL_DIRS:
        dirs += sorted(Path(p).parent for p in glob.glob(str(REPO / "rebuttal_experiments/raw_results" / e / "*/angular_rafm/seed_*/config.yaml")))
    print(f"Re-sampling {len(dirs)} run_real Angular runs at nfe={MATCHED_STEPS} steps (=512 evals), device={device}\n")
    n = 0
    for rd in dirs:
        r = resample_run(rd, device)
        if r:
            n += 1
            print(f"  {rd.relative_to(REPO)}: radial_w1 {r[0]:.4f}->{r[1]:.4f}  sliced_w1 {r[2]:.4f}->{r[3]:.4f}  (nfe now {r[4]})", flush=True)
    print(f"\nDone. Re-sampled {n} runs. Backups saved as *_nfe2048.*")


if __name__ == "__main__":
    main()
