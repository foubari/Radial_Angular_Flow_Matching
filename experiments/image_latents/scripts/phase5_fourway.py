"""Phase 5 — four-way source/path comparison on the DC-AE latent (non-degenerate radial).

Tests whether preserving the empirical radial law (RAFM) improves over a fixed-radius
spherical prior, on a real decodable image latent that HAS radial variation (DC-AE, CoV 12%).
Pure rafm-library MLP flow on the saved 2048-d latents (no diffusers needed here).

Methods (identical MLP arch/budget):
  gaussian_euclidean   : Gaussian source        + Euclidean path
  matched_euclidean    : empirical radial source + Euclidean path        (source-only)
  fixed_spherical      : fixed radius R0 source  + spherical path         (SFM/RFM-style)
  rafm_empirical       : empirical radial source + spherical path         (RAFM)

fixed_spherical is realized by training on the radius-normalized dataset (all ||x1||=R0),
so its empirical source is a point mass at R0 and the geodesic path is well-defined; its
generated samples all have radius R0. All methods are evaluated against the ORIGINAL
(un-normalized) held-out test latents.
"""
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rafm.models.mlp import MLP
from rafm.flow_matching.trainer import Trainer
from rafm.flow_matching.sampler import Sampler
from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.paths.euclidean import EuclideanPath
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics
from rafm.utils.seeds import set_all_seeds


class LatentDS:
    def __init__(self, data, name, project_R0=False, split_seed=0):
        self.name = name; self.dim = data.shape[1]; self.A = None
        n = data.shape[0]
        g = torch.Generator().manual_seed(split_seed); perm = torch.randperm(n, generator=g)
        ntr, nva = int(n*0.6), int(n*0.2)
        self._tr, self._va, self._te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
        mu = data.float()[self._tr].mean(0)           # train-only centering (no leakage)
        self._data = data.float() - mu
        self.R0 = float(self._data[self._tr].norm(dim=1).mean())
        if project_R0:
            r = self._data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._train_data = (self.R0 * self._data / r)[self._tr]
        else:
            self._train_data = self._data[self._tr]
    def get_train_data(self): return self._train_data
    def get_test_data(self): return self._data[self._te]   # ORIGINAL scale always
    @property
    def n_train(self): return len(self._tr)


def directional(gen, test, n_proj=300, seed=0):
    g = gen.numpy().astype(np.float64); t = test.numpy().astype(np.float64)
    gu = g/np.clip(np.linalg.norm(g,axis=1,keepdims=True),1e-8,None)
    tu = t/np.clip(np.linalg.norm(t,axis=1,keepdims=True),1e-8,None)
    rng = np.random.default_rng(seed); d = g.shape[1]
    dirs = rng.standard_normal((d,n_proj)); dirs/=np.linalg.norm(dirs,axis=0,keepdims=True)
    def sw(A,B):
        q=np.linspace(0,1,1000); return float(np.mean(np.abs(np.quantile(A@dirs,q,0)-np.quantile(B@dirs,q,0))))
    dir_sw = sw(gu,tu)
    # common-radial: give gen dirs the TEST radii
    rt = np.linalg.norm(t,axis=1); rdraw = rt[rng.integers(0,len(rt),len(gu))]
    cr = sw(gu*rdraw[:,None]/1.0, t)  # compare re-radialised gen dirs vs test in full space (normalize scale by proj)
    return dir_sw, cr


def run_method(name, source_kind, path_kind, latents, args):
    project = (name == "fixed_spherical")
    ds = LatentDS(latents, "dcae", project_R0=project, split_seed=args.split_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = ds.get_train_data(); test = ds.get_test_data()
    set_all_seeds(args.seed)
    if source_kind == "gaussian":
        source = GaussianSource()
    else:
        source = RadialEmpiricalSource(mode="ecdf").fit(train)   # for fixed_spherical, train is R0-normalized -> point mass R0
    path = SphericalGeodesicPath() if path_kind == "spherical" else EuclideanPath()
    model = MLP(input_dim=ds.dim, hidden_dim=args.hidden, n_layers=args.layers)
    cfg = {"lr":1e-3,"batch_size":args.batch,"n_train_steps":args.steps,"log_every":500,
           "ckpt_every":args.steps,"device":device,"solver":"rk4","nfe":args.nfe,
           "hidden_dim":args.hidden,"n_layers":args.layers,"path":path_kind,
           "dataset":{"name":"dcae","dim":ds.dim}}
    run_dir = Path(args.outdir)/"raw"/name/f"seed_{args.seed}"; run_dir.mkdir(parents=True,exist_ok=True)
    tr = Trainer(model, path, source, ds, cfg, args.seed, run_dir)
    stats = tr.train()
    smp = Sampler(model, source, cfg, project_tangent=(path_kind == "spherical")); n_gen = min(args.n_gen, len(test))
    gen = smp.sample(n_gen, ds.dim)["samples"]
    m = {}
    m.update(radial_metrics(gen, test)); m.update(distributional_metrics(gen, test, n_projections=300))
    m.update(angular_metrics(gen, test, n_bins=4, n_projections=100)); m.update(stability_metrics(gen))
    m["dir_sliced_w1"], m["cr_sliced_w1"] = directional(gen, test)
    m["train_time_s"] = stats["total_train_time_s"]; m["R0"] = ds.R0
    (run_dir/"metrics.json").write_text(json.dumps(m,indent=2))
    print(f"  {name:20} radial_w1={m['radial_w1']:.4f} sliced_w1={m['sliced_w1']:.4f} "
          f"dir_sw1={m['dir_sliced_w1']:.4f} cr_sw1={m['cr_sliced_w1']:.4f} nan={m['nan_rate']:.3f}", flush=True)
    return name, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="experiments/image_latents/data/dcae_latents.pt")
    ap.add_argument("--outdir", default="experiments/image_latents/fourway")
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--nfe", type=int, default=128); ap.add_argument("--n_gen", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=8925); ap.add_argument("--split_seed", type=int, default=0)
    args = ap.parse_args()
    latents = torch.load(args.latents, map_location="cpu").float()
    print(f"DC-AE latents {tuple(latents.shape)} norm-CoV {float(latents.norm(dim=1).std()/latents.norm(dim=1).mean()):.4f}", flush=True)
    METHODS = [("gaussian_euclidean","gaussian","euclidean"),
               ("matched_euclidean","empirical","euclidean"),
               ("fixed_spherical","empirical","spherical"),
               ("rafm_empirical","empirical","spherical")]
    rows = {}
    for name, sk, pk in METHODS:
        _, m = run_method(name, sk, pk, latents, args); rows[name] = m
    # table
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    keys = ["radial_w1","ks_stat","sliced_w1","dir_sliced_w1","cr_sliced_w1","nan_rate","train_time_s"]
    lines = ["# Phase 5 four-way on DC-AE latent (2048-d, non-degenerate radial)\n",
             "| method | "+" | ".join(keys)+" |","|"+"---|"*(len(keys)+1)]
    for name,_ ,_ in METHODS:
        m=rows[name]; lines.append("| "+name+" | "+" | ".join(f"{m[k]:.4f}" for k in keys)+" |")
    (out/"fourway_dcae.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines)); print("\nwrote",out/"fourway_dcae.md")


if __name__ == "__main__":
    main()
