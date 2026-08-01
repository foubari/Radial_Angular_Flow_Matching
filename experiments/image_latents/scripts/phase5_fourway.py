"""Phase 5 — four-way source/path comparison on the DC-AE latent (non-degenerate radial).

3 seeds. Reports (per method): radial_W1, sliced_W1, normalized-direction sliced_W1
(dir_sliced_w1), common-radial sliced_W1 (cr_sliced_w1). Also the train/val and train/test
radial-W1 FLOOR (irreducible for an empirical-radial source). Saves generated latents +
train mean (mu) per method so images can be decoded and a small-sample FID computed.

Methods (identical MLP arch/budget), on TRAIN-mean-centered latents:
  gaussian_euclidean  : Gaussian source        + Euclidean path
  matched_euclidean   : empirical radial source + Euclidean path        (source-only)
  fixed_spherical     : fixed radius R0 source  + spherical path         (SFM/RFM-style)
  rafm_empirical      : empirical radial source + spherical path         (RAFM)
fixed_spherical trains on the radius-normalized dataset (all ||x1||=R0). All methods are
evaluated against the ORIGINAL-scale (centered) held-out test latents. Resumable (skips
runs whose metrics.json exists).
"""
import argparse, json, sys
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
from rafm.metrics.radial import radial_metrics, _wasserstein1_1d
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics
from rafm.utils.seeds import set_all_seeds

METHODS = [("gaussian_euclidean","gaussian","euclidean"),
           ("matched_euclidean","empirical","euclidean"),
           ("fixed_spherical","empirical","spherical"),
           ("rafm_empirical","empirical","spherical")]

class LatentDS:
    def __init__(self, data, project_R0=False, split_seed=0):
        self.dim = data.shape[1]; self.A = None
        n = data.shape[0]
        g = torch.Generator().manual_seed(split_seed); perm = torch.randperm(n, generator=g)
        ntr, nva = int(n*0.6), int(n*0.2)
        self._tr, self._va, self._te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
        self.mu = data.float()[self._tr].mean(0)          # train-only centering
        self._data = data.float() - self.mu
        self.R0 = float(self._data[self._tr].norm(dim=1).mean())
        self._project = project_R0
        if project_R0:
            r = self._data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._train_data = (self.R0 * self._data / r)[self._tr]
        else:
            self._train_data = self._data[self._tr]
    def get_train_data(self): return self._train_data
    def get_test_data(self): return self._data[self._te]
    def get_val_data(self): return self._data[self._va]
    @property
    def n_train(self): return len(self._tr)

def sliced(A, B, dirs):
    q = np.linspace(0,1,1000)
    return float(np.mean(np.abs(np.quantile(A@dirs,q,0)-np.quantile(B@dirs,q,0))))

def directional(gen, test, seed=0, n_proj=300):
    g=gen.numpy().astype(np.float64); t=test.numpy().astype(np.float64)
    gu=g/np.clip(np.linalg.norm(g,axis=1,keepdims=True),1e-8,None)
    tu=t/np.clip(np.linalg.norm(t,axis=1,keepdims=True),1e-8,None)
    rng=np.random.default_rng(seed); d=g.shape[1]
    dirs=rng.standard_normal((d,n_proj)); dirs/=np.linalg.norm(dirs,axis=0,keepdims=True)
    dir_sw=sliced(gu,tu,dirs)                                   # normalized-direction sliced W1
    rt=np.linalg.norm(t,axis=1); rdraw=rt[rng.integers(0,len(rt),len(gu))]
    cr_sw=sliced(gu*rdraw[:,None], t, dirs)                     # common-radial sliced W1
    return dir_sw, cr_sw

def run_method(name, sk, pk, latents, args, seed):
    run_dir = Path(args.outdir)/"raw"/name/f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/"metrics.json").exists():
        return json.loads((run_dir/"metrics.json").read_text())
    ds = LatentDS(latents, project_R0=(name=="fixed_spherical"), split_seed=args.split_seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train, test = ds.get_train_data(), ds.get_test_data()
    set_all_seeds(seed)
    source = GaussianSource() if sk=="gaussian" else RadialEmpiricalSource(mode="ecdf").fit(train)
    path = SphericalGeodesicPath() if pk=="spherical" else EuclideanPath()
    model = MLP(input_dim=ds.dim, hidden_dim=args.hidden, n_layers=args.layers)
    cfg={"lr":1e-3,"batch_size":args.batch,"n_train_steps":args.steps,"log_every":1000,
         "ckpt_every":args.steps,"device":device,"solver":"rk4","nfe":args.nfe,
         "hidden_dim":args.hidden,"n_layers":args.layers,"path":pk,"dataset":{"name":"dcae","dim":ds.dim}}
    tr=Trainer(model,path,source,ds,cfg,seed,run_dir); stats=tr.train()
    smp=Sampler(model,source,cfg,project_tangent=(pk=="spherical"))
    n_gen=min(args.n_gen,len(test)); gen=smp.sample(n_gen,ds.dim)["samples"]
    m={}; m.update(radial_metrics(gen,test)); m.update(distributional_metrics(gen,test,n_projections=300))
    m.update(angular_metrics(gen,test,n_bins=4,n_projections=100)); m.update(stability_metrics(gen))
    m["dir_sliced_w1"], m["cr_sliced_w1"] = directional(gen,test,seed=seed)
    m["train_time_s"]=stats["total_train_time_s"]; m["R0"]=ds.R0
    (run_dir/"metrics.json").write_text(json.dumps(m,indent=2))
    torch.save(gen, run_dir/"gen_latents.pt"); torch.save(ds.mu, run_dir/"mu.pt")
    print(f"  [{name} s{seed}] radial={m['radial_w1']:.4f} sliced={m['sliced_w1']:.4f} "
          f"dir={m['dir_sliced_w1']:.4f} cr={m['cr_sliced_w1']:.4f}", flush=True)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--latents", default="experiments/image_latents/data/dcae_latents.pt")
    ap.add_argument("--outdir", default="experiments/image_latents/fourway")
    ap.add_argument("--steps",type=int,default=10000); ap.add_argument("--batch",type=int,default=2048)
    ap.add_argument("--hidden",type=int,default=256); ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--nfe",type=int,default=128); ap.add_argument("--n_gen",type=int,default=5000)
    ap.add_argument("--seeds",default="8925,77395,65457"); ap.add_argument("--split_seed",type=int,default=0)
    args=ap.parse_args()
    latents=torch.load(args.latents,map_location="cpu").float()
    seeds=[int(s) for s in args.seeds.split(",")]
    print(f"DC-AE latents {tuple(latents.shape)} CoV {float(latents.norm(dim=1).std()/latents.norm(dim=1).mean()):.4f}; seeds {seeds}",flush=True)
    # radial floor (source = train eCDF; irreducible under this split)
    ds0=LatentDS(latents, split_seed=args.split_seed)
    rtr=ds0._train_data.norm(dim=1); rva=ds0.get_val_data().norm(dim=1); rte=ds0.get_test_data().norm(dim=1)
    floor={"radial_w1_train_val":round(_wasserstein1_1d(rtr,rva),4),
           "radial_w1_train_test":round(_wasserstein1_1d(rtr,rte),4),
           "test_norm_mean":float(rte.mean()),"test_norm_std":float(rte.std())}
    Path(args.outdir).mkdir(parents=True,exist_ok=True)
    (Path(args.outdir)/"radial_floor.json").write_text(json.dumps(floor,indent=2))
    print("radial floor train/val %.3f train/test %.3f"%(floor["radial_w1_train_val"],floor["radial_w1_train_test"]),flush=True)
    all_m={}
    for seed in seeds:
        for name,sk,pk in METHODS:
            all_m.setdefault(name,[]).append(run_method(name,sk,pk,latents,args,seed))
    # aggregate mean+-std
    import statistics as st
    keys=["radial_w1","sliced_w1","dir_sliced_w1","cr_sliced_w1","ks_stat","nan_rate"]
    lines=[f"# Phase 5 four-way on DC-AE latent (2048-d, CoV 12%), {len(seeds)} seeds\n",
           f"Radial-W1 floor (empirical source, irreducible): train/val {floor['radial_w1_train_val']}, train/test {floor['radial_w1_train_test']}. "
           "Latent-distribution metrics (not image FID).\n",
           "| method | "+" | ".join(keys)+" |","|"+"---|"*(len(keys)+1)]
    for name,_,_ in METHODS:
        cells=[name]
        for k in keys:
            v=[mm[k] for mm in all_m[name]]; mean=sum(v)/len(v)
            sd=st.pstdev(v) if len(v)>1 else 0.0
            cells.append(f"{mean:.4f}±{sd:.4f}" if sd else f"{mean:.4f}")
        lines.append("| "+" | ".join(cells)+" |")
    (Path(args.outdir)/"fourway_dcae.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines)); print("\nDONE")

if __name__=="__main__": main()
