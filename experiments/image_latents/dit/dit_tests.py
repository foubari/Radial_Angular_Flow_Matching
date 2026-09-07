"""Phase-A unit tests: geometry, velocity, source marginals, preprocessing inverse.
Run in venv (has rafm). Prints PASS/FAIL per test; exits nonzero on any failure.
"""
import sys, torch, numpy as np
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(REPO))
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.paths.euclidean import EuclideanPath
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.sources.gaussian import GaussianSource
from rafm.metrics.radial import _ks_stat

fails=[]
def check(name,cond,extra=""):
    print(("PASS " if cond else "FAIL ")+name+("  "+extra if extra else "")); (fails.append(name) if not cond else None)

d=2048; n=512; SF=0.41407
# ---- spherical geometry ----
sp=SphericalGeodesicPath()
R=(torch.rand(n,1)*3+1)  # radii
u1=torch.randn(n,d); u1=u1/u1.norm(dim=1,keepdim=True); x1=R*u1
u0=torch.randn(n,d); u0=u0/u0.norm(dim=1,keepdim=True); x0=R*u0     # SAME sphere as x1
for tt in [0.0,0.3,0.7,1.0]:
    xt=sp.sample_path(x0,x1,torch.full((n,),tt))
    check(f"slerp norm preserved @t={tt}", torch.allclose(xt.norm(dim=1),R.squeeze(1),atol=1e-3),
          f"max|dev| {float((xt.norm(dim=1)-R.squeeze(1)).abs().max()):.2e}")
check("endpoint t=0 == source", torch.allclose(sp.sample_path(x0,x1,torch.zeros(n)),x0,atol=1e-4))
check("endpoint t=1 == target", torch.allclose(sp.sample_path(x0,x1,torch.ones(n)),x1,atol=1e-4))
# tangent velocity orthogonal to state
for tt in [0.2,0.5,0.9]:
    xt=sp.sample_path(x0,x1,torch.full((n,),tt)); ut=sp.conditional_vector_field(x0,x1,torch.full((n,),tt))
    ip=(xt*ut).sum(1)/(xt.norm(dim=1)*ut.norm(dim=1)+1e-9)
    check(f"tangent velocity ⊥ state @t={tt}", float(ip.abs().max())<1e-2, f"max|cos| {float(ip.abs().max()):.2e}")
# analytic velocity ~ finite difference
tt=0.4; h=1e-3
xt=sp.sample_path(x0,x1,torch.full((n,),tt)); ut=sp.conditional_vector_field(x0,x1,torch.full((n,),tt))
fd=(sp.sample_path(x0,x1,torch.full((n,),tt+h))-sp.sample_path(x0,x1,torch.full((n,),tt-h)))/(2*h)
rel=((ut-fd).norm(dim=1)/(fd.norm(dim=1)+1e-9)).mean()
check("spherical velocity ≈ finite diff", float(rel)<1e-2, f"rel err {float(rel):.2e}")
# euclidean velocity = x1-x0
eu=EuclideanPath(); ue=eu.conditional_vector_field(x0,x1,torch.full((n,),0.5))
check("euclidean velocity == x1-x0", torch.allclose(ue,x1-x0,atol=1e-4))

# ---- source marginals ----
train=torch.randn(4000,d)*2*SF   # mimic scaled data scale
emp=RadialEmpiricalSource(mode="ecdf").fit(train)
s=emp.sample(4000,d); ks=_ks_stat(s.norm(dim=1),train.norm(dim=1))
check("RAFM/matched empirical radii ~ train radii", ks<0.05, f"KS {ks:.3f}")
R0=float(train.norm(dim=1).mean())
proj=(R0*train/train.norm(dim=1,keepdim=True))
empf=RadialEmpiricalSource(mode="ecdf").fit(proj); sf_=empf.sample(2000,d)
check("fixed_spherical source radius == R0 (point mass)", float((sf_.norm(dim=1)-R0).abs().max())<1e-2*R0)
g=GaussianSource().sample(4000,d)
check("gaussian source ~ unit variance", abs(float(g.std())-1.0)<0.05, f"std {float(g.std()):.3f}")

# ---- preprocessing inverse (scaled + centering) ----
raw=torch.randn(100,d)
mu=raw[:60].mean(0)                      # train-only mean
scaled=raw*SF; z=scaled-mu               # model space
recovered_raw=(z+mu)/SF                  # inverse in reverse order
check("preproc inverse recovers raw latent (exact)", torch.allclose(recovered_raw,raw,atol=1e-5),
      f"max|dev| {float((recovered_raw-raw).abs().max()):.2e}")

print("\n"+("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
