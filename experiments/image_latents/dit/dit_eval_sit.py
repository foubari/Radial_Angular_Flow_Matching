"""Standardized evaluator for a trained SiT flow (one method, one checkpoint).

Pipeline: sample latents (RK4 ODE, tangent projection for spherical, CFG=1 unguided, class-balanced)
-> undo train-centering -> decode with frozen DC-AE -> save PNGs -> metrics.

METRICS (all standardized / off-the-shelf, per reviewer):
  * FID, KID(+-std)  via **torch-fidelity** (its InceptionV3-FID feature space) — NOT a custom FID.
  * precision / recall / density / coverage via **prdc** on the SAME InceptionV3-pool features.
  * latent metrics (radial_w1, ks, sliced_w1, dir_sw1, cr_sw1) vs held-out centered TEST latents.
Real reference = a FIXED, cached folder of resized val images (seed-fixed); identical for every method.
rFID (decoder floor) = decode of real TEST latents vs the same real reference.

One process, foreground. Example:
  .venv_img/Scripts/python.exe dit/dit_eval_sit.py --method rafm --step 40000 --n 5000
"""
import argparse, json, sys, time, shutil, glob, os
from pathlib import Path
import numpy as np, torch

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"third_party/SiT"))
from models import SiT
from dit_train_sit import build, split_idx     # reuse the exact preprocessing
from rafm.flow_matching.sampler import _project_tangent
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import sliced_wasserstein

SF=0.41407
REAL_DIR=REPO/"experiments/image_latents/dit/eval_std/_real_ref"     # cached shared reference PNGs

def build_real_ref(img_root, n, size=256, seed=1):
    if REAL_DIR.exists() and len(list(REAL_DIR.glob("*.png")))>=n:
        print(f"[real] cached {REAL_DIR} ({len(list(REAL_DIR.glob('*.png')))} imgs)",flush=True); return
    from torchvision import transforms; from torchvision.utils import save_image; from PIL import Image
    ps=sorted(glob.glob(os.path.join(img_root,"**","*.JPEG"),recursive=True))
    rng=np.random.default_rng(seed); ps=[ps[i] for i in sorted(rng.permutation(len(ps))[:n])]
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(size),transforms.ToTensor()])
    REAL_DIR.mkdir(parents=True,exist_ok=True)
    for i,p in enumerate(ps): save_image(tf(Image.open(p).convert("RGB")), REAL_DIR/f"{i:05d}.png")
    print(f"[real] built {len(ps)} ref imgs -> {REAL_DIR}",flush=True)

@torch.no_grad()
def rk4_sample(model, x0, y, spherical, nfe=50):
    dev=x0.device; x=x0; n=x0.shape[0]; dt=1.0/nfe
    def v(xx,tt):
        out=model(xx.reshape(n,32,8,8),torch.full((n,),tt,device=dev),y).reshape(n,-1)
        return _project_tangent(out,xx) if spherical else out
    for i in range(nfe):
        t0=i*dt
        k1=v(x,t0); k2=v(x+dt/2*k1,t0+dt/2); k3=v(x+dt/2*k2,t0+dt/2); k4=v(x+dt*k3,t0+dt)
        x=x+dt/6*(k1+2*k2+2*k3+k4)
    return x

@torch.no_grad()
def decode_to_dir(latc, mu, ae, dev, out_dir, batch=64):   # batch only affects speed, not pixels
    from torchvision.utils import save_image
    out_dir.mkdir(parents=True,exist_ok=True); k=0
    for i in range(0,len(latc),batch):
        z=((latc[i:i+batch]+mu)/SF).to(dev).reshape(-1,32,8,8)      # undo centering+scaling -> raw
        im=((ae.decode(z).sample.clamp(-1,1)+1)/2).cpu(); del z
        for j in range(im.shape[0]): save_image(im[j], out_dir/f"{k:05d}.png"); k+=1
    return k

def incep_features(img_dir, dev, bs=128):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torchvision import transforms; from PIL import Image
    fe=FeatureExtractorInceptionV3("inception",["2048"]).to(dev).eval()
    ps=sorted(glob.glob(str(Path(img_dir)/"*.png"))); tf=transforms.PILToTensor(); feats=[]
    with torch.no_grad():
        for i in range(0,len(ps),bs):
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in ps[i:i+bs]]).to(dev)  # uint8
            feats.append(fe(x)[0].float().cpu().numpy())
    return np.concatenate(feats)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True,choices=["gaussian_euclidean","matched_euclidean","fixed_spherical","rafm"])
    ap.add_argument("--run_root",default="experiments/image_latents/dit_sit")
    ap.add_argument("--step",type=int,default=40000); ap.add_argument("--n",type=int,default=5000)
    ap.add_argument("--real_n",type=int,default=5000); ap.add_argument("--nfe",type=int,default=25)  # RK4 nfe25=100 fn-evals, ample for a trained flow; identical across all evals -> fair
    ap.add_argument("--latents",default="experiments/image_latents/data/dcae_latents_scaled.pt")
    ap.add_argument("--labels",default="experiments/image_latents/data/dcae_labels.pt")
    ap.add_argument("--img_root",default="experiments/image_latents/data/imagenette2-320/val")
    ap.add_argument("--split_seed",type=int,default=0); ap.add_argument("--sample_seed",type=int,default=0)
    ap.add_argument("--hidden",type=int,default=384); ap.add_argument("--depth",type=int,default=12); ap.add_argument("--heads",type=int,default=6)
    a=ap.parse_args(); torch.set_num_threads(2)
    dev="cuda" if torch.cuda.is_available() else "cpu"; t0=time.time()
    lat=torch.load(a.latents,map_location="cpu").float(); lab=torch.load(a.labels,map_location="cpu").long()
    st=build(a.method,lat,lab,a.split_seed); D=lat.shape[1]; mu=st["mu"]
    test=st["data"][st["te"]]                                       # held-out centered test latents
    # model
    model=SiT(input_size=8,patch_size=1,in_channels=32,hidden_size=a.hidden,depth=a.depth,num_heads=a.heads,
              num_classes=10,class_dropout_prob=0.1,learn_sigma=False).to(dev).eval()   # 0.1 -> null row, matches trained table; dropout inactive in eval()
    ema=torch.load(Path(a.run_root)/"runs"/a.method/f"ema_{a.step}.pt",map_location="cpu")["ema"]
    model.load_state_dict(ema)
    # class-balanced source draw (method-specific), fixed sample seed
    torch.manual_seed(a.sample_seed); np.random.seed(a.sample_seed)
    y=torch.arange(10).repeat_interleave(int(np.ceil(a.n/10)))[:a.n].to(dev)
    if a.method=="gaussian_euclidean":
        x0=torch.randn(a.n,D,device=dev)
    else:
        r=st["src"].sample(a.n,D).norm(dim=1,keepdim=True).to(dev)  # empirical radii
        u0=torch.randn(a.n,D,device=dev); x0=r*u0/u0.norm(dim=1,keepdim=True)
    # sample in minibatches
    gen=[]
    for i in range(0,a.n,512):
        gen.append(rk4_sample(model,x0[i:i+512],y[i:i+512],st["spherical"],a.nfe).cpu())
    genlat=torch.cat(gen)
    # --- latent metrics vs test ---
    rm=radial_metrics(genlat,test); rw1=float(rm["radial_w1"]); ks=float(rm["ks_stat"])
    torch.manual_seed(0); sw1=float(sliced_wasserstein(genlat,test,n_projections=200))
    ev=Path(a.run_root)/"eval_std"/a.method; ev.mkdir(parents=True,exist_ok=True)
    # --- decode + image metrics ---
    from diffusers import AutoencoderDC
    ae=AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",torch_dtype=torch.float32).to(dev).eval()
    build_real_ref(a.img_root,a.real_n)
    gen_dir=ev/f"gen_{a.step}";
    if gen_dir.exists(): shutil.rmtree(gen_dir)
    decode_to_dir(genlat,mu,ae,dev,gen_dir)
    del ae; torch.cuda.empty_cache()
    import torch_fidelity
    m=torch_fidelity.calculate_metrics(input1=str(gen_dir),input2=str(REAL_DIR),cuda=(dev=="cuda"),
        fid=True,kid=True,kid_subset_size=min(1000,a.n//2),batch_size=128,verbose=False)
    fid=float(m["frechet_inception_distance"]); kid=float(m["kernel_inception_distance_mean"]); kid_std=float(m["kernel_inception_distance_std"])
    # prdc on same inception-pool features
    from prdc import compute_prdc
    fg=incep_features(gen_dir,dev); fr=incep_features(REAL_DIR,dev)
    pr=compute_prdc(fr,fg,nearest_k=5)
    out={"method":a.method,"step":a.step,"n":a.n,"real_n":a.real_n,"nfe":a.nfe,"cfg":1.0,
         "latent":{"radial_w1":rw1,"ks":ks,"sliced_w1":sw1},
         "image":{"fid":round(fid,3),"kid":round(kid,5),"kid_std":round(kid_std,5),
                  "precision":round(pr["precision"],4),"recall":round(pr["recall"],4),
                  "density":round(pr["density"],4),"coverage":round(pr["coverage"],4)},
         "runtime_s":round(time.time()-t0,1)}
    (ev/f"eval_{a.step}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

if __name__=="__main__": main()
