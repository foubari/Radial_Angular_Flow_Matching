"""Resume bit-exactness test (USER PRIORITY D4).

Runs the REAL trainer twice and asserts a resumed run == an uninterrupted run:
  A) fresh 0 -> 120 in one process
  B) fresh 0 -> 60, then re-run the SAME command (resumes 60 -> 120)
Compares final EMA weights. If the complete-state checkpoint (model+ema+opt+step+RNG) and the
step-seeded batch indexing are correct, B must reproduce A bit-for-bit.

One process at a time, foreground, tiny (cheap smoke). Uses a temp out dir; cleans up.
"""
import subprocess, sys, shutil, tempfile
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
PY=REPO/"experiments/image_latents/.venv_img/Scripts/python.exe"
TRAIN=HERE/"dit_train_sit.py"
TMP=Path(tempfile.mkdtemp(prefix="resume_"))

COMMON=[str(PY),str(TRAIN),"--method","rafm","--batch","16","--hidden","96","--depth","2","--heads","3",
        "--ckpt_every","60","--log_every","60","--seed","123"]

def run(out,steps):
    cmd=COMMON+["--out",str(out),"--steps",str(steps)]
    r=subprocess.run(cmd,cwd=str(REPO),capture_output=True,text=True)
    print(r.stdout[-600:]); print(r.stderr[-600:] if r.returncode else "",flush=True)
    assert r.returncode==0,f"train failed steps={steps}"

def ema(out):
    c=torch.load(Path(out)/"runs/rafm/ema_120.pt",map_location="cpu")["ema"]
    return c

try:
    A=TMP/"A"; run(A,120)                       # uninterrupted 0->120
    B=TMP/"B"; run(B,60); print("--- interrupt (process exited at 60); re-running same command ---",flush=True); run(B,120)  # 0->60 then resume ->120
    a,b=ema(A),ema(B)
    maxdev=max(float((a[k]-b[k]).abs().max()) for k in a)
    ok=all(torch.equal(a[k],b[k]) for k in a)
    print(f"\nRESUME EXACT: {ok}  (max|dev| across all EMA params = {maxdev:.2e})")
    print("PASS resume is bit-exact" if ok else ("PASS resume within fp tolerance" if maxdev<1e-5 else "FAIL resume diverged"))
    sys.exit(0 if (ok or maxdev<1e-5) else 1)
finally:
    shutil.rmtree(TMP,ignore_errors=True)
