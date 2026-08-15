@echo off
setlocal enabledelayedexpansion
REM Angular RAFM on DC-AE latents (SiT backbone), 3 seeds x 40k, split_seed=0 (same split as existing 4 methods).
REM Resumable: training continues from ckpt.pt; eval skipped if eval_40000.json already exists. One process.
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "TR=experiments\image_latents\dit\dit_train_sit.py"
set "EV=experiments\image_latents\dit\dit_eval_sit.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for %%S in (8925 7 1234) do (
  set "ROOT=experiments/image_latents/dit_sit_s%%S"
  echo ====== TRAIN angular_rafm seed %%S ======
  "%PY%" "%TR%" --method angular_rafm --seed %%S --split_seed 0 --steps 40000 --out !ROOT! || exit /b 1
  if exist "experiments\image_latents\dit_sit_s%%S\eval_std\angular_rafm\eval_40000.json" (
    echo ====== EVAL seed %%S already done, skip ======
  ) else (
    echo ====== EVAL angular_rafm seed %%S ======
    "%PY%" "%EV%" --method angular_rafm --step 40000 --run_root !ROOT! --n 5000 || exit /b 1
  )
)
echo ALL ANGULAR DIT DONE
