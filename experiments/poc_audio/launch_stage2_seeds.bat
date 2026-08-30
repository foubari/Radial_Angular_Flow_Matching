@echo off
setlocal
REM Stage 2: seed uncertainty (3 seeds total: 8925 already done). Adds seeds 1234 & 7 for THREE methods:
REM   matched-Euclidean, standard RAFM, angular RAFM. Same UNet/24k/batch32. Resumable. One process at a time.
REM   out dirs: runs_s<seed> (matched + std rafm),  runs_s<seed>_ang (angular rafm).
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "A=experiments\poc_audio\audio_flow.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CFG=--arch unet --ch 96 --batch 32 --steps 24000 --ckpt_every 2000"
for %%S in (1234 7) do (
  echo ============ seed %%S matched ============
  "%PY%" "%A%" --method matched_euclidean %CFG% --seed %%S --out experiments/poc_audio/runs_s%%S || (echo !! stopped seed %%S matched & exit /b 1)
  echo ============ seed %%S rafm-std ============
  "%PY%" "%A%" --method rafm %CFG% --seed %%S --out experiments/poc_audio/runs_s%%S || (echo !! stopped seed %%S rafm & exit /b 1)
  echo ============ seed %%S rafm-angular ============
  "%PY%" "%A%" --method rafm --angular %CFG% --seed %%S --out experiments/poc_audio/runs_s%%S_ang || (echo !! stopped seed %%S angular & exit /b 1)
)
echo ALL STAGE-2 SEED RUNS DONE
