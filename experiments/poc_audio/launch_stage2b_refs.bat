@echo off
setlocal
REM Stage 2b: complete 3-seed CIs for the REFERENCE methods gaussian + fixed_spherical (seed 8925 already
REM done). Adds seeds 1234 & 7. Same UNet/24k/batch32. Resumable. -> runs_s<seed>/<method>
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "A=experiments\poc_audio\audio_flow.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CFG=--arch unet --ch 96 --batch 32 --steps 24000 --ckpt_every 2000"
for %%S in (1234 7) do (
  for %%M in (gaussian_euclidean fixed_spherical) do (
    echo ============ seed %%S %%M ============
    "%PY%" "%A%" --method %%M %CFG% --seed %%S --out experiments/poc_audio/runs_s%%S || (echo !! stopped seed %%S %%M - re-run to resume & exit /b 1)
  )
)
echo ALL STAGE-2B REF RUNS DONE
