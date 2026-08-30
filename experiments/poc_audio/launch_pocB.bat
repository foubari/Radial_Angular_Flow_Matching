@echo off
setlocal
REM PoC-B UNet sweep: 4 methods x 1 seed, UNet ch=96 (27.9M), batch 32, 24k steps (=768k examples, matches pilot).
REM Checkpoints every 2k (12k/18k/24k saved). RESUMABLE (re-run to continue). One process at a time.
REM Identical arch/init/optimizer/conditioning/data-order across methods (same seed); only source+path differ.
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "T=experiments\poc_audio\audio_flow.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for %%M in (gaussian_euclidean matched_euclidean fixed_spherical rafm) do (
  echo ================= %%M =================
  "%PY%" "%T%" --method %%M --arch unet --ch 96 --batch 32 --steps 24000 --ckpt_every 2000 --seed 8925 --out experiments/poc_audio/runs_unet || (echo !! stopped %%M - re-run to resume & exit /b 1)
)
echo ALL POC-B UNET RUNS DONE
