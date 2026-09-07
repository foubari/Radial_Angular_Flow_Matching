@echo off
setlocal
REM Stage 3: scale-free ANGULAR RAFM (predict A=v/||x_t||, reconstruct v=||x||*A). Same UNet/seed/budget as
REM standard rafm; only the regression target differs. Resumable. -> runs_angular/rafm
cd /d "%~dp0\..\.."
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
experiments\image_latents\.venv_img\Scripts\python.exe experiments\poc_audio\audio_flow.py --method rafm --angular --arch unet --ch 96 --batch 32 --steps 24000 --ckpt_every 2000 --seed 8925 --out experiments/poc_audio/runs_angular || (echo !! stopped - re-run to resume & exit /b 1)
echo ANGULAR RAFM DONE
