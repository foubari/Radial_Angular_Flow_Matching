@echo off
setlocal
REM LibriSpeech word-conditioned pilot: matched / std-RAFM / angular-RAFM, 1 seed, UNet ch96, 24k. Resumable.
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "A=experiments\poc_audio\audio_flow.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CFG=--arch unet --ch 96 --batch 32 --ncls 20 --data experiments/poc_audio/data/libri_word_train.pt --steps 24000 --ckpt_every 2000 --seed 8925"
echo ====== matched ======
"%PY%" "%A%" --method matched_euclidean %CFG% --out experiments/poc_audio/runs_libri || (echo !! stopped matched & exit /b 1)
echo ====== rafm-std ======
"%PY%" "%A%" --method rafm %CFG% --out experiments/poc_audio/runs_libri || (echo !! stopped rafm & exit /b 1)
echo ====== rafm-angular ======
"%PY%" "%A%" --method rafm --angular %CFG% --out experiments/poc_audio/runs_libri_ang || (echo !! stopped angular & exit /b 1)
echo ALL LIBRI RUNS DONE
