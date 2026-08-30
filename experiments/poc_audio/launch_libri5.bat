@echo off
setlocal
REM LibriSpeech word pilot — FAST 5k version: matched / std-RAFM / angular, 1 seed, UNet ch96, 5k steps,
REM ckpt every 500 (fine resume + intermediate eval at 2k/3.5k/5k). Fresh dirs. Resumable.
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "A=experiments\poc_audio\audio_flow.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CFG=--arch unet --ch 96 --batch 32 --ncls 20 --data experiments/poc_audio/data/libri_word_train.pt --steps 5000 --ckpt_every 500 --log_every 500 --seed 8925"
echo ====== matched ======
"%PY%" "%A%" --method matched_euclidean %CFG% --out experiments/poc_audio/runs_libri5 || (echo !! stopped matched & exit /b 1)
echo ====== rafm-std ======
"%PY%" "%A%" --method rafm %CFG% --out experiments/poc_audio/runs_libri5 || (echo !! stopped rafm & exit /b 1)
echo ====== rafm-angular ======
"%PY%" "%A%" --method rafm --angular %CFG% --out experiments/poc_audio/runs_libri5_ang || (echo !! stopped angular & exit /b 1)
echo ALL LIBRI5 RUNS DONE
