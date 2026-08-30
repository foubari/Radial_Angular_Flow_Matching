@echo off
setlocal
cd /d "%~dp0\..\.."
set "PYTHONIOENCODING=utf-8"
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "A=experiments\poc_audio\audio_eval.py"
set "CFG=--dataset libri_word --ncls 20 --clf libri_word_classifier.pt --step 5000 --nfe 40 --n 300 --whisper 1 --nwhisper 300 --sbatch 64"
echo ===== rafm std nfe40 =====
"%PY%" "%A%" --method rafm --run_dir runs_libri5 %CFG% || exit /b 1
echo ===== rafm angular nfe40 =====
"%PY%" "%A%" --method rafm --run_dir runs_libri5_ang %CFG% || exit /b 1
echo NFE40 DONE
