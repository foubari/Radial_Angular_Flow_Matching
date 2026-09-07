@echo off
setlocal
cd /d "%~dp0\..\.."
set "PYTHONIOENCODING=utf-8"
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "A=experiments\poc_audio\audio_eval.py"
set "CFG=--dataset libri_word --ncls 20 --clf libri_word_classifier.pt --step 5000 --nfe 6 --n 300 --whisper 1 --nwhisper 300 --sbatch 64"
echo ===== matched_euclidean =====
"%PY%" "%A%" --method matched_euclidean --run_dir runs_libri5 %CFG% || exit /b 1
echo ===== rafm (std) =====
"%PY%" "%A%" --method rafm --run_dir runs_libri5 %CFG% || exit /b 1
echo ===== rafm (angular) =====
"%PY%" "%A%" --method rafm --run_dir runs_libri5_ang %CFG% || exit /b 1
echo LIBRI5 EVAL DONE
