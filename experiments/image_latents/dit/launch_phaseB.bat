@echo off
setlocal
REM ============================================================================
REM  Phase B launcher - 4 methods x 3 seeds to 40k steps (SiT, scaled DC-AE).
REM  RESUMABLE: safe to double-click / re-run any time. One process at a time.
REM  Each trainer call resumes from its own complete-state checkpoint; a run that
REM  already reached 40k exits instantly, an interrupted run continues bit-exactly.
REM  If the VM stops mid-run, just run this file again - it picks up where it left off.
REM  NO background respawn (that caused the earlier RAM/pagefile crash).
REM ============================================================================
cd /d "%~dp0\..\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "TRAIN=experiments\image_latents\dit\dit_train_sit.py"
set "STEPS=40000"

for %%S in (8925 1234 7) do (
  for %%M in (gaussian_euclidean matched_euclidean fixed_spherical rafm) do (
    echo ==================================================================
    echo   seed %%S  method %%M   ^(resumes if a checkpoint exists^)
    echo ==================================================================
    "%PY%" "%TRAIN%" --method %%M --seed %%S --split_seed 0 --out experiments/image_latents/dit_sit/seed_%%S --steps %STEPS% || (echo !! run stopped seed %%S method %%M - re-run this .bat to resume. & exit /b 1)
  )
)
echo(
echo  ALL 12 RUNS COMPLETE ^(4 methods x 3 seeds @ 40k^).
