@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM  Phase B EVALUATION - run ONLY after training is complete and the GPU is free.
REM  Standardized evaluator (torch-fidelity FID/KID + prdc + latent metrics), CFG=1.
REM  Evaluates every method x seed at 40k first (headline), then 30k, 20k (trajectory).
REM  Re-runnable: an eval whose JSON already exists is skipped. One process at a time.
REM  A shared 5k-image real reference is built once on the first eval and cached.
REM ============================================================================
cd /d "%~dp0\..\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "EVAL=experiments\image_latents\dit\dit_eval_sit.py"
set "N=3000"
set "REALN=3925"
REM  N = generated samples per eval; REALN = real-ref size (= all Imagenette val imgs, so the
REM  reference is BUILT ONCE and CACHED — with real_n>val_count it rebuilt every eval).

for %%T in (40000 30000 20000) do (
  for %%S in (8925 1234 7) do (
    for %%M in (gaussian_euclidean matched_euclidean fixed_spherical rafm) do (
      set "OUT=experiments\image_latents\dit_sit\seed_%%S\eval_std\%%M\eval_%%T.json"
      set "EMA=experiments\image_latents\dit_sit\seed_%%S\runs\%%M\ema_%%T.pt"
      if not exist "!EMA!" (
        echo  -- skip ^(no checkpoint^): seed %%S %%M step %%T
      ) else if exist "!OUT!" (
        echo  -- skip ^(already evaluated^): seed %%S %%M step %%T
      ) else (
        echo ==================================================================
        echo   EVAL seed %%S  method %%M  step %%T
        echo ==================================================================
        "%PY%" "%EVAL%" --method %%M --run_root experiments/image_latents/dit_sit/seed_%%S --step %%T --n %N% --real_n %REALN%
        if errorlevel 1 (echo  !! eval failed seed %%S %%M step %%T - re-run to continue. & exit /b 1)
      )
    )
  )
)
echo(
echo  ALL EVALS COMPLETE. Aggregate with: %PY% experiments\image_latents\dit\aggregate_results.py
