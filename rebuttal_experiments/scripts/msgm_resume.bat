@echo off
REM ============================================================
REM  MSGM resume launcher (finance + weather, 1 seed each).
REM  RESUMABLE: run_msgm_real skips completed runs and resumes
REM  partial ones from msgm_ckpt.pt (checkpoint every 1000 steps).
REM  SELF-HEALING: loops and retries if a run crashes.
REM  Double-click this file to (re)start/continue MSGM training
REM  independently of any terminal or Claude session.
REM ============================================================
cd /d C:\Users\Shadow\Desktop\Radial_Angular_FM
set PY=C:\Users\Shadow\miniforge3\envs\dgm_tire\python.exe
set LOG=rebuttal_experiments\logs\msgm_resume.log
echo MSGM_RESUME_START %DATE% %TIME% >> %LOG%

set FIN=rebuttal_experiments\raw_results\E_finance\finance_ff49\msgm\seed_8925\metrics.json
set WEA=rebuttal_experiments\raw_results\E_weather\weather_au_wind\msgm\seed_8925\metrics.json

:loop
if exist "%FIN%" if exist "%WEA%" goto done

echo [finance seed 8925] %DATE% %TIME% >> %LOG%
%PY% rebuttal_experiments\scripts\run_msgm_real.py --pt rebuttal_experiments/data/finance/finance_returns_raw.pt --name finance_ff49 --out rebuttal_experiments/raw_results/E_finance --steps 10000 --batch 4096 --seeds 1 --n_gen 10000 >> %LOG% 2>&1

echo [weather seed 8925] %DATE% %TIME% >> %LOG%
%PY% rebuttal_experiments\scripts\run_msgm_real.py --pt rebuttal_experiments/data/weather/weather_au_wind.pt --name weather_au_wind --out rebuttal_experiments/raw_results/E_weather --steps 10000 --batch 2048 --seeds 1 --n_gen 10000 >> %LOG% 2>&1

if exist "%FIN%" if exist "%WEA%" goto done
echo retry in 15s (a run exited before finishing) %DATE% %TIME% >> %LOG%
timeout /t 15 /nobreak >nul
goto loop

:done
echo MSGM_RESUME_ALL_DONE %DATE% %TIME% >> %LOG%
