@echo off
REM ============================================================
REM  MSGM extra seeds (77395, 65457) for finance + weather.
REM  WAITS for the current weather seed 8925 to finish (no GPU
REM  contention with the running job), THEN runs the remaining
REM  seeds. run_msgm_real --seeds 3 skips completed seeds
REM  (8925) and resumes any partial one from msgm_ckpt.pt.
REM  SELF-HEALING loop; exits when all 3 seeds x 2 datasets done.
REM  Safe to launch now; does not touch the running process.
REM ============================================================
cd /d C:\Users\Shadow\Desktop\Radial_Angular_FM
set PY=C:\Users\Shadow\miniforge3\envs\dgm_tire\python.exe
set LOG=rebuttal_experiments\logs\msgm_extra_seeds.log
set WEA8925=rebuttal_experiments\raw_results\E_weather\weather_au_wind\msgm\seed_8925\metrics.json
set FIN_LAST=rebuttal_experiments\raw_results\E_finance\finance_ff49\msgm\seed_65457\metrics.json
set WEA_LAST=rebuttal_experiments\raw_results\E_weather\weather_au_wind\msgm\seed_65457\metrics.json

echo MSGM_EXTRA_WAIT_START %DATE% %TIME% >> %LOG%
:wait
if not exist "%WEA8925%" ( timeout /t 120 /nobreak >nul & goto wait )
echo current weather seed 8925 done; starting extra seeds %DATE% %TIME% >> %LOG%

:loop
if exist "%FIN_LAST%" if exist "%WEA_LAST%" goto done
echo [finance seeds 77395,65457] %DATE% %TIME% >> %LOG%
%PY% rebuttal_experiments\scripts\run_msgm_real.py --pt rebuttal_experiments/data/finance/finance_returns_raw.pt --name finance_ff49 --out rebuttal_experiments/raw_results/E_finance --steps 10000 --batch 4096 --seeds 3 --n_gen 10000 >> %LOG% 2>&1
echo [weather seeds 77395,65457] %DATE% %TIME% >> %LOG%
%PY% rebuttal_experiments\scripts\run_msgm_real.py --pt rebuttal_experiments/data/weather/weather_au_wind.pt --name weather_au_wind --out rebuttal_experiments/raw_results/E_weather --steps 10000 --batch 2048 --seeds 3 --n_gen 10000 >> %LOG% 2>&1
if exist "%FIN_LAST%" if exist "%WEA_LAST%" goto done
echo retry in 15s %DATE% %TIME% >> %LOG%
timeout /t 15 /nobreak >nul
goto loop

:done
echo MSGM_EXTRA_ALL_DONE %DATE% %TIME% >> %LOG%
