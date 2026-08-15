@echo off
setlocal
cd /d "%~dp0\..\.."
set "PY=C:\Users\Shadow\miniforge3\envs\dgm_tire\python.exe"
"%PY%" rebuttal_experiments\scripts\run_real.py --pt data/piv/piv_d64.pt --name piv_d64 --out rebuttal_experiments/raw_results/E10_piv --steps 10000 --batch 4096 --nfe 512 --seeds 3 --methods angular_rafm || exit /b 1
"%PY%" rebuttal_experiments\scripts\run_real.py --pt rebuttal_experiments/data/finance/finance_returns_raw.pt --name finance_ff49 --out rebuttal_experiments/raw_results/E_finance --steps 10000 --batch 4096 --nfe 512 --seeds 3 --methods angular_rafm || exit /b 1
"%PY%" rebuttal_experiments\scripts\run_real.py --pt rebuttal_experiments/data/weather/weather_au_wind.pt --name weather_au_wind --out rebuttal_experiments/raw_results/E_weather --steps 10000 --batch 2048 --nfe 512 --seeds 3 --methods angular_rafm || exit /b 1
echo ALL ANGULAR REAL DONE
