@echo off
setlocal
REM Complete Angular RAFM across ALL previously-run experiments still missing it.
REM exp1 config families (angular only) + run_real families (aniso/randomsplit/bigmodel).
REM Both runners skip-if-metrics.json-exists -> fully resumable if VM dies. One process.
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "RR=rebuttal_experiments\scripts\run_real.py"

REM ---- exp1 config families: student-t dim/df sweeps, gaussian-aniso ctrl, toy2d, d32 ----
for %%C in (E1_gaussian_d16 E1_studentt_d32 E5_toy2d E6_dim2 E6_dim8 E6_dim16 E6_dim32 E6_dim64 E6_dim128 E6_dim256 E8_df1.5 E8_df2.0 E8_df3.0 E8_df5.0 E8_df10.0 E8_df50.0) do (
  echo ====== exp1 angular %%C ======
  "%PY%" -m experiments.exp1_main_benchmark --config rebuttal_experiments/configs/%%C.yaml --method angular_rafm || exit /b 1
)

REM ---- run_real: E9 anisotropy sweep k1..k300 (6) ----
for %%K in (1 3 10 30 100 300) do (
  echo ====== aniso k%%K ======
  "%PY%" "%RR%" --pt rebuttal_experiments/data/aniso/aniso_gauss_d32_k%%K.pt --name aniso_k%%K --out rebuttal_experiments/raw_results/E9_aniso --nfe 512 --seeds 3 --methods angular_rafm || exit /b 1
)

REM ---- finance random-split diagnostic ----
echo ====== finance randomsplit ======
"%PY%" "%RR%" --pt rebuttal_experiments/data/finance/finance_returns_raw.pt --name finance_ff49 --out rebuttal_experiments/raw_results/E_finance_randomsplit --split random --nfe 512 --seeds 3 --methods angular_rafm || exit /b 1

REM ---- weather big-model capacity ablation (Residual MLP 256x4 = 576096 params) ----
echo ====== weather bigmodel ======
"%PY%" "%RR%" --pt rebuttal_experiments/data/weather/weather_au_wind.pt --name weather_au_wind --out rebuttal_experiments/raw_results/E_weather_bigmodel --arch resmlp --hidden_dim 256 --n_layers 4 --batch 2048 --nfe 512 --seeds 3 --methods angular_rafm || exit /b 1

echo ALL ANGULAR SUITE DONE
