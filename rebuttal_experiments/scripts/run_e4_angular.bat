@echo off
setlocal
REM E4 NFE x solver x tangent-projection sweep for Angular RAFM checkpoints (sampler-only, no retrain).
REM Mirrors the retained E4 CSVs (d16, toy2d, piv, weather) that already exist for gaussian + std-RAFM.
cd /d "%~dp0\..\.."
set "PY=experiments\image_latents\.venv_img\Scripts\python.exe"
set "E4=rebuttal_experiments\scripts\E4_nfe_solver_sweep.py"
set "O=rebuttal_experiments/raw_results/E4_nfe_solver"
"%PY%" "%E4%" --run_dir rebuttal_experiments/raw_results/E1_reproduction/student_t_d16_df3.0_cor/angular_rafm/seed_8925 --path spherical_geodesic --angular --out %O%/angular_rafm_d16.csv || exit /b 1
"%PY%" "%E4%" --run_dir rebuttal_experiments/raw_results/E5_toy2d/toy_radial_angular/angular_rafm/seed_8925 --path spherical_geodesic --angular --out %O%/angular_rafm_toy2d.csv || exit /b 1
"%PY%" "%E4%" --run_dir rebuttal_experiments/raw_results/E10_piv/piv_d64/angular_rafm/seed_8925 --path spherical_geodesic --angular --dataset_pt data/piv/piv_d64.pt --out %O%/piv_angular.csv || exit /b 1
"%PY%" "%E4%" --run_dir rebuttal_experiments/raw_results/E_weather/weather_au_wind/angular_rafm/seed_8925 --path spherical_geodesic --angular --dataset_pt rebuttal_experiments/data/weather/weather_au_wind.pt --out %O%/weather_angular.csv || exit /b 1
echo ALL E4 ANGULAR DONE
