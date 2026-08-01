@echo off
REM DiT four-way pilot on DC-AE latents (Imagenette-10). Resumable + self-healing.
REM Double-click to (re)start; survives terminal/session teardown.
cd /d C:\Users\Shadow\Desktop\Radial_Angular_FM
set PY=C:\Users\Shadow\miniforge3\envs\dgm_tire\python.exe
set LOG=experiments\image_latents\dit\pilot.log
set STEPS=20000
echo PILOT_START %DATE% %TIME% >> %LOG%
:loop
for %%M in (gaussian_euclidean matched_euclidean fixed_spherical rafm) do (
  echo [%%M] %DATE% %TIME% >> %LOG%
  %PY% experiments\image_latents\dit\dit_train.py --method %%M --steps %STEPS% --batch 64 --ckpt_every 2000 --log_every 500 --out experiments/image_latents/dit >> %LOG% 2>&1
)
if exist experiments\image_latents\dit\runs\rafm\ema_20000.pt if exist experiments\image_latents\dit\runs\gaussian_euclidean\ema_20000.pt if exist experiments\image_latents\dit\runs\fixed_spherical\ema_20000.pt if exist experiments\image_latents\dit\runs\matched_euclidean\ema_20000.pt goto done
timeout /t 15 /nobreak >nul
goto loop
:done
echo PILOT_DONE %DATE% %TIME% >> %LOG%
