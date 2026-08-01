#!/usr/bin/env bash
# Stage B — P1 sweeps. Run from repo root. Logs to rebuttal_experiments/logs/stageB.log
set -e
cd "$(dirname "$0")/../.."
LOG=rebuttal_experiments/logs/stageB.log
echo "STAGE_B_START $(date)" > $LOG

# E6 dimension scaling (Student-t df3): 2,8,16,32,64,128,256
for d in 2 8 16 32 64 128 256; do
  echo "[E6 dim=$d] $(date)" >> $LOG
  python -m experiments.exp1_main_benchmark --config rebuttal_experiments/configs/E6_dim${d}.yaml >> $LOG 2>&1
done

# E8 tail-heaviness (Student-t d16): df sweep
for df in 1.5 2.0 3.0 5.0 10.0 50.0; do
  echo "[E8 df=$df] $(date)" >> $LOG
  python -m experiments.exp1_main_benchmark --config rebuttal_experiments/configs/E8_df${df}.yaml >> $LOG 2>&1
done

# E10 PIV d64 (real data) FM ladder
echo "[E10 PIV d64] $(date)" >> $LOG
python -m experiments.exp1_main_benchmark --config rebuttal_experiments/configs/E10_piv_d64.yaml >> $LOG 2>&1

echo "STAGE_B_DONE $(date)" >> $LOG
