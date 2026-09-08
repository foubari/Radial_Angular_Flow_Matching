#!/usr/bin/env bash
# MSGM-SPARSE on the tabular/low-dim datasets, through the SAME exp1 harness as every other method.
# Isolated: writes to outputs_msgm_sparse/ (never the paper's outputs/). 7 datasets x 3 seeds {8925,77395,
# 65457}, protocol from configs/defaults.yaml (10k steps, batch 256, nfe 128) — identical to dense MSGM.
# Resumable: each seed skips if its metrics.json exists. Run different configs on different GPUs to
# parallelize (each config runs its 3 seeds sequentially on one GPU).
set -euo pipefail
PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CFGS=(gaussian_d16 studentt_d16 studentt_d32 piv_d16 piv_d64 piv_d256 toy2d)
for c in "${CFGS[@]}"; do
  echo "===== msgm_sparse: $c (3 seeds) ====="
  $PY -m experiments.exp1_main_benchmark --config "configs/exp1_sparse/${c}.yaml" --method msgm_sparse
done
echo "ALL MSGM-SPARSE TABULAR DONE -> outputs_msgm_sparse/exp1_main_benchmark/<dataset>/msgm_sparse/seed_*/metrics.json"
