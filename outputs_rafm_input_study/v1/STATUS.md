# Authorized matched study — launched 2026-09-10

Source repository: `foubari/Radial_Angular_Flow_Matching`, branch `experiments/tflow-empirical-gain`, experiment source commit `f23c838` (full hash in the launch manifests). Existing paper results remain unchanged.

| Study | Compute node | Slurm array | Work |
|---|---|---|---|
| t-Flow validation-only source selection | auh7-3b-gpu-008 | 765741 | 26 conditions, fixed nine-candidate budget |
| t-Flow final training/evaluation | auh7-3b-gpu-008 | 765742 | 78 final seeds; ready vector workers overlap audio tuning |
| RAFM-Ang A/B/C | auh7-3b-gpu-015 | 765748 | 234 final condition/arm/seed runs |
| Automatic tables/plots/accounting | compute node, no GPU | 765757 | Runs after both final arrays end, including failures |

Each study uses at most six single-MI210 workers. A worker executes independent models sequentially and releases its allocation when its shard ends. There is no model sharding. Exact commands, source/config/data hashes, seeds, locks and tuning-selection receipts are recorded in the launch/execution directories. Do not repeat a full `--submit`: duplicate submissions are deliberately refused.

All 26 t-Flow sanity runs passed, with 95 unit tests and 51 subtests. All 26 A/B/C sanity runs passed, with 50 tests and all nine full-size backbone checks. Fresh public-trainer checks passed for audio and vectors, including bitwise-identical resumed versus uninterrupted model/EMA checkpoints. The disposable checks are separate from final training and tuning.

Two implementation/startup failures were fixed and preserved: a CPU-fixture GPU-memory logging guard (job 765729), and a fresh-process ROCm allocator initialization error (cancelled t-Flow v1 arrays 765731/765732). The latter occurred before model/optimizer construction: five candidate failure records, no optimizer updates, no checkpoints. Current t-Flow outputs are `outputs_tflow_full/v2`; v1 remains an audit record. No model, objective, precision, split, seed or budget was changed to address either error.

The shared study has 28 requested conditions. Twenty-six are launchable. **PIV d32** still needs the authoritative native tensor/ordering; **ImageNette DC-AE** still needs the paper's actual generator split and image/reference mapping. Their 24 final runs remain explicitly blocked. New synthetic tensors are explicitly seeded shared realizations, not claimed to be historical caches; A/B/C and t-Flow use these same new files.

Current status and completed metrics: [report/README.md](report/README.md), [report/report.json](report/report.json). Scheduler observations are timestamped under `scheduler_snapshots/`. Run `/usr/bin/python3 tools/report_shared_study.py` from the repository to refresh the lightweight report. Scientific figures are generated on compute by job 765757.

The verified fixed-spherical + empirical-gain audio reference remains separate: measured accuracy 0.8066667 ± 0.00735225 (population SD), with zero changed digit predictions across 6,000 paired outputs. Its discrepancy from the paper's 0.810 ± 0.013 is preserved. No new A/B/C or t-Flow quality conclusion is available until complete seed groups finish.

Orchestration update: t-Flow final workers 3/4/5 (39 tasks with verified frozen selections) start after the short saved-audio backend audit 765758; workers 0/1/2 retain the complete-tuning dependency. No configuration or training budget changed. The failed first backend audit 765756 is preserved; its writable MIOpen cache was corrected for retry.

Saved-audio reevaluation job 765758 completed all 3 seeds in 9.62 measured seconds. All 12,000 cached-versus-current classifier predictions and all 6,000 original-versus-rescaled predictions agree. Accuracy, energy KS, and q90/q95/q99/low-tail coverage are exactly unchanged. Its strict archival-identity status remains `compatibility_discrepancy`: the new CPU norm path changes posthoc PIT by approximately -5e-7 and radial W1 by at most 1.42e-6; nearly equal original radii also make the energy-bin assignment sensitive to reduction rounding. Comparisons use the separately labelled current-backend measurements, while the old artifact and this discrepancy are preserved.

Measured current audio training throughput is approximately 0.37 seconds/update on an MI210, or 2.5 hours/24k-step seed before evaluation. The initial full 26-condition wall-time estimate is 6–8 hours from launch (not a completed runtime measurement), with data/protocol-blocked conditions excluded from that estimate.
