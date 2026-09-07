# RAFM-Ang — Table-1 completion (protocol-matched)

Protocol: batch 256, 10k steps, n_gen 10000, RK4 **128 steps = 512 NFE**, seeds {8925, 77395, 65457}. Metrics lower = better.

## PIV d=64 — protocol-matched (rerun)

Dataset: n_train=598, n_test=201, dim=64, split_seed=0 (exp1 PIVDataset _make_splits); train md5 `5d0d421eb0c3c5c9e9b4d16b`, test md5 `62bd077c8ad2246a60a47914`. nfe stored = 512.

### Radial W1

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 0.04565 / 0.04899 / 0.05002 | **0.04822±0.00187** |
| RAFM-Vel (ref) | 0.04565 / 0.04899 / 0.05002 | **0.04822±0.00187** |

### Radial KS

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 0.04322 / 0.04932 / 0.04818 | **0.04691±0.00265** |
| RAFM-Vel (ref) | 0.04322 / 0.04932 / 0.04818 | **0.04691±0.00265** |

### Sliced W1

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 0.03079 / 0.02471 / 0.02819 | **0.0279±0.00249** |
| RAFM-Vel (ref) | 0.03025 / 0.02335 / 0.02825 | **0.02728±0.0029** |

### Train time (s)

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 42.93965 / 42.31964 / 43.81353 | **43.02427±0.61281** |
| RAFM-Vel (ref) | 35.7195 / 35.6925 / 36.112 | **35.84133±0.19171** |

## PIV d=256 — protocol-matched (already run; protocol verified)

Dataset: n_train=598, n_test=201, dim=256, split_seed=0 (exp1 PIVDataset _make_splits); train md5 `8c997feba59290785fcb915a`, test md5 `7ef7ee53bb00810bffb1442e`. nfe stored = 512.

### Radial W1

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 0.03865 / 0.03555 / 0.03696 | **0.03705±0.00127** |
| RAFM-Vel (ref) | 0.03865 / 0.03555 / 0.03696 | **0.03705±0.00127** |

### Radial KS

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 0.05828 / 0.05936 / 0.05596 | **0.05787±0.00142** |
| RAFM-Vel (ref) | 0.05828 / 0.05936 / 0.05596 | **0.05787±0.00142** |

### Sliced W1

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 0.0231 / 0.02234 / 0.02262 | **0.02268±0.00032** |
| RAFM-Vel (ref) | 0.02359 / 0.02083 / 0.02812 | **0.02418±0.00301** |

### Train time (s)

| method | per-seed (8925/77395/65457) | mean±std |
|---|---|---|
| RAFM-Ang | 67.65989 / 65.08598 / 63.77769 | **65.50785±1.61273** |
| RAFM-Vel (ref) | 35.4525 / 35.306 / 35.479 | **35.4125±0.07608** |

## Student-t ν=3, d={16,32} — BLOCKED (not run)

Table-1 Student-t realization is not recoverable: StudentT._generate draws z from the UNSEEDED global torch RNG (only the mixing matrix A is seeded, matrix_seed=42); exp1_main_benchmark does not seed before dataset construction; no dataset .pt was cached. Two fresh processes give different test-set md5 (dea8f374... vs 250e7363...), same A (eb934f15...). Per instruction, stopped rather than evaluate on an incompatible realization.

The existing rebuttal Student-t angular runs used batch 4096 **and** a different (also non-recoverable) realization, so they are not protocol-matched to Table-1 either. No compatible RAFM-Ang Student-t result can be produced without re-running all baselines on a freshly-cached realization — out of scope per instruction.