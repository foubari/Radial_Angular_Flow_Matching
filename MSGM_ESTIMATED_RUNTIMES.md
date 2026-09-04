# MSGM estimated run times

Measured MSGM training time is ~dim-independent (1.87–2.01 s/step over d=49..256, 10k steps, RK4
Stratonovich, SSM double-backward), so the binding constraint for the un-run experiments is the dense
(d,d,d) generator memory G = 4·d³ bytes, not compute.

| Experiment | d | runs | status | per-seed time | total | note / G size |
|---|---|---|---|---|---|---|
| Finance | 49 | 3 | measured | 5.18 h (1.87 s/step) | 15.5 h | feasible; G=0.5 MB |
| Weather | 96 | 3 | measured | 5.58 h (2.01 s/step) | 16.7 h | feasible; G=3.4 MB |
| PIV | 256 | 3 | measured | 5.39 h (1.94 s/step) | 16.2 h | feasible; G=67 MB |
| Student-t tail sweep | 16 | 6×3=18 | estimated | ~5.0–5.3 h | ~90–95 GPU-h | time only; G=16 KB |
| ImageNette / DC-AE | 2048 | 3 | memory-infeasible | — | — | G=34.4 GB > 15 GB VRAM |
| AudioMNIST | 16254 | ≥3 | memory-infeasible | — | — | G=17.2 TB, cannot instantiate |
