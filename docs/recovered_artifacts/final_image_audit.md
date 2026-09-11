# Final ImageNette result audit

All 12 final runs passed the independent read-only audit: three seeds each for original RAFM-Ang (A), direction plus radius (B), direction with constant radius input (C), and t-Flow. The audit hashed 3,996 current files, including all checkpoint/sample artifacts, pinned implementation sources, input files and every one of the 3,925 FID-reference PNGs. No mismatches were found. No model was loaded or executed.

| Method | FID ↓, mean ± population SD | KID ↓, mean ± population SD |
|---|---:|---:|
| A | 147.9571 ± 0.3780 | 0.095769 ± 0.000055 |
| B | 311.4028 ± 11.0580 | 0.318479 ± 0.021294 |
| C | 227.4368 ± 67.7244 | 0.198235 ± 0.106321 |
| tflow | 223.7961 ± 6.2719 | 0.090255 ± 0.006147 |

All current sample audits match their result JSON SHA-256 and report finite, class-balanced samples. Training logs, training statistics and checkpoint result metadata agree on the prescribed final 40,000 updates. Checkpoint file hashes match the recorded final checkpoints; this audit did not deserialize them. The 12 runs share the same cached centered data, split indices, SiT backbone settings, batch 64, AdamW, BF16 training, EMA 0.9999, 3,000 samples, seed 0 and 100 network evaluations per trajectory. Each used a single AMD MI210.

A has lower FID than t-Flow on every seed, but t-Flow has slightly lower mean KID. B has zero recall on every seed. C varies strongly across seeds (seed 1234 FID 323.1638). t-Flow samples are finite but have much larger latent radial errors: W1 405.5948 ±66.3700 and KS 0.673867 ±0.089583. These observations do not establish statistical significance or general method rankings.

The original ImageNette evaluation metrics are complete for all 12 runs. Additional conditional angular diagnostics appear for ABC but not t-Flow. The original image evaluator (`experiments/image_latents/dit/dit_eval_sit.py:110`) and frozen IMAGE completion schema do not require them; ABC adds `rafm.metrics.angular.angular_metrics` at `experiments/rafm_inputs/run.py:311`. The t-Flow image branch at `experiments/tflow/run.py:591` omits this added diagnostic. Computing it from saved samples remains additional evaluation work; no frozen results were altered.

The FID reference retains the authorized audited overlap: 2,339 generator-training images, 775 validation and 811 test. It is not a held-out reference. The aggregate PNG digest is `79064131fa4cad5529be9ba8b37d3da62a2002adb1b011cfc6bdf148b25382b7`. Historical extraction/training provenance remains limited: `d3006dc8…` is the checkout holding the supplied artifacts, not a verified historical training commit.

The common downstream evaluator writes a nested `method: tflow` value even for ABC. The verified top-level `method` and `arm` identify the actual models.

Per-seed identities, current hashes, all 15 check outcomes, metrics, settings and runtime records are in [final_image_audit.json](final_image_audit.json). Under the repository root, source results are `outputs_rafm_input_study/v1/final/imagenette_dcae/{A,B,C}/seed_{8925,1234,7}/` and `outputs_tflow_full/v2/final/imagenette_dcae/seed_{8925,1234,7}/`.
