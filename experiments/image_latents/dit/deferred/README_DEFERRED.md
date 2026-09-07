# Deferred analyses — RUN ONLY AFTER TRAINING FINISHES

These scripts were **prepared but NOT executed** while the Phase-B training sweep is running,
because each uses the GPU and/or a neural model (DC-AE / DINO / Inception) or heavy disk I/O and
would compete with the active training job. Do not run any of them until the sweep is complete
(all 12 runs at 40k, `dit_sit/seed_*/runs/*/ema_40000.pt` present) and the GPU is free.

Inputs already produced (CPU-safe, during training):
- `radial_semantics/radial_quantile_indices.npz` — per-sample radii, labels, train/val/test idx,
  and index groups (bottom05/10, median45_55, top10/05/01). Index `i` ↔ i-th path of
  `sorted(glob("data/imagenette2-320/**/*.JPEG"))` (see `extract_dcae_latents.py`).
- `radial_semantics/pca_subsample.npz` — bounded randomized-PCA scores/components + radii.

| script | needs | produces |
|---|---|---|
| `deferred_radial_grids.py` | CPU + many JPEG reads | uncurated image grids per radial quantile |
| `deferred_radial_intervention.py` | GPU + DC-AE | decode of radius-rescaled latents (radius→image effect) |
| `deferred_dino_rarity.py` | GPU + DINO/Inception | features, kNN rarity vs radius, nearest neighbours |
| `deferred_radius_regressor.py` | CPU (after features) | radius-prediction R², class-from-radius accuracy |
| `deferred_tail_calibration.py` | GPU (sampling from trained EMA) | generated vs data radial-tail coverage/calibration |

Suggested order after training: grids → intervention → dino_rarity → radius_regressor → tail_calibration.
Run one at a time, foreground, as with training (no background respawn).
