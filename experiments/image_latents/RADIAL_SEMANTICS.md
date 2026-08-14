# Radial-semantics analysis (CPU-only, run alongside training)

Model space = DC-AE latent × 0.41407 (official scaling) then train-only centering (seed-0 60/20/20
split) — the exact space where RAFM's empirical radial source is fit. GPU hidden, 1 thread, 1 process,
no neural model, no decode. n=13 394 (train 8 036), D=2 048.

## 1. Radial law (train)
- radius **mean 41.67, std 4.89, CoV 0.117** — a *non-degenerate* radial spread (contrast DINO/RAE
  ~1.6% CoV). This is the regime where a matched-radial source can matter.
- quantiles: q05 34.89 · q50 41.06 · q95 51.13 · q99 56.57 (min 27.46, max 61.96).

## 2. Radial-quantile index groups (saved → `radial_semantics/radial_quantile_indices.npz`)
Thresholds from the **train** law, applied to the full dataset:
bottom05=651 · bottom10=1336 · median45_55=1319 · top10=1340 · top05=681 · top01=149.
(Ready for uncurated image grids and interventions — deferred scripts.)

## 3–4. Class structure of the radius
- **η² = 0.162** — the class label explains ~16% of radius variance: modest but non-trivial, i.e. the
  scalar radius carries *some* class signal (not a pure nuisance scale, not class-defining either).
- classes low→high mean radius: tench, English-springer, church, garbage-truck, gas-pump, French-horn,
  cassette-player, golf-ball, parachute, chain-saw *(folder order n01440764…; names indicative)*.
- per-class radius mean/std/q05/q50/q95 in `radial_semantics.json`.

## 5. Structure (randomized PCA, bounded subsample n=4000, 50 comp)
- top-10 PCs explain only 0.224 of variance, 50 PCs 0.357 → latents are genuinely high-dimensional,
  no low-rank collapse.
- **max |corr(radius, PC)| = 0.307 @ PC3** → radius is a *distributed* property, not aligned with any
  single principal direction. Consistent with radius being a global magnitude rather than one feature.

## Takeaway
The DC-AE latent radius is a real, spread-out, mildly class-correlated, distributed quantity — which is
why matching its law (RAFM) is a *sensible* target, and also why the effect is subtle (radius is not the
dominant axis of variation). This sets up the deferred perceptual tests (does radius ↔ image rarity?
does rescaling radius change the decoded image?).

Deferred (GPU/heavy, prepared not run — see `dit/deferred/README_DEFERRED.md`): uncurated grids by
quantile, radius-intervention decode, DINO/Inception rarity + NN, radius-prediction R²/class-from-radius,
generated radial-tail calibration/coverage.
