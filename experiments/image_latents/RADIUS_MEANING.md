# What does the DC-AE latent radius actually mean? — and why RAFM's radial win doesn't help images

Follow-up to the robust negative result (Gaussian FM best global FID at 20/30/40k; RAFM best latent
radial; not a solver/drift artefact). Goal: understand what the DC-AE latent radius encodes and whether
RAFM captures anything useful despite worse global FID. All analyses reuse the trained checkpoints,
cached latents, decoded generations, and DINOv2 features — **no retraining**. Reproducible + uncurated.

## TL;DR — supported conclusions

1. **The DC-AE latent radius encodes a low-level, secondary image property: background flatness /
   contrast (studio-vs-natural), not semantic content.** ✅ (Q1, Q3)
2. **RAFM calibrates the radial tail, but that tail is NOT perceptually rare** — radius is uncorrelated
   with perceptual (DINO) rarity. ✅ (Q2)
3. **The DC-AE decoder is largely insensitive to global radius**: changing only the radius shifts
   contrast/saturation, never semantics. ✅ (Q3)
4. **The single global radius is too coarse**: it is ~1.4 % of latent variance and near-chance for class;
   content lives in the direction / local structure. ✅ (Q6)
5. **RAFM does NOT better cover meaningful rare/extreme images** — [Q5, pending].

The unifying story: *RAFM perfectly matches the distribution of a scalar (the latent norm) that turns
out to control mainly image contrast / background flatness — a real but perceptually secondary property
the radius-tolerant decoder barely renders — so matching it cannot, and does not, improve decoded-image
quality or cover perceptually rare images.*

---

## Q1 — What do low / medium / high-radius real images look like?

Uncurated grids (fixed seed) at radial quantiles → `figures/radial_quantile_grids/`. Clear visual trend:

- **Low radius (bottom 5 %):** busy **natural photographs** — forests, grass, water, sky; textured
  backgrounds, moderate contrast.
- **Median:** mixed everyday photos.
- **High radius (top 5 %) / extreme (top 1 %):** **flat white/black-background studio & catalog product
  shots** — isolated high-contrast object, often text/logos ("SALE", price tags).

Quantified on 3000 images (`q1_lowlevel.json`), Pearson corr of radius with:

| property | corr | low-radius decile → high-radius decile |
|---|---|---|
| **flat_extreme_frac** (near-white∪near-black pixels) | **0.70** | 0.04 → **0.49** |
| white_frac | 0.58 | 0.02 → 0.31 |
| global contrast (pixel std) | 0.44 | 0.17 → 0.28 |
| black_frac | 0.33 | 0.03 → 0.17 |
| brightness | 0.18 | — |
| colour saturation / border-std | ~0.09 | — |
| **class** (η², from earlier) | 0.16 | weak |

→ **Latent radius ≈ background flatness / fraction of extreme pixels / contrast** ("product-catalog-ness"),
only weakly tied to class, essentially a low-level style axis. Figure: `radial_quantile_grids/*`.

## Q2 — Does latent radius correspond to perceptual rarity? — No.

Rarity = mean distance to 5 nearest neighbours in **DINOv2-B** feature space (independent of the latent).

- corr(radius, rarity) **global = −0.04**; within-class mean |corr| = **0.076** (all in [−0.20, +0.05], a
  few mildly negative). (`q2_rarity.json`, figure `figures/q2_radius_vs_rarity.png`.)

→ **Radius is unrelated to perceptual rarity, globally and within class.** The high-radius "tail" is a
*common* catalog/studio style, not a set of rare images. So calibrating the radial tail ≠ modelling rare
events.

## Q3 — Causal effect of changing only the radius (direction fixed, radius → real-data quantiles)

Grids `figures/radius_intervention/*`: across radius q05→q95 the **semantics are perfectly preserved**
(same object, layout, class); only **contrast/saturation** shifts (paler → punchier). Quantified
(`q3_measure.py`, 32 directions):

| radius | decoded contrast (pixel std) | mean \|Δpix\| vs q50 |
|---|---|---|
| q05 (R=34.9) | 0.254 | 0.038 |
| q50 (R=41.1) | 0.286 | 0.000 |
| q95 (R=51.1) | 0.329 | 0.053 |

→ Radius monotonically controls **contrast (+29 % q05→q95)** with a small global pixel change (~4–5 %)
and **zero semantic change**. Radius is a low-level contrast/saturation knob; **the decoder is largely
insensitive to global radius** semantically.

## Q6 — Is the global radius too coarse? — Yes.

DC-AE latent = 32 channels × 8×8 tokens = 2048-d; the radius is one scalar. On the centered latents
(`q6_local_structure.json`):

- **Global radius = 1.37 % of total latent variance.**
- **Class accuracy from the global norm alone = 0.17** (chance 0.10) vs **from the direction (radius
  removed) = 0.62** = **from the full latent = 0.61**. Channel-norms (32-d) already reach 0.46.
- Within-image token-norm CoV ≈ 0.16 → the magnitude field varies spatially; the global norm is a coarse
  spatial average.

→ **Content (class) lives in the direction / local structure; removing the global radius loses almost no
information.** The single global norm is a coarse 1-D summary — RAFM optimises exactly this coarse scalar.

## Q4 — Radial-tail calibration across the four methods — RAFM wins clearly

Generated radii vs the real (test) radial law; tail mass above real q95/q99 and PIT calibration
(`radial_semantics/tail_*.json`, n=3000, seed 8925):

| method | cov > q95 (target 0.05) | cov > q99 (target 0.01) | PIT mean (target 0.5) | gen radius mean (data 41.9) |
|---|---|---|---|---|
| gaussian_euclidean | 0.006 | 0.0007 | 0.230 | 37.3 (systematically small) |
| matched_euclidean | 0.016 | 0.0007 | 0.353 | 39.5 |
| fixed_spherical | 0.000 | 0.000 | 0.535 | 41.7 (point mass, no spread) |
| **rafm** | **0.045** | **0.0083** | **0.484** | 41.6 |

→ **RAFM is the only method that calibrates the radial tail** (cov ≈ target 0.05 / 0.01, PIT mean ≈ 0.5).
**Gaussian FM severely under-covers** the upper tail (PIT 0.23, radii systematically too small): it
generates almost none of the high-radius (flat-background / high-contrast) images. So RAFM *does* capture
a real, otherwise-missed property — the correct proportion of high-contrast/flat-background images. The
open question (Q5) is whether covering that tail = covering perceptually meaningful images.

## Q5 — Does better radial calibration give better perceptual coverage of real tail examples? — No.

Real tail = top-10 % radius images (n=1340). For each method, DINOv2 features of its 3000 generations;
coverage / recall / nearest-generated-neighbour distance of the real tail in DINO space
(`q5_coverage.json`):

| method | tail NN dist (mean) ↓ | tail coverage ↑ | tail recall ↑ |
|---|---|---|---|
| gaussian_euclidean | 55.46 | 0.0433 | 0.063 |
| matched_euclidean | 54.66 | 0.0425 | 0.063 |
| fixed_spherical | 54.54 | 0.048 | 0.152 |
| **rafm** | 55.32 | 0.0425 | 0.075 |

→ **RAFM covers the real perceptual tail no better than Gaussian** (coverage 0.0425 vs 0.0433; NN distance
55.3 vs 55.5 — tied). Despite RAFM's ~7× better radial-tail *calibration* (Q4), its *perceptual* coverage
of real high-radius images is indistinguishable from Gaussian's; if anything fixed_spherical has the best
tail recall. Covering these images perceptually depends on the **direction/content** (modelled similarly
by all methods via the shared SiT), not on getting the **radius** right (which the decoder barely renders).

---

## Final verdict — which conclusions are supported

| candidate conclusion | verdict | evidence |
|---|---|---|
| RAFM better covers meaningful **rare/extreme** events | **❌ refuted** | Q5 tail coverage 0.0425 vs Gaussian 0.0433 (tied); Q2 radius⊥rarity |
| RAFM **calibrates radial tails, but they are not perceptually rare** | **✅ supported** | Q4 (rafm cov 0.045≈target vs gaussian 0.006) + Q2 (radius⊥rarity) + Q5 |
| radius controls a **visible but secondary** image property | **✅ supported** | Q1 (flat-bg/contrast corr 0.70/0.44) + Q3 (contrast +29 %, no semantics) |
| the DC-AE decoder is **largely insensitive to global radius** | **✅ supported** | Q3 (radius sweep = subtle contrast, zero semantic change) |
| **global radius is too coarse**; local geometry matters more | **✅ supported** | Q6 (radius = 1.4 % of variance; class-from-norm 0.17≈chance vs direction 0.62) |

**Synthesis.** The DC-AE latent radius is a **low-level, perceptually secondary scalar** — it tracks
background flatness / contrast ("studio-vs-natural"), is **uncorrelated with perceptual rarity**, changes
**only contrast/saturation** when intervened on, and is **1.4 % of the latent variance** while the
direction/local structure carries the content. **RAFM genuinely and uniquely calibrates the distribution
of this scalar** (the radial tail Gaussian FM systematically misses) — that is the real thing RAFM
captures. But because the scalar is perceptually minor, decoder-near-invisible, and not a rarity signal,
**this calibration does not improve decoded-image FID nor perceptual coverage of real tail examples.**
This fully explains the robust negative result. **Corollary (see follow-up):** RAFM should help precisely
in domains where the latent magnitude *is* perceptually/semantically important and the decoder is
radius-sensitive (e.g. medical intensity, audio energy, physical fields) — a controlled synthetic PoC
would prove the mechanism before choosing such a dataset.

## Figures (uncurated, reproducible)
- `figures/radial_quantile_grids/*` — real images by radial quantile (Q1)
- `figures/q2_radius_vs_rarity.png` — radius vs perceptual rarity (Q2)
- `figures/radius_intervention/*` — fixed-direction radius interventions (Q3)
- `figures/q4q5_tail_vs_perceptual.png` — radial-tail calibration vs perceptual coverage (Q4 vs Q5)

## Reproduce
Scripts in `dit/deferred/`: `deferred_radial_grids.py`, `q1_lowlevel.py`, `deferred_dino_rarity.py`,
`q2_rarity_detail.py`, `deferred_radial_intervention.py`, `q3_measure.py`,
`deferred_tail_calibration.py` (×4 methods), `q5_coverage.py`, `q6_local_structure.py`. Inputs:
`radial_semantics/*.npz`, cached `dino_feats.npy`, decoded `eval_std/*/gen_40000/`.
