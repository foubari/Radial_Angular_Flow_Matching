# Screening A — DC-AE local-radius screening + blockwise-RAFM go/no-go

Question: is the global DC-AE radius too coarse — do **local/token-wise radii** carry meaningful image
information that a **blockwise / product-of-spheres RAFM** should transport? Reuses existing real latents,
trained EMAs, cached DINO features, decoded generations; only the missing **generated latents** were
regenerated (re-sampled from each EMA, seed 8925, 40k, RK4 nfe25, n=3000, no decode). **No retraining.**

Latent `z ∈ R^{32×8×8}`. Local-radius families: global(1), channel(32), block2×2(4 of 4×4),
block4×4(16 of 2×2), token(64). `global² = Σ token_r²`.

## Results

### 1. Distributions — non-degenerate; channel radii heavy-tailed (`A2_local_radius.json`)

| family | CoV | skew | excess kurt | q99/q50 | KS(train,test) |
|---|---|---|---|---|---|
| global | 0.118 | 0.87 | 1.52 | 1.38 | 0.038 |
| channel | **0.347** | **5.07** | **46.8** | 2.55 | 0.021 |
| block2×2 | 0.135 | 0.75 | 1.34 | 1.42 | 0.018 |
| block4×4 | 0.165 | 0.71 | 1.27 | 1.53 | 0.017 |
| token | 0.201 | 0.67 | 0.88 | 1.61 | 0.015 |

→ Local radii are **non-degenerate** (token CoV 0.20 > global 0.12) and **channel radii are strongly
heavy-tailed** (skew 5.1, exkurt 47). All families are **very stable** train↔test (KS < 0.04). ✅ criteria
"non-degenerate" and "stable" met.

### 2. Semantic content beyond global (class accuracy, chance 0.10)

| feature | global(1) | block2×2(4) | token(64) | block4×4(16) | **channel(32)** | direction(2048) | full |
|---|---|---|---|---|---|---|---|
| class acc | 0.165 | 0.192 | 0.235 | 0.244 | **0.462** | **0.616** | 0.611 |

→ Local radii carry **more class info than the global radius** (channel 0.46, token 0.24 vs global 0.17),
so the global norm *is* too coarse. **But the direction still dominates** (0.62 = full-latent), i.e. most
content is directional, not in any radius. ⚠️ "informative" partly met, secondary to direction.

### 3. Perceptual activity — weak (`A3_perceptual.json`, 400 imgs, 25 600 tokens)

corr(token-radius, ·): edge **−0.18**, contrast **+0.21**, brightness +0.13, **DC-AE recon-error −0.25**.
All |r| ≤ 0.25. Token-radius heatmaps (`figures/A_token_radius_heatmaps.png`) are noisy and **not
object-aligned**. **Single-token magnitude intervention** (`figures/A_token_intervention.png`, direction
fixed): ×0.25→×2 produces **no visible change**; only ×4 (far out-of-distribution) yields a localized
*artifact*. → local radius is **only weakly perceptually active**; the decoder is radius-tolerant locally
too. ❌ "perceptually active" essentially not met within the data distribution.

### 4. Reproduced or missed by the existing models (real vs generated, `A2`)

Local-radius marginal KS (real test vs generated) and token-radius joint dependence (Frobenius Δ of the
64×64 token-radius correlation matrix):

| method | token KS | channel KS | block2×2 KS | token-corr Δ (joint) |
|---|---|---|---|---|
| gaussian_euclidean | 0.238 | 0.245 | 0.365 | 2.41 |
| matched_euclidean | 0.123 | 0.126 | 0.200 | **1.96** |
| fixed_spherical | 0.059 | 0.062 | 0.175 | **23.2** (destroys it) |
| **rafm** | **0.023** | **0.034** | **0.029** | 2.43 |

→ **Global-RAFM already reproduces the local-radius *marginals* best** (token KS 0.023 — real and RAFM
token-radius histograms overlap almost perfectly, `figures/A_token_radius_real_vs_gen.png`), even though
it only constrains the *global* norm at sampling. Gaussian systematically **under-shoots** local radii
(token KS 0.238). On the **joint** token-radius dependence, RAFM (2.43) ≈ Gaussian (2.41); matched is best
(1.96); fixed_spherical destroys it (23.2) — **no method, RAFM included, specially preserves the joint
local-radius structure.**

## Decision — blockwise / product-of-spheres RAFM on DC-AE images: **NO-GO**

Applying the decision rule (*informative + perceptually active + stable → PoC A; weak or unstable →
document, don't force*):

- ✅ non-degenerate, ✅ stable, ✅ carry *some* extra class info (channel-norms).
- ❌ **perceptually weak** (|corr| ≤ 0.25, not object-aligned, in-distribution interventions invisible —
  decoder radius-tolerant locally).
- ❌ **global-RAFM already matches the local-radius marginals** (token KS 0.023), so a blockwise method
  would add ~nothing on the marginals for *this* decoder.
- The one genuine gap — **joint** local-radius dependence — is **not** uniquely addressed by RAFM (2.43 ≈
  Gaussian), so blockwise RAFM is not the indicated fix even there.

**Therefore: do NOT build a blockwise RAFM to improve DC-AE image generation.** It is not justified — it
would tighten a perceptually secondary, decoder-tolerant, already-well-matched quantity. Forcing it would
repeat the global-RAFM negative result at finer granularity.

### Consequence for the PoCs
- **PoC A (synthetic product-of-spheres)** is *not* justified as a DC-AE remedy, but remains valuable as a
  **controlled proof-of-mechanism** — it constructs a task where local radii *are* perceptually important
  and dependent, i.e. exactly the regime DC-AE lacks. Recommend running it **only** framed that way (show
  *where* blockwise is necessary), with the joint block-radius dependence preserved (product-of-spheres
  with copula, not independent blocks) — not as an image-quality fix.
- **PoC B (audio energy)** is unaffected and is the stronger positive test: there the global magnitude is
  the signal by construction, so **global RAFM** (not blockwise) is the relevant method.

## Artifacts / reproduce
`radial_semantics/A2_local_radius.json`, `A3_perceptual.json`, `genlat_*.pt`, `real_centered.pt`;
figures `A_token_radius_heatmaps.png`, `A_token_intervention.png`, `A_token_radius_real_vs_gen.png`.
Scripts `dit/deferred/A1_gen_latents.py`, `A2_local_radius_stats.py`, `A3_perceptual.py`,
`A4_interventions_figures.py`. Seed 8925, EMA@40k, nfe25, n=3000.
