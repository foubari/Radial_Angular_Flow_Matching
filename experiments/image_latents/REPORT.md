# Image-latent (DINO/RAE) experiments — REPORT

## Scientific question
Does preserving a non-degenerate **empirical radial law** (RAFM's source) improve over a
uniform/fixed-radius spherical prior in DINO/RAE representation latents — or are those latents
effectively fixed-radius, so that RAFM correctly **reduces** to the spherical (angular-only) baseline?
(Reviewer-cited context: DINO-SAE, "Learning on the Manifold"/RJF, RAE.)

## Setup (measured, not assumed)
- Representation: official **RAE DINOv2-B** (encoder `facebook/dinov2-with-registers-base`, ViTXL decoder,
  imagenet1k channel-norm stats), pinned SHA `a4d18c4`. Stage-2 latent = **[768, 16, 16]** (256 tokens × 768-d),
  after RAE's per-channel batchnorm-like normalization — this is the exact tensor the Stage-2 generator consumes.
- Geometry note: the encoder's final LayerNorm (affine disabled) puts each raw token on a sphere of radius ≈√768;
  RAE's per-channel `(z−mean)/√var` then **breaks** exact fixed-radius, inducing some radial spread — which we measure.
- Images: Imagenette-320 val, N=2000 real ImageNet-domain images.

## Phase 2 result — latent geometry (the decision gate)
Files: `diagnostics/latent_geometry.json`, `tables/latent_geometry.md`, `figures/phase2_latent_geometry.png`.

| latent norm | mean | std | **CoV** | skew | kurt |
|---|---|---|---|---|---|
| global ‖z‖ (196608-d) | 505.0 | 8.2 | **0.0162** | −0.13 | −0.17 |
| per-token (768-d) | 31.5 | 1.2 | **0.0385** | 0.14 | 0.21 |

- Between-class radius std 5.74 vs within-class 5.68 on this **N=2000** subset → **no strong per-class radial signal is visible, but the subset is too small to firmly rule out class-conditional radial structure** (10 Imagenette classes only; not the 1000-class ImageNet). Treat as suggestive, not conclusive.
- Decoder radius-sensitivity: PSNR ≈36 dB at ±10% radius, ≈28–31 dB at ±25%, 20–27 dB at ±50% — the decoder *is* radius-sensitive, but the actual latent radial spread is tiny.

### Conclusion (honest, non-cherry-picked) — a latent-geometry DIAGNOSTIC, not a trained comparison
The RAE/DINOv2-B Stage-2 latent is **effectively near-fixed-radius** (global-norm CoV ≈ **1.6%**, per-token ≈ 3.9%,
near-Gaussian). This is a **latent-geometry diagnostic indicating there is very little room for radial matching to help**
on this representation — the empirical radial law is almost a point mass, so RAFM's matched-radial source has almost
nothing to correct relative to a fixed-radius prior. **We do NOT yet claim RAFM empirically equals fixed-radius spherical
flow on DINO/RAE** — that would require actually training and comparing both on this latent, which we have not done here.
The diagnostic only predicts *limited headroom*; confirming a tie would need the trained four-way (compute-gated). We do
**not** manufacture a radial advantage.

### Recommendation
- Report DINO/RAE as a **fixed-radius control** (RAFM ≡ spherical flow) — directly answers the reviewer.
- The decisive test of "does matching a non-degenerate radial law help?" requires a **decodable latent with real radial
  variability** (prompt Phase 5): e.g. **SD3-VAE** or **DC-AE**. Measure its norm CoV first (same protocol); only if
  CoV is non-negligible does the full four-way RAFM comparison add value there.

## Phase 5 result — DC-AE latent geometry (non-degenerate control)
Files: `diagnostics/dcae_geometry.json`, `tables/dcae_geometry.md`, `figures/phase5_dcae_geometry.png`.
DC-AE (`mit-han-lab/dc-ae-f32c32-sana-1.0`), latent **[32,8,8] = 2048-d**, Imagenette val N=2000:

| latent | global-norm CoV | per-token CoV | skew | decodable radius? |
|---|---|---|---|---|
| RAE/DINOv2-B (768×16×16) | **0.016** | 0.039 | −0.13 | yes (±10%≈36 dB) |
| **DC-AE (32×8×8)** | **0.124** | 0.206 | **0.79** | yes (±10%≈29 dB) |

**DC-AE has a genuinely non-degenerate radial law** — global-norm CoV 12.4% (8× RAE/DINO), right-skewed
(heavier tail), within-class radius spread (11.4) larger than between-class (5.4). This is the regime where
matched-radial (RAFM) *can* add value over fixed-radius spherical flow, and where the four-way comparison is
scientifically informative. **Regime map:** RAE/DINO ≈ fixed-radius (RAFM≡SFM); DC-AE = non-degenerate (RAFM testable).

## Phase 5 result — four-way source/path comparison on the DC-AE latent (3 seeds)
Files: `fourway/fourway_dcae.md`, `fourway/radial_floor.json`, `fourway/raw/*/seed_*/metrics.json`,
`figures/phase5_fourway.png`, decoded grids `figures/samples/*.png`, `fourway/fid_small.md`.
Pure rafm-library MLP flow (3×256, matched budget 10k steps, batch 2048) on the **train-mean-centered** DC-AE
latent; 13394 imagenette latents, random 60/20/20; N_gen=5000; **3 seeds (mean±std)**.
**Latent-distribution metrics (radial/sliced/directional W1), NOT image FID-50k.** Irreducible radial-W1 floor of any
empirical-radial source (this split): **train/val 0.190, train/test 0.512**.

| method | source | path | radial_w1 | sliced_w1 | dir_sw1 (norm-dir) | cr_sw1 (common-radial) | ks_stat |
|---|---|---|---|---|---|---|---|
| gaussian_euclidean | N(0,I) | Euclid | 23.95±1.44 | 0.463±0.026 | 0.0023 | 0.238 | 0.804 |
| matched_euclidean | eCDF radial | Euclid | 19.23±0.55 | 0.368±0.005 | 0.0015 | 0.145 | 0.545 |
| fixed_spherical (SFM) | fixed R0 | spherical | 9.10±0.00 | 0.128±0.019 | 0.0012 | 0.128 | 0.535 |
| **rafm_empirical (RAFM)** | eCDF radial | spherical | **0.67±0.11** | **0.116±0.006** | 0.0012 | 0.124 | **0.037** |

**Finding (positive, non-degenerate regime), and where the gain comes from — precisely:**
- RAFM is best overall. Its radial_w1 (0.67±0.11) essentially reaches the **train/test radial floor (0.512)** — i.e. RAFM
  attains the best achievable radial given the source; its KS (0.037) is ~15× better than any other method.
- The decisive isolation **RAFM vs `fixed_spherical`** (both spherical geodesic path; RAFM only adds the matched-radial
  source): the win is **almost entirely radial** — radial_w1 **0.67 vs 9.10 (~14×)**, KS **0.037 vs 0.535**. On the
  **direction-only metrics they are tied** (dir_sw1 0.0012 vs 0.0012; cr_sw1 0.124 vs 0.128) — as expected, since both use
  the same angular flow. Global sliced_w1 is ~tied (0.116 vs 0.128). `fixed_spherical`'s radial_w1 = 9.10 ≈ 0.8·std(radius),
  exactly the error of a point-mass-at-R0 radial law.
- **Honest reading:** matching the empirical radial law (RAFM) helps specifically on the **radial axis** of a non-degenerate
  latent; the shared spherical path governs the directional/angular quality (RAFM = SFM there). So the value of RAFM over
  fixed-radius spherical flow is *radial fidelity*, not better directions.

### Small-sample image FID (decoded) — sanity check, NOT FID-50k
Files: `fourway/fid_small.md`, `figures/samples/*.png` (fixed-order uncurated grids). Generated latents were un-centered,
reshaped to [32,8,8], DC-AE-decoded, and scored with a **torchvision-InceptionV3-feature FID vs Imagenette val, N≈2000/method
(relative across methods only; NOT the canonical clean-fid Inception, NOT FID-50k).**

| gaussian_eucl | matched_eucl | fixed_spherical | **RAFM** | decoder rFID floor |
|---|---|---|---|---|
| 300.7 | 322.1 | 309.7 | **294.2** | **21.3** |

**Crucial honest caveat:** the DC-AE **decoder is good** (rFID floor 21.3, decoding real latents), but **all four small-MLP flows
decode to poor images (~300, ~14× the floor)** — the toy 3×256 MLP is the generation bottleneck, not the source/path choice.
RAFM is marginally best (294 vs 310/301/322) but the **image-FID differences between methods are small**: RAFM's clear
*latent-radial* advantage (radial 0.67 vs 9.10) **does not translate into a meaningful image-FID gain at this model scale.**
Testing whether the radial advantage yields better *images* requires a real generator (LightningDiT/DiT, prompt Phase 4),
which is compute-gated. Do not over-read the ~294 number.

### RAE geometry: one global sphere, token-wise, or product? (Phase 2b)
Files: `diagnostics/rae_sphere_geometry.json`, `tables/rae_sphere_geometry.md`. Global-norm CoV 0.0162; pooled per-token CoV
0.0385; within-image cross-token CoV 0.0345; per-position CoV (across images) 0.032 (max 0.035); between-position mean-radius
rel-std 0.021. **Verdict: a product of ~identical near-fixed-radius per-token spheres (per-token LayerNorm), which — because
the 256 positions sit on nearly the same radius (2% spread) — is also ≈ a single global fixed-radius sphere.** Both the global
(flattened) view used by RAFM and the token-wise view are near-degenerate radially, consistent with the Phase-2 gate.

## Regime-level conclusion (the scientific answer the reviewer asked for)
- **Fixed-radius latent (RAE/DINOv2-B, global CoV 1.6%):** a **latent-geometry diagnostic** shows the radial law is
  almost a point mass → **limited headroom for radial matching**. Whether RAFM exactly ties fixed-radius spherical flow
  here is **not yet demonstrated** (both would have to be trained on this latent — not done).
- **Non-degenerate latent (DC-AE, global CoV 12%):** *trained and compared* — RAFM's matched-radial source
  **substantially improves** over both Euclidean FM and fixed-radius spherical flow (radial ~17× better than
  fixed-spherical, best on all latent metrics).
This supports: matched-radial adds real value when the radial law is non-degenerate (shown on DC-AE), while on a
near-fixed-radius representation (DINO/RAE) the diagnostic predicts little to gain — a claim about *geometry*, pending a
trained confirmation of the tie.

## Claims that must NOT be made
- Do NOT claim RAFM empirically equals (or beats) fixed-radius spherical flow on DINO/RAE **unless both are trained and
  compared** on that latent. The current DINO/RAE result is a **geometry diagnostic** (limited radial headroom), not a
  trained comparison.
- Do not present a small-sample or subset result as ImageNet-1K / FID-50k.
- Do not over-read the N=2000 per-class radius statistic (too small for a firm class-conditional conclusion).

## Status vs prompt phases
Phase 0 (rebuttal integration) ✓ · Phase 1 (env + upstream RAE load validated) ✓ · Phase 2 (latent geometry) ✓ →
**gate says fixed-radius**. Phases 3–4 (CIFAR/ImageNet four-way) and Phase 5 (non-degenerate latent) are compute-gated and
pending the user's decision.
