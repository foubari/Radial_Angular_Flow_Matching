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

- Between-class radius std 5.74 ≈ within-class 5.68 → **no meaningful class-conditional radial structure**.
- Decoder radius-sensitivity: PSNR ≈36 dB at ±10% radius, ≈28–31 dB at ±25%, 20–27 dB at ±50% — the decoder *is* radius-sensitive, but the actual latent radial spread is tiny.

### Conclusion (honest, non-cherry-picked)
The RAE/DINOv2-B Stage-2 latent is **effectively near-fixed-radius** (global-norm CoV ≈ **1.6%**, per-token ≈ 3.9%,
near-Gaussian, no class-radial structure). By the pre-registered decision rule this is a **fixed-radius / angular-only
regime**: RAFM's empirical radial matching has almost nothing to correct, so RAFM is expected to **reduce to the
fixed-radius spherical baseline** here. This is a scientifically meaningful *neutral* result — it confirms RAFM behaves
correctly (reduces to angular flow) when the representation is already fixed-radius, and it means a heavy CIFAR/ImageNet
RAFM-vs-SFM run on this latent would (expensively) confirm a **tie**, not a RAFM win. We do **not** manufacture a radial
advantage.

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

## Claims that must NOT be made
- Do not claim RAFM beats spherical flow on DINO/RAE latents (the latent is ~fixed-radius; expect a tie).
- Do not present a small-sample or subset result as ImageNet-1K / FID-50k.

## Status vs prompt phases
Phase 0 (rebuttal integration) ✓ · Phase 1 (env + upstream RAE load validated) ✓ · Phase 2 (latent geometry) ✓ →
**gate says fixed-radius**. Phases 3–4 (CIFAR/ImageNet four-way) and Phase 5 (non-degenerate latent) are compute-gated and
pending the user's decision.
