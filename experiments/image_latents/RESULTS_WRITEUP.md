# Does RAFM's latent radial fidelity translate to better decoded images? — No.

**Authoritative results for the DC-AE image-generation experiment (supersedes the raw-space / custom-FID
pilot).** All numbers are 3 seeds (8925, 1234, 7), mean ± std, standardized evaluator.

## Setup

Frozen DC-AE (`mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`), latent [32,8,8]=2048-d, official
scaling ×0.41407 + train-only centering. Generator: **official SiT backbone** (vendored, pinned
`cbde832`), `input_size=8, patch_size=1, in_channels=32, hidden=384, depth=12, heads=6, num_classes=10,
class_dropout=0.1, learn_sigma=False` → 32.5 M params, velocity output. Training: CFM, AdamW lr 1e-4,
bf16, EMA 0.9999, **40 000 steps**, checkpoints 20/30/40k. Dataset: **Imagenette-10** (13 394 imgs;
60/20/20 split, seed 0). Only source + path differ across the four methods; everything else is identical.

**Four methods** (source × path):

| method | source x0 | path | tangent proj. at sampling |
|---|---|---|---|
| gaussian_euclidean | N(0,I) | Euclidean | no |
| matched_euclidean | empirical radial (eCDF) · random dir | Euclidean | no |
| fixed_spherical | fixed radius R0 · random dir | spherical geodesic | yes |
| **rafm** | empirical radial (eCDF) · random dir | spherical geodesic | yes |

**Evaluation (standardized, identical for all):** class-conditional **unguided (CFG=1)** generation from
noise (not reconstruction), class-balanced; RK4 solver, **25 steps = 100 model evaluations** (instrumented,
`verify_solver.py`); decode with DC-AE; **torch-fidelity** FID/KID + **prdc** precision/recall/density/
coverage; n=3000 generated vs 3925 real Imagenette-val (fixed cached reference). Latent metrics vs
held-out centered test. FID is **limited-sample (n=3000) — relative, not FID-50k.**

## Headline result — checkpoint 40 000 (3 seeds)

| method | radial_w1 ↓ | ks ↓ | sliced_w1 ↓ | **FID ↓** | KID ↓ | precision | recall |
|---|---|---|---|---|---|---|---|
| gaussian_euclidean | 4.68 ± 0.10 | 0.42 ± 0.01 | 0.093 | **161.0 ± 0.6** | **0.104** | 0.135 | 0.693 |
| fixed_spherical | 3.77 ± 0.00 | 0.535 | 0.045 | 166.5 ± 0.4 | 0.118 | 0.137 | 0.671 |
| matched_euclidean | 2.63 ± 0.15 | 0.24 ± 0.02 | 0.063 | 171.7 ± 1.2 | 0.119 | 0.124 | 0.685 |
| **rafm** | **0.27 ± 0.00** | **0.04** | **0.042** | **174.5 ± 1.2** | 0.126 | 0.130 | 0.667 |

**RAFM wins every latent metric and loses every image metric.** Its latent radial-W1 (0.27) is ~17×
better than gaussian's (4.68) and its KS (0.04) ~10× better; it also has the best global sliced-W1
(0.042). Yet it has the **worst** gFID (174.5) and worst KID, while **gaussian_euclidean — the worst
latent radial — has the best gFID (161.0) and KID.** The FID ranking is essentially the reverse of the
radial ranking. Differences are far larger than the seed noise (FID std 0.4–1.2), so the ranking is
**robust**, not a limited-sample artefact.

Decomposition: the **spherical path** improves the global sliced-W1 (rafm/fixed 0.042/0.045 vs Euclidean
0.063/0.093); the **empirical-radial source** improves radial-W1/KS (rafm/matched ≪ gaussian). Both
distributional gains are real — and **neither buys image quality**; RAFM's full combination is the worst
on FID/KID.

## Trajectory — FID (3 seeds) at 20k / 30k / 40k

| method | FID @20k | FID @30k | FID @40k | radial_w1 @40k |
|---|---|---|---|---|
| gaussian_euclidean | **233.4** | **186.1** | **161.0** | 4.68 |
| matched_euclidean | 238.2 | 192.2 | 171.7 | 2.63 |
| fixed_spherical | 264.0 | 202.4 | 166.5 | 3.77 |
| rafm | 251.6 | 203.1 | **174.5** | **0.27** |

**gaussian_euclidean has the best FID at *every* checkpoint; RAFM is worst or near-worst at every
checkpoint** — the negative result holds across the whole training trajectory, not just the endpoint.
All methods improve monotonically with training (no over-fitting); the spherical methods improve fastest
(fixed 264→166, rafm 252→174) yet never overtake gaussian. RAFM's latent radial-W1 is locked at 0.27
throughout (source-determined), while gaussian's improves 15.6→8.2→4.7 with training — and still gaussian
wins FID at all points. The conclusion is not a "not-trained-enough" or single-checkpoint artefact.

## Why (mechanism)

1. **RAFM's radial fidelity is essentially free and does not require the generator to learn it.** The
   sampler uses tangent-projected velocity on the spherical methods; instrumentation
   (`verify_solver.py`, RK4/100 NFE) shows the **state norm drifts < 0.001 %** over sampling with **no
   explicit renormalization** — so the generated radius stays locked to the source radius. RAFM's source
   radii are drawn from the empirical (train) law, so its latent radial-W1 is near-zero **by construction
   of the transport**, independent of the seed (hence the ±0.000). This is why matching the radial law
   does not, on its own, imply better images: it is a property of the source+path, not of image quality.
2. **The DC-AE decoder is radius-tolerant** (Phase-2b: ~29–36 dB PSNR at ±10 % radius). Getting the
   latent radial law exactly right therefore has little leverage on the decoded pixels the FID/Inception
   features actually see.
3. **The latent radius is a distributed, only mildly class-correlated quantity** (radial-semantics:
   CoV 0.117, η²=0.16 class-explained, max |corr(radius, any PC)| = 0.31) — a global magnitude, not the
   dominant axis of image variation. Constraining it does not steer the perceptually important structure,
   and RAFM's extra geometric constraints (matched radius + spherical geodesic + tangent projection)
   appear to mildly *mismatch* what an expressive velocity field would otherwise fit freely, costing FID.

## Honest conclusion

> RAFM substantially and reliably improves **latent radial fidelity** (radial-W1 0.27 vs 4.68, KS 0.04 vs
> 0.42) with an expressive official SiT generator on Imagenette-10 DC-AE latents. **This advantage does
> not transfer to decoded-image quality — it is anti-correlated with it:** under a standardized unguided
> evaluator (torch-fidelity FID/KID + prdc, 3 seeds), RAFM has the **worst** limited-sample gFID/KID and
> gaussian_euclidean, with the worst latent radial, the **best**. The spherical path and matched-radial
> source each improve latent distributional metrics but neither improves images; the full RAFM
> combination degrades FID. The mechanism is consistent across three independent diagnostics: the radial
> fidelity is a sampling-time norm-preservation artefact (not learned image quality), the DC-AE decoder is
> radius-tolerant, and the latent radius is a distributed, weakly-semantic magnitude.

**Scope / caveats:** Imagenette-10 (10 classes, small), limited-sample FID (n=3000, torch-fidelity
Inception — **not** FID-50k), unguided class-conditional sampling, single dataset/decoder. This is a
robust *pilot-scale negative*, not a universal claim about all latent spaces or all decoders.

**Still to run (deferred, GPU):** rFID decoder floor, radius-intervention decode, radial-tail
calibration, DINO rarity — to further probe the mechanism (scripts prepared in `dit/deferred/`).
