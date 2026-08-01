# DiT image-generation experiment — results (DC-AE latents, Imagenette-10)

**Scientific question:** does RAFM's improved *latent radial fidelity* translate into better decoded image
generation when the latent generator is expressive (a DiT rather than the toy MLP)?

**Setup (this Shadow machine, RTX 2000 Ada 16 GB):** frozen DC-AE encoder/decoder; a self-contained
class-conditional **DiT (adaLN-zero, 32.5M params)** trained with flow matching in the DC-AE latent space
([32,8,8]=2048-d), bf16, ~70 ms/it. Dataset = **Imagenette-10** (13 394 images, 10 classes; **no ImageNet-1K
on this machine**). Four methods, identical DiT/optimizer/schedule/EMA/budget — only source + path differ.
Pilot budget: **20 000 steps/method**, EMA, checkpoints every 2 000. Sampling RK4 (NFE 200), tangent projection
for spherical methods. Latent metrics vs held-out centered test; images = decode of generated latents.

## Latent metrics — four-way (DiT, EMA @ 20k steps) — COMPLETE

| method | source | path | radial_w1 | sliced_w1 | dir_sw1 | cr_sw1 | ks |
|---|---|---|---|---|---|---|---|
| gaussian_euclidean | N(0,I) | Euclid | 7.17 | 0.150 | 0.0009 | 0.097 | 0.287 |
| matched_euclidean | eCDF radial | Euclid | 5.36 | 0.130 | 0.0009 | 0.103 | 0.197 |
| fixed_spherical | fixed R0 | spherical | 9.10 | 0.098 | 0.0008 | 0.093 | 0.535 |
| **rafm** | eCDF radial | spherical | **0.479** | 0.105 | 0.0009 | 0.104 | **0.026** |

**Radial trajectory (4k→12k→20k):** gaussian 9.9→7.8→7.2; matched 6.1→4.9→5.4; fixed_spherical 9.10 (constant, fixed radius);
rafm 0.479 (constant — norm fixed by the source under tangent projection). Quality improves with training for the
Euclidean methods; radial is source-determined for the projected spherical methods.

**Findings (latent space, DiT — confirms the MLP proxy with a much better generator):**
- The DiT is a **far better generator than the MLP proxy** (gaussian radial 7.2 vs the MLP's 24; sliced 0.15 vs 0.46).
- **RAFM again has the best radial by a large margin: 0.479 vs fixed_spherical 9.10 (~19×)** and best KS (0.026 vs 0.535).
  The win over fixed_spherical (same spherical path) is **radial** (matched-radial source); directional metrics
  (dir_sw1, cr_sw1) are essentially tied, and fixed_spherical's global sliced_w1 (0.098) is marginally best — i.e. the
  **shared spherical path governs directions/global, the matched-radial source governs the radius.** RAFM approaches the
  train/test radial floor (0.512).

## Image FID — BLOCKED on this machine (documented, not fabricated)
The decode→FID step (venv/DC-AE + InceptionV3) did **not complete**: repeated process pile-ups exhausted the Windows
paging file (WinError 1455), leaving decode processes wedged in kernel I/O (un-killable until OS timeout). **No image-FID
number is reported here** — reporting the earlier MLP-proxy FID≈300 as image quality would be wrong (kept as a documented
failed low-capacity experiment in REPORT.md). The generated latents for all 4 methods × {4k,12k,20k} are saved
(`dit/eval/<method>/gen_<step>.pt` + `mu.pt`); the FID can be computed cleanly once the machine is healthy:

```bash
# after a clean machine (no stray python), one method at a time:
experiments/image_latents/.venv_img/Scripts/python.exe experiments/image_latents/dit/dit_decode_fid.py --method rafm --n 2000
```

## Honest interpretation (limited)
- **Demonstrated:** RAFM's latent radial advantage **persists with an expressive DiT generator** (not just the toy MLP):
  radial 0.479 vs fixed_spherical 9.10, KS 0.026 vs 0.535, on Imagenette-10 DC-AE latents.
- **Not yet demonstrated:** whether this latent-radial advantage **transfers to decoded-image FID** — the image-FID
  comparison is incomplete (machine resource block). A valid outcome remains either transfer or a documented
  non-transfer; we do not claim transfer without the FID.
- **Scope caveats:** Imagenette-10 (10 classes, small), pilot budget 20k steps, single seed, limited-sample FID
  intended (not FID-50k), no ImageNet-1K available on this machine.

## Status vs prompt stages
Stage 1 (pipeline validation) ✓ · Stage 2 (short 4-way pilot, latent metrics + trajectory) ✓ · image FID ✗ (machine
resource block) · Stage 3 (long run) NOT started — gated on a healthy machine + the pilot FID confirming the DiT is not
the bottleneck (latent metrics already show the DiT is far better than the MLP).
