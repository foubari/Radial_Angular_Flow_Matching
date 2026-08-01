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

## Image FID — COMPLETE (post-reboot, controlled one-at-a-time protocol)
After a VM reboot (clean: 0 stray python, pagefile ~2 MB used, 14.5 GB GPU free), each method was decoded
**one at a time, foreground, no background/respawn**, small batch (16), with a **single cached shared InceptionV3
real reference** (Imagenette val, N=2000, seed 1) and the same evaluator. Files: `dit/eval/<method>/fid_pilot.json`,
grids `figures/dit_samples/<method>_20000.png` (images valid, range [0,1]). Limited-sample FID (n=2000, torchvision
InceptionV3 features — relative-only, NOT FID-50k). Per-method runtime ~180–230 s; 0 procs after each; no failures.

| method | source | path | **latent** radial_w1 | latent ks | **image FID (limited)** |
|---|---|---|---|---|---|
| gaussian_euclidean | N(0,I) | Euclid | 7.17 | 0.287 | 176.84 |
| matched_euclidean | eCDF radial | Euclid | 5.36 | 0.197 | 172.61 |
| **fixed_spherical** | fixed R0 | spherical | 9.10 | 0.535 | **158.76 (best)** |
| **rafm** | eCDF radial | spherical | **0.479** | **0.026** | 167.19 |

### Does the latent radial advantage transfer to image quality? — **No (negative result).**
- **RAFM has by far the best latent radial (0.479 vs fixed_spherical 9.10, ~19×) and best KS — but NOT the best image FID.**
  **fixed_spherical, which fixes the radius and has the *worst* latent radial, achieves the *best* image FID (158.8 vs
  RAFM 167.2).**
- The **spherical path transfers to images** (both spherical methods 159/167 beat both Euclidean 173/177), but the
  **matched-radial source — RAFM's specific contribution — does not**: at this DiT scale it gives no image-FID gain over a
  fixed radius, and is marginally worse. The DC-AE decoder is fairly radius-tolerant (Phase-2b: ~29–36 dB at ±10%), which
  plausibly explains why nailing the radial law does not visibly improve decoded images.
- **All four DiT models are real generators** (FID ~160–177, far below the MLP-proxy ~300; well above the decoder floor 21).
- **Honest statement:** *RAFM substantially improves latent radial fidelity, but under the tested generator (DiT-S, 20k
  steps), dataset (Imagenette-10) and evaluator (limited-sample FID), this did not translate into better decoded-image FID;
  fixed-radius spherical flow was marginally best.* Differences (159–177) are modest and within limited-sample-FID noise;
  do not overclaim either direction.

<details><summary>(superseded) earlier FID block — machine resource failure</summary>

## Image FID — was BLOCKED before reboot (documented, not fabricated)
The decode→FID step (venv/DC-AE + InceptionV3) did **not complete**: repeated process pile-ups exhausted the Windows
paging file (WinError 1455), leaving decode processes wedged in kernel I/O (un-killable until OS timeout). **No image-FID
number is reported here** — reporting the earlier MLP-proxy FID≈300 as image quality would be wrong (kept as a documented
failed low-capacity experiment in REPORT.md). The generated latents for all 4 methods × {4k,12k,20k} are saved
(`dit/eval/<method>/gen_<step>.pt` + `mu.pt`); the FID can be computed cleanly once the machine is healthy:

```bash
# after a clean machine (no stray python), one method at a time:
experiments/image_latents/.venv_img/Scripts/python.exe experiments/image_latents/dit/dit_decode_fid.py --method rafm --n 2000
```
</details>

## Honest interpretation (limited)
- **Demonstrated:** RAFM's latent radial advantage **persists with an expressive DiT generator** (not just the toy MLP):
  radial 0.479 vs fixed_spherical 9.10, KS 0.026 vs 0.535, on Imagenette-10 DC-AE latents.
- **Answered (negative):** the latent-radial advantage **did NOT transfer to decoded-image FID** at the tested scale —
  fixed_spherical (worst latent radial) had the best image FID (158.8 vs RAFM 167.2). The spherical *path* transferred;
  the matched-radial *source* did not.
- **Scope caveats:** Imagenette-10 (10 classes, small), pilot budget 20k steps, single seed, **limited-sample FID
  (n=2000, torchvision Inception — NOT FID-50k)**, no ImageNet-1K on this machine. Differences are modest and within
  limited-sample-FID noise — this is a pilot-level negative, not a definitive equivalence proof.

## Status vs prompt stages
Stage 1 (pipeline validation) ✓ · Stage 2 (short 4-way pilot: latent metrics + trajectory + **image FID all 4 methods**) ✓
· Stage 3 (long run) NOT started — see recommendation below.

## Is Stage 3 (long run) scientifically justified on this Shadow machine? — Recommendation: NOT as a full long run.
The pilot already answers the core question at pilot scale: **RAFM's latent-radial advantage did not transfer to image FID;
fixed-radius spherical was marginally best.** Reasons a full multi-day Stage-3 is *not* well justified here:
1. **Data ceiling:** only Imagenette-10 is available on this machine — no ImageNet-1K → Stage 3 cannot produce a real
   ImageNet-256 / FID-50k result; it would only sharpen a small-data, limited-sample-FID comparison.
2. **Mechanism:** the DC-AE decoder is radius-tolerant (Phase-2b: ~29–36 dB at ±10% radius), a plausible reason matching
   the radial law does not visibly help decoded images — a longer run is unlikely to overturn this.
3. **Current differences (159–177) are within limited-sample-FID noise**, so the ranking is not yet robust.

**Better bounded next steps on this machine (cheaper than a full Stage-3, more informative):**
- Tighten the FID: n=5000–10000 generated + real (reduces FID variance) to confirm the ranking is real, not noise.
- Robustness: repeat the 4-way at 2–3 seeds and/or 2× steps (40k) as a sensitivity check on the non-transfer.
- If the non-transfer holds under tighter FID + more steps → report the **negative result** confidently and stop.
A full ImageNet-256 LightningDiT Stage-3 is only worth it on real ImageNet + A100-class compute, which this machine lacks.
