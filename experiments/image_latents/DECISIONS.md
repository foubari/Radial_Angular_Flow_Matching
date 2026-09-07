# DECISIONS.md — image-latent DiT experiment (choices + justifications)

Branch `iclr-image-experiments`. This log records *why* each choice was made (per reviewer request).
Supersedes the raw-latent / custom-DiT pilot where noted.

## D1 — DC-AE model & latent scaling  (AUDIT FIX, changes results)
- Model: `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers` (diffusers `AutoencoderDC`), frozen. latent [32,8,8]=2048-d.
- **Official scaling_factor = 0.41407** (verified from `ae.config`). NOTE: the reviewer's `0.3189` is the `-in-1.0`
  variant, NOT the sana model we use.
- `encode(x).latent` is the **raw** latent (std ≈ 2.0). The prior pilot trained the flow in this raw space.
- **Decision: train/evaluate the flow in the SCALED latent space** `z = encode(x).latent * 0.41407` (≈ unit variance),
  then **train-only** centering; decode by inverting: `raw = (z_model / 0.41407)` (undo centering first) → `decode(raw)`.
  - **Why:** (a) official convention (the scaling_factor exists to give the generator ~unit-variance inputs);
    (b) in raw space (std 2) the Gaussian source N(0,I) (std 1) suffers a *trivial scale mismatch*, unfairly penalising
    `gaussian_euclidean` and inflating RAFM's apparent radial advantage. Scaled space makes gaussian a fair baseline and
    isolates the *radial-shape* question (further separated by the moment-matched-Gaussian diagnostic).
  - **Consequence:** the raw-space pilot numbers (latent radial 0.48 vs 9.10; FID 159–177) are **superseded**; latents
    are re-extracted in scaled space and all four methods re-trained. Round-trip test: `decode((raw*sf)/sf)==decode(raw)`
    (global scalar ⇒ exact), plus train-only centering inverted in reverse order.

## D2 — Generator backbone: official SiT (vendored) instead of the custom DiT
- **Decision: use `willisma/SiT` `models.py` (class `SiT`) as the backbone**, vendored under `third_party/SiT` at a
  pinned SHA, configured for DC-AE: `input_size=8, in_channels=32, patch_size=1, num_classes=10, learn_sigma=False`
  (velocity output). Our rafm library supplies the 4 source/path variants (SiT's own transport is Gaussian-linear only).
  - **Why:** the reviewer asked for an official architectural reference; SiT is the closest FM/velocity DiT, is
    self-contained (one file, deps torch+einops), decoupled from its transport/VAE/dataset, and adapts cleanly to the
    DC-AE latent. It is *not* heavy, so per the user's instruction we prefer the official base over from-scratch.
  - The prior custom `dit_model.py` (DiT-style, adaLN-zero, 32.5M) is kept as a documented reference/fallback but is
    superseded. Re-training is required anyway because of D1 (scaling), so switching backbone now is free.
  - Label to use in the report: **"official SiT backbone (vendored, pinned) with custom RAFM transport"** — not
    "LightningDiT" (LightningDiT is VA-VAE-specific and not used).

## D3 — Standardized image evaluator (replaces the custom FID)
- **Decision: one pinned evaluator for all new results** — `torch-fidelity` (FID + KID) and `prdc`/generative-evaluation
  (precision/recall/density/coverage). Report gFID, rFID (decoder floor), KID±CI, precision/recall, density/coverage,
  all with the **same** real reference set, feature extractor, resize, seed, and generated sample count.
  - **Why:** the pilot's custom torchvision-InceptionV3 FID (159–177) is non-standard and not comparable; the reviewer
    requires one reproducible evaluator. Those pilot FID numbers are **superseded**.
- Primary sampling **unguided (CFG=1)**; any CFG study reported separately with identical CFG per method.
- Always labelled **limited-sample FID (n=...)**, never compared to published ImageNet FID-50k.

## D4 — Resumability (USER PRIORITY: VM can stop mid-run)
- **Decision: complete-state checkpoints** — model + EMA + optimizer + step + **RNG state (torch, torch.cuda, numpy,
  python `random`)** + deterministic step-indexed data sampling. A single **re-runnable** foreground command resumes
  exactly from the last checkpoint. **No background respawn daemon** (that caused the earlier RAM/pagefile exhaustion).
  - **Why:** the VM can be interrupted; re-running must continue bit-exactly. One process at a time on Windows.
  - Resume is triggered manually (user re-runs the command, or the agent re-runs next session). A `.bat` is provided
    that is safe to double-click / re-run and continues where it stopped.

## D5 — Scope (this Shadow machine)
- **In scope:** provenance audit, unit tests, scaled-latent re-extraction, SiT backbone, 4-way × 3 seeds to 40k
  (checkpoints 20/30/40k; 80k only if still improving), standardized eval, radius-semantics, radius intervention,
  radial-quantile, radial/direction swaps, moment-matched-Gaussian diagnostic.
- **Excluded here (documented):** **ImageNet-100** (dataset not on the machine; large download + manifest) and
  **pretrained-model transfer** (published DiT/SiT checkpoints use a different VAE/latent, incompatible with DC-AE).
  - **Why:** infeasible or scientifically meaningless on this single-GPU, Imagenette-only machine.

## D6 — Datasets & DINO/RAE status
- Image dataset: **Imagenette-10** (13 394 imgs, 10 classes). Real reference = a fixed val split, class-balanced
  generation. Labelled clearly (not ImageNet).
- DINO/RAE: keep the completed **geometry diagnostic** (near-fixed-radius, global CoV 1.6%); do **not** claim
  RAFM≡fixed-spherical there without training both. No claim of novelty for spherical image-latent transport
  (DINO-SAE, RJF exist).

_Open items surfaced to the user before long runs: (a) reduced scope A+B confirmed; (b) SiT-base chosen (this doc)._
