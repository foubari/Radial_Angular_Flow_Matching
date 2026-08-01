# ICLR_AUDIT.md — repository audit for the image-latent (DINO/RAE) program

Date: 2026-07-30.

## Git state
- Canonical repo (this checkout): `https://github.com/foubari/radial_angular_FM` (remote `origin`). NOTE: the DINO prompt names `foubari/rafm_for_speech_embeddings` as canonical; user confirmed to proceed in **this** local repo.
- Base commit of rebuttal work: `2e659c7` (master).
- Rebuttal integrated on branch `rebuttal-experiments`:
  - `6ec10cf` exp(rebuttal): complete NeurIPS rebuttal package (1520 text files, 868 KB; heavy binaries gitignored).
  - `e36b0dd` chore(rebuttal): resumable MSGM adapter + local batch default.
- New work branch: **`iclr-image-experiments`** (created from `rebuttal-experiments`). `master` untouched; nothing pushed.
- Uncommitted pre-existing user files left intact: `appendix.tex` (manuscript — not edited), `IMPLEMENTATION_DETAILS.md`, `RESULTS_SUMMARY.md`, `scripts/figure*.py`.

## What is committed vs ignored
- Committed: all `rebuttal_experiments/` compact artifacts (configs, scripts, metrics.json/csv, tables, figures, reports), `experiments/image_latents/` code + diagnostics, `third_party/README.md` (pins).
- Gitignored (reproduce, don't commit): `*.pt/.zip/.npy`, `outputs/`, tb events, `experiments/image_latents/{.venv_img,data}`, `third_party/*/models`, cloned external repos (`third_party/RAE|RJF|flow_matching`), `*.tgz`.

## Environment
- OS Windows 11; GPU RTX 2000 Ada (single, modest). System env `dgm_tire`: torch 2.6.0+cu124, transformers 4.44 (UNCHANGED — protected).
- Isolated image venv `experiments/image_latents/.venv_img` (`--system-site-packages`, reuses system torch): transformers 4.56.2, timm 0.9.16, omegaconf 2.3.0.
- Pinned external: RAE @ `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`; weights `nyu-visionx/RAE-collections` (DINOv2-B ViTXL decoder + imagenet1k stats only); encoder `facebook/dinov2-with-registers-base`.

## Rebuttal reproducibility (Phase 0)
Aggregate tables regenerate from tracked `raw_results/**/metrics.json` via `rebuttal_experiments/scripts/aggregate*.py` (no retraining). MSGM (3 seeds) complete. Full rebuttal results and negative regimes preserved in `rebuttal_experiments/FINAL_REPORT.md` and `tables/ALL_RESULTS.md`.

## Compute posture
ImageNet-256 / LightningDiT-B / 80-epoch / FID-50k is **infeasible on this GPU** (multi-A100 scale). Per user: **diagnostics-first**. Phase 2 (latent geometry) done; CIFAR/ImageNet gated on the Phase-2 outcome and a compute decision.
