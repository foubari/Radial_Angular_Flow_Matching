# image_latents STATUS

Branch iclr-image-experiments. Env: .venv_img (transformers 4.56.2, torch 2.6). RAE @ a4d18c4.

- [x] Phase 0: rebuttal integrated (branch commits 6ec10cf, e36b0dd). ICLR_AUDIT.md.
- [x] Phase 1: isolated venv; RAE DINOv2-B loads + encode/decode validated (latent [768,16,16], recon 256).
- [x] Phase 2: latent geometry on Imagenette val N=2000 -> global-norm CoV 1.6%, per-token 3.9%, no class-radial structure. GATE = near-fixed-radius (angular-only regime). REPORT.md, diagnostics/latent_geometry.json, tables/latent_geometry.md, figures/phase2_latent_geometry.png.
- [ ] DECISION PENDING (user): (a) stop with fixed-radius control result; or (b) Phase 5 non-degenerate latent (SD3-VAE/DC-AE) where radial variation is real; or (c) heavy CIFAR/ImageNet four-way (needs cloud/A100, expected tie on this latent).

Downloads done: RAE code (pinned), DINOv2-B decoder+stats (1.6GB, gitignored), Imagenette-320 val (gitignored).
System dgm_tire env untouched (transformers still 4.44).
