# image_latents STATUS

Branch iclr-image-experiments. Env: .venv_img (transformers 4.56.2, torch 2.6). RAE @ a4d18c4.

- [x] Phase 0: rebuttal integrated (branch commits 6ec10cf, e36b0dd). ICLR_AUDIT.md.
- [x] Phase 1: isolated venv; RAE DINOv2-B loads + encode/decode validated (latent [768,16,16], recon 256).
- [x] Phase 2: latent geometry on Imagenette val N=2000 -> global-norm CoV 1.6%, per-token 3.9%, no class-radial structure. GATE = near-fixed-radius (angular-only regime). REPORT.md, diagnostics/latent_geometry.json, tables/latent_geometry.md, figures/phase2_latent_geometry.png.
- [x] Phase 5 four-way on DC-AE (non-degenerate) DONE: RAFM best (radial 0.53 vs fixed_spherical 9.10, 17x; sliced 0.108 vs 0.154). Matched-radial helps when radial law non-degenerate. fourway/fourway_dcae.md.
- [ ] Optional: 3 seeds on four-way; decode->small-N FID smoke; CIFAR/ImageNet (compute-gated).
- [x] DECISION was (user): (a) stop with fixed-radius control result; or (b) Phase 5 non-degenerate latent (SD3-VAE/DC-AE) where radial variation is real; or (c) heavy CIFAR/ImageNet four-way (needs cloud/A100, expected tie on this latent).

Downloads done: RAE code (pinned), DINOv2-B decoder+stats (1.6GB, gitignored), Imagenette-320 val (gitignored).
System dgm_tire env untouched (transformers still 4.44).

## 3-seed four-way + extras (done)
- 3-seed DC-AE four-way: RAFM radial 0.67+-0.11 (~floor 0.512), fixed_spherical 9.10; win is radial (KS 0.037 vs 0.535), directions tied. fourway/fourway_dcae.md + radial_floor.json.
- RAE geometry (Phase 2b): product of ~identical near-fixed-radius token spheres ~= single global sphere. tables/rae_sphere_geometry.md.
- Small-sample FID (labeled, torchvision-Inception, N~2000): RAFM 294 (best) but all ~300 vs decoder floor 21 -> toy MLP is bottleneck; radial advantage doesn't reach image FID at this scale. fourway/fid_small.md, figures/samples/*.png.
- Report reframed: DINO/RAE = geometry diagnostic (not trained tie); class claim weakened.
