# Phase A — build complete, verified, ready for the 10.7 h Phase B

Everything below is **built and tested on this Shadow VM** (no long runs yet). Phase B (the 12
training runs) is **not launched** — it waits for your go. See [DECISIONS.md](DECISIONS.md) for *why*
each choice was made.

## 1. What was verified (all green)

| check | how | result |
|---|---|---|
| **Resume is bit-exact** (your #1 priority) | real trainer 0→120 vs 0→60 then re-run→120, compare EMA | **max\|dev\| = 0.00e+00**, identical loss — `dit/resume_test.py` |
| Geometry / velocity / sources / preprocessing | 15 unit tests | **ALL PASS** — `dit/dit_tests.py` |
| All 4 methods train at full arch on real scaled latents | 120-step smoke each, one at a time | 32.5M SiT, ~**80 ms/it**, no failures |
| Standardized evaluator runs end-to-end | sample→decode→FID/KID/prdc+latent, JSON out | **works** (torch-fidelity 0.4.0 + prdc) — `dit/dit_eval_sit.py` |

The unit tests cover: slerp norm preservation (max dev 1e-6), endpoints t=0/1, tangent velocity ⊥ state
(cos < 3e-7), analytic vs finite-diff velocity (rel err 3e-5), Euclidean velocity = x1−x0, empirical
radial source matches train radii (KS 0.013), fixed_spherical = point-mass radius, Gaussian unit
variance, and the scaling+centering inverse round-trip (exact, 5e-7).

## 2. Provenance (what is official vs ours)

| component | source | pinned |
|---|---|---|
| Backbone `SiT` | `willisma/SiT` `models.py`, vendored `third_party/SiT` | SHA `cbde832` |
| Autoencoder | `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers` (diffusers `AutoencoderDC`), **frozen** | HF pinned |
| Evaluator | `torch-fidelity` 0.4.0 (FID/KID) + `prdc` (P/R/density/coverage) | InceptionV3-2015 weights |
| Transport (4 variants) | **ours** — `rafm/` (sources, paths, tangent projection) | repo |

**Label for the paper:** *"official SiT backbone (vendored, pinned) with custom RAFM transport."* Not
"LightningDiT".

## 3. Exact architecture & preprocessing

- SiT: `input_size=8, patch_size=1, in_channels=32, hidden=384, depth=12, heads=6, num_classes=10,
  class_dropout_prob=0.1, learn_sigma=False` → **32.5 M params**, velocity output.
- Latent space: raw DC-AE latent **× 0.41407** (official scaling), then **train-only** centering
  (mean over the train split of the 60/20/20 = 8036/2679/2679 split, seed 0). Decode inverts:
  `raw = (z_model + μ)/0.41407` → `AutoencoderDC.decode`.
- Loss: CFM, `‖v_θ(x_t,t,y) − u_t‖²` summed over dim, mean over batch, bf16 autocast, AdamW
  (lr 1e-4, β=(0.9,0.95), wd 0), EMA 0.9999.
- Sampling (eval): RK4, **NFE 50**, **CFG = 1 (unguided)** primary, tangent projection on for the two
  spherical methods, class-balanced.

## 4. Four methods (source × path)

| method | source x0 | path | tangent proj |
|---|---|---|---|
| gaussian_euclidean | N(0,I) | Euclidean | no |
| matched_euclidean | empirical radial (eCDF) · random direction | Euclidean | no |
| fixed_spherical | fixed radius R0 · random direction | spherical geodesic | yes |
| **rafm** | empirical radial (eCDF) · random direction | spherical geodesic | yes |

## 5. Cost estimate (this VM, RTX 2000 Ada)

- **80 ms/it × 40 000 = ~53 min per run.**
- **4 methods × 3 seeds (8925, 1234, 7) = 12 runs ≈ 10.7 h** total training. Fits your "10 h is fine".
- Checkpoints saved at **20k / 30k / 40k** (+ a rolling complete-state `ckpt.pt` every 2000 for resume).
- **Disk:** rolling `ckpt.pt` ≈ 520 MB/run (model+EMA+AdamW+RNG) + three EMA snapshots ≈ 390 MB/run →
  **~11 GB** for all 12 runs. Eval adds a shared 5k-image real reference (~a few hundred MB).
- Evaluation (after training): ~2 min/checkpoint × (12 runs × 3 checkpoints) ≈ **~1–1.5 h**, one at a time.

## 6. How to launch Phase B (resumable, one command)

```bash
experiments/image_latents/dit/launch_phaseB.bat
```

Double-click or run it. It runs the 12 trainings **sequentially, one process at a time, no background
respawn**. If the VM stops, **just run the same file again** — each run resumes bit-exactly from its
last checkpoint, and runs already at 40k exit instantly. To launch a single run manually instead:

```bash
experiments/image_latents/.venv_img/Scripts/python.exe experiments/image_latents/dit/dit_train_sit.py --method rafm --seed 8925 --out experiments/image_latents/dit_sit/seed_8925 --steps 40000
```

## 7. Still to do (Phase B, after your go)

1. Run `launch_phaseB.bat` (~10.7 h, resumable).
2. Evaluate each method × {20k,30k,40k} × 3 seeds with `dit_eval_sit.py` (FID/KID/prdc + latent),
   plus rFID decoder floor; aggregate mean±std over seeds.
3. Radius studies (as required by the reviewer): radius-semantics, radius intervention, radial-quantile,
   radial/direction swaps, and the moment-matched-Gaussian diagnostic.
4. Write results into a refreshed `DIT_IMAGE_RESULTS.md`, preserving the honest negative/partial
   conclusion.
