# REPO_AUDIT.md — Radial-Angular Flow Matching (RAFM)

Audit date: 2026-07-27. Auditor: automated rebuttal-prep agent.
Repo: `C:\Users\Shadow\Desktop\Radial_Angular_FM`.

Paper: *Correcting Source Mismatch in Flow Matching with Radial-Angular Transport* (submission title given by user: "Source-Matched Flow Matching via Radial-Angular Transport"). This audit is read-only except for creating the safe branch and the `rebuttal_experiments/` tree.

---

## 0. Git state (safety record)

| Item | Value |
|---|---|
| Original branch | `master` (tracking `origin/master`) |
| Original HEAD commit | `2e659c750149e9eb4c6f36177142b7793188d97f` ("Add method overview figure to README") |
| New working branch | `rebuttal-experiments` (created from master; uncommitted changes carried over intact) |
| Recent history | `2e659c7`, `bd37386` (README figure), `3dbcd3b` (no_projection ablation), `7e2ad3f` (PIV configs, MSGM resume), `8cb013e` (initial commit) |

### Uncommitted user changes — MUST remain untouched (preserved on new branch)

Tracked, modified (kept as-is, NOT reverted):
- `configs/defaults.yaml` — `batch_size: 4096 → 256` (only change).
- `baselines/msgm_adapter.py` — default `n_train_steps 100_000 → 10_000`; added `train_log.csv` logging block.

Untracked user files (kept, not overwritten):
- `IMPLEMENTATION_DETAILS.md`, `RESULTS_SUMMARY.md`, `appendix.tex`
- `scripts/convergence_rafm_msgm.py`, `scripts/figure1_method*.py`, `scripts/figure_*.py` (11 figure scripts)

> **Reproduction caveat (important).** The committed/documented training `batch_size` is **4096** (IMPLEMENTATION_DETAILS §4.1, §11), but the working-tree `configs/defaults.yaml` now says **256**. Existing results in `outputs/` were produced under the documented 4096 setting. All experiment configs (`configs/exp1/*.yaml`) inherit `defaults.yaml`, so a run *today* would silently use 256. To reproduce faithfully, rebuttal configs set `batch_size: 4096` explicitly and never edit the existing configs.

---

## 1. What the method is

Standard Flow Matching (FM) uses a Gaussian source N(0,I). Its radial law is a χ-distribution, which mismatches heavy-tailed / real data. **RAFM** corrects the source and the transport path:

1. **Radial coupling.** For each target `x1`, source is `x0 = R·u0` with `R = ‖x1‖`, `u0 ~ Unif(S^{d-1})`. So `‖x0‖ = ‖x1‖` — source and target share a sphere.
2. **Spherical geodesic (slerp) path.** `x_t = R·slerp(u0,u1,t)`, keeping `‖x_t‖ = R` for all t. Closed-form tangent conditional velocity (Riemannian log).
3. **Tangent projection at sampling.** ODE velocity is projected onto the sphere tangent at each step (`v − (⟨x,v⟩/‖x‖²)x`), so norm never drifts.

**Consequence (central honesty point).** With tangent projection, `‖x_t‖ = ‖x0‖` exactly for the whole trajectory. Therefore the **radial distribution of RAFM samples equals the radial distribution of its source** — it does not depend on the trained network at all. The network only learns *angular* rotation. This is confirmed by the repo's own convergence test (radial_W1 identical at 10k/50k/100k steps) and stated in `RESULTS_SUMMARY.md`. Any radial-metric advantage of RAFM over Gaussian FM is attributable to the **eCDF source**, not to the flow. The path/flow contribution shows up in *angular/global* metrics (sliced-W1, angular-SW, MMD). Ablations must reflect this attribution.

## 2. Method / source / path matrix (as implemented)

| Method | Source | Path | Norm-preserving? |
|---|---|---|---|
| Gaussian FM (baseline) | N(0,I) | Euclidean | no |
| Source-only (oracle) | analytic radial CDF | Euclidean | no (only x0 shares R) |
| Source-only (empirical) | radial eCDF | Euclidean | no |
| RAFM (oracle) | analytic radial CDF | spherical geodesic | yes (tangent proj.) |
| RAFM (empirical) | radial eCDF | spherical geodesic | yes |
| MSGM (baseline) | multiplicative-noise SDE, eCDF prior | Stratonovich SDE | optional norm correction |

Source-only shares RAFM's **coupling** (`x0=R·u0`) but uses a straight Euclidean path → clean isolation of the *path* contribution. Gaussian FM is the uncoupled baseline → isolates the *source* contribution. The ablation ladder Gaussian → source-only → RAFM is already built in and mathematically valid.

## 3. Code map (`rafm/` package, installed editable as `rafm`)

| Area | File | Role |
|---|---|---|
| Model | `models/mlp.py` | Time-concat MLP, Swish, 3 hidden×128. `premodule` (NormalizeLogRadius / Polar) optional, off by default. |
| Sources | `sources/gaussian.py`, `radial_oracle.py`, `radial_empirical.py`, `student_t.py` | Gaussian; oracle analytic radial samplers (student_t/gaussian_aniso/toy); eCDF/KDE empirical (+log-radius). |
| Paths | `paths/euclidean.py`, `paths/spherical_geodesic.py` | Linear path (`u_t=x1−x0`); slerp path (closed-form tangent velocity). |
| Geometry | `utils/sphere.py` | `uniform_on_sphere`, `slerp`, `slerp_velocity`, `riemannian_log`, antipodal/near-zero handling. |
| Flow matching | `flow_matching/loss.py`, `trainer.py`, `sampler.py` | CFM MSE loss w/ coupling; Adam trainer; ODE sampler (euler/heun/rk4) + tangent projection. |
| Metrics | `metrics/radial.py`, `distributional.py`, `angular.py`, `stability.py` | radial_W1/KS/quantile/tail; sliced-W1 + MMD; per-radial-bin angular SW; NaN/exploding rates. |
| Data | `data/{student_t,gaussian_aniso,toy_radial_angular,piv}.py`, `data/base.py`, `data/prepare_piv.py` | synthetic generators + PIV loader/preprocessor. |
| Baselines | `baselines/msgm_adapter.py`, `msgm_runner.py` | wraps `18727_Multiplicative_Diffusion_code/` unchanged. |
| Experiments | `experiments/exp0..exp4` | source diag / main benchmark / sample-eff / runtime / solver. |
| Scripts | `scripts/*` | aggregate_results, ablation_no_projection, convergence_test, figures, tables, inspect_piv. |

## 4. Datasets supported

| Dataset | Dims present | N | Tails | Anisotropy | Oracle source? | Data on disk |
|---|---|---|---|---|---|---|
| Student-t (correlated, df=3) | 16, 32 | 50k | heavy (poly) | yes (A, seed 42) | yes | generated on the fly |
| Gaussian aniso (control) | 16, 32 | 50k | light | yes (same A) | yes | on the fly |
| Toy 2D radial-angular | 2 | 50k | heavy radius, 4 vonMises modes | n/a | yes | on the fly |
| PIV (Re=3900 cylinder vorticity) | 16/32/64/256 | see .pt | heavy (real) | yes | **no** (unknown CDF) | `data/piv/*.pt` present (piv_d16/32/64/256 + trunc variants) |

PIV `.pt` files already exist (no need to re-run the 3.95 GB `dataverse_files.zip`). Split 60/20/20, `split_seed=0`. Synthetic train=30k/val=10k/test=10k.

## 5. Training / sampling / metric settings

- **Optimizer** Adam, lr 1e-3, wd 0, no scheduler/warmup/clip/EMA. **Steps** 10,000. **Batch** documented 4096 (working tree 256 — see caveat). Data preloaded on GPU, `torch.randint` minibatching. `torch.compile` on Linux+CUDA only (off on Windows — this is Windows).
- **Sampling** default RK4, N=128 steps → NFE=512; solvers euler(NFE=N)/heun(2N)/rk4(4N). Tangent projection auto-on for spherical path, off for euclidean. n_gen=10,000.
- **Metrics** radial_W1, KS, quantile err (95/99/99.5), tail exceedance (95/99); sliced-W1 (500 proj); MMD (median-heuristic RBF); angular SW per 4 radial bins (200 proj) + mean; stability nan_rate/exploding_norm_rate/invalid_rate.
- **Seeds** model seeds `{8925, 77395, 65457}` (3 seeds), split_seed 0, matrix_seed 42.
- **NaN handling** slerp clamps (min 1e-12), antipodal perturbation (θ>π−1e-3), near-zero linear fallback (θ<1e-6); tangent projection skipped for ‖x‖<1e-3.

## 6. How the numerical-constraint questions are answered by the code

| Question | Implementation |
|---|---|
| Radius sampling | eCDF: `torch.quantile(r_train, U)` (inverse-CDF, linear interp); oracle: analytic `‖A z‖`; KDE + optional log-radius variants. |
| Angular interpolation | slerp on unit vectors, rescaled by R. |
| Spherical geodesics | `utils/sphere.slerp` with acos-clamped angle; re-normalize then rescale by R. |
| Tangent-space velocity | `slerp_velocity` closed form (= Riemannian log / (1−t)); tangent by construction. |
| Radius preservation | slerp keeps ‖·‖=R analytically; at sampling, `_project_tangent` removes radial velocity each step. |
| Numerical integration | fixed-step Euler / Heun / RK4 (no adaptive solver present). |
| Projection / renorm | tangent projection each ODE step (spherical only); slerp re-normalizes; no explicit sphere re-projection of state (only velocity). MSGM has optional norm correction `x·(‖x0‖/‖x‖)`. |

## 7. Existing results & checkpoints (do not overwrite)

`outputs/exp1_main_benchmark/{student_t_d16_df3.0_cor, student_t_d32_df3.0_cor, gaussian_aniso_d16_cor, toy_radial_angular, piv_d16, piv_d64, piv_d256}/<method>/seed_*/` each with `metrics.json`, `samples.pt`, `checkpoint.pt`, `config.yaml`, `train_log.csv`, `tb/`. Also `outputs/exp0/*`, `outputs/convergence_test/*`, `outputs/convergence_rafm_msgm/`. Figures in `figures/` and `assets/`. These are treated as **read-only ground truth** for reproduction comparison; all new runs go under `rebuttal_experiments/`.

Sample sanity check: `outputs/.../gaussian_aniso_d16_cor/rafm_empirical/seed_8925/metrics.json` → radial_w1 0.058, sliced_w1 0.142, nan_rate 0, train 35 s. Consistent order-of-magnitude with `RESULTS_SUMMARY.md` (per-seed vs 3-seed mean differ).

## 8. Known issues / reviewer-relevant weaknesses (from repo + code reading)

1. **Toy 2D RAFM oracle NaN 18.4%** — near-origin (small R) slerp instability in d=2. Documented limitation; candidate for diagnosis/fix (numerical ablation, not a silent change).
2. **radial_W1 flat across training steps** — by design (tangent projection fixes norm). Must be framed as "radial = source, flow = angular", not as a training win.
3. **Exp 4 solver curve noisy** — single model seed + re-training per point; RESULTS_SUMMARY notes it "would benefit from a pre-trained checkpoint". Fix: evaluate one fixed checkpoint across NFE/solver.
4. **No adaptive/midpoint/RK4-vs-dopri solver** — only euler/heun/rk4. Adding midpoint + adaptive is a safe *additive* numerical ablation.
5. **Batch-size drift 4096→256** in working tree (reproduction hazard, above).
6. **Small MLP only** (33k–66k params). Higher-capacity arch untested — reviewer likely asks.
7. **PIV normalization uses whole-dataset mean** (pre-split) — mild train/test leakage of the centering statistic; inherited from MSGM. Worth a documented robustness check (center on train only).

## 9. Environment

- OS Windows 11; Python 3.10.19; torch 2.6.0+cu124; CUDA available; GPU **NVIDIA RTX 2000 Ada Generation**; numpy 2.2.6, scipy 1.15.3, sklearn 1.7.2. `torch.compile` disabled on Windows.
- Compute reality: MLP runs are tiny — one FM training (10k steps, batch 4096) ≈ 35–110 s on this GPU; sampling 10k @ NFE512 < 1 s. **MSGM ≈ 300 ms/step ⇒ ~50 min / 10k-step run** (the expensive item). Single GPU, so runs are serial.

## 10. Smoke tests performed (this audit)

- `pytest tests/` → **45 passed** in ~6 s. Covers slerp norm preservation, tangent velocity orthogonality, antipodal handling, sources, metrics, data.
- Tiny exp1 (`rebuttal_experiments/configs/smoke_exp1_studentt_d16.yaml`, 200 steps, 4k samples, NFE16, 1 seed, output isolated under `rebuttal_experiments/raw_results/_smoke`) → pipeline OK, ordering already gaussian_fm 8.62 > source_only 3.88 > rafm 0.90 (radial_W1), matching the paper's direction. No existing output touched.

## 11. Reproduction commands (existing experiments)

```bash
# from repo root, conda env with torch+cuda
pytest tests/ -v
python -m experiments.exp0_source_diagnostics
python -m experiments.exp1_main_benchmark --config configs/exp1/studentt_d16.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/studentt_d32.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/gaussian_d16.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/piv_d64.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/piv_d256.yaml
python -m experiments.exp2_sample_efficiency
python -m experiments.exp3_runtime
python -m experiments.exp4_solver_sensitivity
python scripts/ablation_no_projection.py
python scripts/aggregate_results.py --out outputs/results_summary.csv
# PIV prep (only if regenerating from raw zip; .pt already present):
python -m rafm.data.prepare_piv --zip dataverse_files.zip --out_dir data/piv --grids 8x4,8x8,16x16
```
