# PLAN.md — RAFM rebuttal experiment plan

Branch `rebuttal-experiments`. All outputs under `rebuttal_experiments/`. Existing `outputs/`, `configs/`, `data/`, manuscript files are read-only.

## Guiding principles
- **Attribution honesty.** RAFM's radial law = its source's radial law (tangent projection fixes norm). So radial metrics test the *source*; angular/global metrics (sliced-W1, angular-SW, MMD) test the *flow/path*. Every table separates the two.
- **Matched everything.** Same MLP (3×128, Swish), same optimizer budget (Adam 1e-3, 10k steps, batch **4096** to match the paper), same n_gen=10k, same metric settings, same seeds `{8925,77395,65457}` across methods.
- **No silent changes.** New solvers, metrics, datasets, and architectures are added as new files/configs. Existing code changed only if unavoidable and backward-compatibly.
- **Compute.** Single RTX 2000 Ada. FM run ≈ 0.5–2 min; MSGM ≈ ~50 min/run (10k steps). Runtime tags: **S** ≤5 min, **M** 5–30 min, **L** >30 min.

## Reviewer-concern → experiment map

| # | Reviewer concern (typical) | Experiment | Priority |
|---|---|---|---|
| R1 | "Are the submitted numbers reproducible / correctly measured?" | E1 Reproduction & independent metric verification | P0 |
| R2 | "What does the spherical path add over just fixing the source?" | E2 Component ablation ladder + attribution | P0 |
| R3 | "Radial gains are trivially from the eCDF source; is the *flow* actually better on directions?" | E3 Angular/direction-sensitive diagnostics (incl. common-radial diagnostic) | P0 |
| R4 | "Exp4 NFE curve is noise; how does quality scale with NFE and solver?" | E4 Fixed-checkpoint NFE×solver sweep + radius drift | P0 |
| R5 | "Toy2D NaN — is the method numerically fragile?" | E5 Toy2D / small-radius numerical diagnostics | P0 |
| R6 | "How does it behave as dimension grows?" | E6 Dimension scaling | P1 |
| R7 | "Sensitivity to training-set size?" | E7 Sample-size scaling | P1 |
| R8 | "Sensitivity to tail heaviness?" | E8 Tail-heaviness (Student-t df sweep) | P1 |
| R9 | "Sensitivity to anisotropy?" | E9 Anisotropy scaling | P1 |
| R10 | "Real-data (PIV) robustness: seeds, splits, angular/radial split, preprocessing." | E10 PIV extended | P1 |
| R11 | "Radial-sampler choice (eCDF vs interp vs KDE vs log)?" | E11 Radial-sampler ablation | P1 |
| R12 | "Bigger models?" | E12 Higher-capacity architectures | P2 |
| R13 | "Other realistic datasets (financial returns, image latents)?" | E13 New datasets (feasibility-gated) | P2 |
| R14 | "Radius–direction dependence regimes?" | E14 Radius-direction coupling suite | P2 |
| R15 | "Adaptive/other solvers?" | E15 Additional solvers (midpoint/adaptive) | P2 |
| R16 | "Compute & reproducibility accounting." | E16 Compute/repro ledger (rolled into every run + final) | P0 (ongoing) |

---

## P0 — run first

### E1 — Reproduction & independent metric verification
- **RQ.** Do current code + documented settings reproduce `RESULTS_SUMMARY.md` (Student-t d16/d32, Gaussian d16), and do independently recomputed metrics match the saved `metrics.json`?
- **Dataset.** Student-t d16 & d32 (df3, corr), Gaussian-aniso d16.
- **Methods.** gaussian_fm, source_only_{oracle,empirical}, rafm_{oracle,empirical}. MSGM: 1 seed only on d16 (cost).
- **Arch.** MLP 3×128. **Metrics.** full suite. **Seeds.** 3 (MSGM 1).
- **Compute/runtime.** FM part **M** (~15 runs × ~1 min). MSGM **M/L** (~50 min).
- **Impl work.** New configs (batch 4096) under `rebuttal_experiments/configs/E1`; reuse exp1 runner; a `verify_metrics.py` that recomputes metrics from saved `samples.pt` with an independent implementation (scipy).
- **Outputs.** `raw_results/E1_reproduction/…`, `tables/E1_repro_vs_paper.csv`.
- **Success.** Reproduced 3-seed means within ~1 std of RESULTS_SUMMARY; independent metrics within 1e-3 relative of saved. **Failure.** Systematic deviation → investigate (batch size, seed list).
- **Priority.** P0.

### E2 — Component ablation ladder + attribution
- **RQ.** Decompose the gain: source correction (Gaussian→source-only) vs spherical transport (source-only→RAFM). Quantify how much of radial vs angular improvement each step buys.
- **Datasets.** Student-t d16, d32. **Methods.** the 5 FM variants. **Metrics.** radial_W1/KS/tail (source effect) + sliced_W1/MMD/angular_SW (path effect) + train/val loss, train time, sample time, nan_rate. **Seeds.** 3.
- **Compute.** **M** (shares E1 runs — no re-train needed; adds val-loss logging + a decomposition table).
- **Impl.** `tables/E2_ablation_decomposition.*` from E1 raw results; add validation-loss eval hook via a small standalone eval script (does not modify trainer).
- **Success.** Clear, signed attribution table with uncertainty. Report even if spherical path adds little. **Priority.** P0.

### E3 — Angular / direction-sensitive diagnostics
- **RQ.** Independently of norms, are RAFM's generated *directions* better than baselines'? Does the flow help beyond the source?
- **New metrics (additive module).** cosine-distance stats to NN in test set; mean angular error; MMD on x/‖x‖; spherical sliced-Wasserstein on directions; NN angular distance; angular metrics per radial-quantile bin. **Diagnostic:** "common-radial" — replace every method's sample norms with draws from the *test* radial distribution while keeping generated directions, then recompute global/angular metrics. Explicitly labelled diagnostic, not the generation procedure.
- **Datasets.** Student-t d16/d32, Toy2D. **Methods.** all FM + MSGM (from saved samples). **Seeds.** 3.
- **Compute.** **S** (operates on saved `samples.pt`).
- **Impl.** `scripts/E3_angular_diagnostics.py` (+ `rebuttal_experiments/lib` metric helpers). No change to `rafm/`.
- **Success.** Direction-only ranking reported with CIs; state plainly whether flow beats source-only on directions. **Priority.** P0.

### E4 — Fixed-checkpoint NFE × solver sweep + radius drift
- **RQ.** Quality vs NFE ∈ {1,2,5,10,20,50,100} for euler/heun/rk4 on a *single fixed trained checkpoint* per method (removes the exp4 seed noise). Radius drift `|‖x_t‖−‖x_0‖|` along trajectories.
- **Datasets.** Student-t d16. **Methods.** rafm_empirical, gaussian_fm, source_only_empirical (+ MSGM NFE curve if checkpoint available). **Seeds.** 1 fixed checkpoint (reused), generation seed fixed.
- **Metrics.** radial_W1, sliced_W1, angular_SW, sample_time, NFE, radius-drift mean/max, nan_rate.
- **Compute.** **S/M** (no training; many samplings).
- **Impl.** `scripts/E4_nfe_solver_sweep.py` loading `checkpoint.pt`; a drift-instrumented sampler variant (new file, does not edit `rafm/flow_matching/sampler.py`).
- **Success.** Monotone-ish quality↑ with NFE for RK4; drift ≈0 for projected RAFM vs growing drift for unprojected/euclidean. **Priority.** P0.

### E5 — Toy2D / small-radius numerical diagnostics
- **RQ.** Reproduce and localize the Toy2D RAFM-oracle 18.4% NaN. Is it small-R slerp? Quantify NaN vs radius; test whether the existing `r_min` guard / a documented mitigation removes it (as a labelled numerical ablation, not a silent method change).
- **Dataset.** Toy2D (d=2). **Methods.** rafm_oracle, rafm_empirical, source_only, gaussian_fm. **Seeds.** 3.
- **Metrics.** nan_rate, exploding_norm_rate, NaN-vs-radial-bin, trajectory drift near origin.
- **Compute.** **S**.
- **Impl.** `scripts/E5_toy2d_nan_diag.py`; reuse trainer/sampler; instrument NaN origin.
- **Success.** NaN mechanism identified with evidence; mitigation quantified. Negative result (unfixable without method change) is acceptable and reported. **Priority.** P0.

---

## P1 — after P0

### E6 — Dimension scaling
d ∈ {2,8,16,32,64,128,256} Student-t(df3, corr). Methods: gaussian_fm, source_only_empirical, rafm_empirical. Metrics: full + train/sample time, peak GPU mem. Seeds 3 (or 2 if time). Runtime **M**. New configs `E6/*`. Success: map where RAFM's advantage grows/shrinks with d; report Gaussian-FM blow-up quantitatively.

### E7 — Sample-size scaling
n_train ∈ {500,1000,5000,10000,50000} on Student-t d16. Methods: gaussian_fm, source_only_{empirical,empirical_log,oracle}, rafm_{empirical,oracle}. Reuses exp2 design but under rebuttal configs. Metrics: radial + sliced + variance across 3 seeds. Runtime **M**. Success: eCDF→oracle convergence curve; small-n RAFM underfit quantified.

### E8 — Tail-heaviness scaling
Student-t df ∈ {1.5,2,3,5,10,50} (→ near-Gaussian), d16. Methods: gaussian_fm, source_only_empirical, rafm_empirical. Metrics: tail quantile err, exceedance, radial_W1, sliced_W1. Runtime **M**. New configs. Success: monotone story linking tail weight to RAFM advantage; include light-tail control where advantage should vanish.

### E9 — Anisotropy scaling
Gaussian-aniso and Student-t with mixing matrices of increasing condition number κ ∈ {1,3,10,30,100}. Metrics: global, covariance/spectral error, angular_SW, convergence. Runtime **M**. Impl: new dataset config exposing κ (build A with prescribed spectrum in a new data helper, additive). Success: whether spherical path helps/hurts as anisotropy rises.

### E10 — PIV extended
PIV d16/64/256: 3 seeds, extra split_seed ∈ {0,1,2}, NFE curve, angular-only & radial-only metrics, tail-event metrics, batch 4096, runtime/memory, all baselines (gaussian_fm, source_only_empirical, rafm_empirical, MSGM where feasible), plus a train-only-centering robustness variant (documented). Runtime **M/L**. Success: real-data confirmation or refutation of synthetic story, with seed/split uncertainty.

### E11 — Radial-sampler ablation
Source samplers: empirical eCDF (inverse-CDF interp, default), KDE, log-radius eCDF, log-radius KDE. On Student-t d16 + PIV d64. Compare radial fidelity + downstream RAFM angular. Runtime **S/M** (mostly reuses sampling). Treated as explicit ablation, official method unchanged. Success: which sampler best in tails without hurting bulk.

---

## P2 — if resources permit

### E12 — Higher-capacity architectures
Deeper (n_layers 6), wider (hidden 512), residual MLP (new additive module). Student-t d32, PIV d256. Compare stability/metrics/mem/runtime vs 3×128. Runtime **M**.

### E13 — New realistic datasets (feasibility-gated)
Write `dataset_notes/DATASET_FEASIBILITY.md` first. Candidates: daily multivariate financial log-returns (public, tiny, heavy-tailed, anisotropic — top pick); CIFAR-10 / DINO latents (heavier). Only proceed on feasible, license-clear, small candidates. Preserve raw+processed+indices; analyze tails/anisotropy/Gaussian-mismatch; run RAFM + baselines. Runtime **M/L**.

### E14 — Radius–direction coupling suite
Synthetic distributions with independent vs strongly-coupled radius/direction (radius-dependent vonMises concentration / mode). Measure conditional angular law per radial bin, coupling stats, per-bin quality, failure under strong coupling & multimodal conditional angle. Runtime **M**. Additive dataset generators.

### E15 — Additional solvers
Add midpoint (RK2) and an adaptive RK (Dormand-Prince / `torchdiffeq` if available, else fixed-step ladder) as a new sampler module; compare NFE/tolerance/runtime/drift/failures against euler/heun/rk4. Additive, labelled numerical ablation. Runtime **S**.

### E16 — Compute/repro ledger
Not a separate run: every run records config, command, git state, environment, seeds, timings, GPU mem, param count, checkpoint size; final report aggregates GPU-hours, inference time, memory, versions, hardware.

---

## Execution order (this session)
1. E1 (reproduce + verify) → 2. E2 (attribution table from E1) → 3. E3 (angular diagnostics on saved samples) → 4. E4 (NFE/solver on fixed ckpt) → 5. E5 (toy2d NaN). Then P1 E6→E8 as time permits; P1/P2 remainder documented in STATUS.md with reasons if not reached. MSGM runs are queued opportunistically (long).

Each run: dedicated config + run_id (timestamp) → smoke → full → save stdout/stderr/command/git_state/environment/seeds/metrics/timings/notes → update STATUS.md. Failures preserved under their run_id; fixes documented; rerun under new id. No fabricated values.
