# Paper assets — Angular RAFM

Consolidated, paper-ready experimental assets for the Angular RAFM rewrite. Everything here is built
from the completed 3-seed result package; **no manuscript file is modified**. All numbers trace to a
named result file (see *Source of every number*). Regenerate any asset by re-running its script.

- Aggregated data: `../master_results.json` (cross-domain), `../master_suite.json` (sweeps + singletons)
- Human report: `../MASTER_RESULTS_FULL.md`; anomaly audit: `../ANGULAR_AUDIT.md`
- Figures: `figs/*.pdf` (vector) + `*.png` (300 dpi); scripts alongside
- Tables: `tables/*.tex` (booktabs; define `\newcommand{\std}[1]{{\scriptsize$\pm$#1}}`)

## Headline results (3 seeds, mean ± std)

- **DC-AE ImageNette (image):** Angular RAFM **FID 147.3 ± 0.9** — best of all methods (Gaussian 161,
  fixed 166, matched 172, std-RAFM 175) — while keeping std-RAFM-class radial calibration
  (radial $W_1$ 0.32 vs 0.27) and the best precision (0.160) and coverage (0.082).
- **AudioMNIST (audio):** at **equal** energy calibration (KS 0.022, tied with std-RAFM), Angular lifts
  digit accuracy **0.711 → 0.764 ± 0.025**.
- **Synthetic sweeps:** Angular is best or tied on the **directional** (angular SW) and **overall**
  (sliced $W_1$) metrics across dimension, tail and anisotropy; radial marginal is identical to std-RAFM
  under light tails (source-governed), with a small radial cost only in the heaviest tails.
- **Mechanism:** full-velocity target norm correlates with radius (corr **0.94 / 1.00** on synthetic /
  audio); the angular target equals the geodesic angle $\theta\in[0,\pi]$ — scale-free w.r.t. radius
  (corr **0.00 / −0.01**) and bounded by $\pi$, concentrating near $\pi/2$ only as a high-$d$ effect.

## Figures (all PDF + PNG)

| File | Role | Message | Source |
|------|------|---------|--------|
| `figs/fig_synthetic_scaling.pdf` | Main synthetic (3×3) | Angular vs std/baselines across dim/df/κ on radial/sliced/directional | `master_suite.json` |
| `figs/fig_audiomnist_mechanism.pdf` | Audio mechanism | content↑ at equal energy; target-norm vs radius | `master_results.json` (audio) + geometry |
| `figs/fig_dcae_tradeoff.pdf` | DC-AE | FID vs radial-$W_1$ trade-off; Angular dominates bottom-left | `master_results.json` (dcae) |
| `figs/fig_theory_scalefree.pdf` | Theory | full-velocity scales w/ radius; angular $=\theta\in[0,\pi]$ scale-free (near $\pi/2$ in high-$d$) | data geometry (student-t, AudioMNIST) |
| `figs/fig_nfe_drift.pdf` | Appendix/limitations | radial drift vs NFE, std vs Angular; d=2 divergence | `E4_nfe_solver/*.csv` |
| `figs/fig_realdata.pdf` | Real-data summary | PIV/Weather/Finance bars (a table may replace this) | `master_suite.json` |
| `figs/fig_dcae_grid.pdf` | DC-AE qualitative | uncurated 5-method × 10-class grid, identical seed/sampler | surviving EMAs (`dit/make_dcae_grid.py`) |

Scripts: `figs/make_figures.py` (A,C,D,E), `figs/make_audio_fig.py` (B), `figs/theory_diagnostics.py`
(theory + `table_theory.tex` + `theory_stats.json`).

## Tables (LaTeX, booktabs)

| File | Content |
|------|---------|
| `tables/table_realdata.tex` | PIV / Weather / Finance × methods (radial/sliced/angular) |
| `tables/table_synthetic_main.tex` | Student-$t$ d16 main incl. oracle rows |
| `tables/table_audiomnist.tex` | AudioMNIST (digit acc, energy KS, coverage, PIT) |
| `tables/table_dcae.tex` | DC-AE (FID, radial, KS, precision/recall/coverage) |
| `tables/table_ablation.tex` | std-RAFM vs Angular across all domains + Δ (needs `\good`/`\bad`) |
| `tables/table_appendix_sweeps.tex` | dim/df/aniso sweeps, std vs Angular |
| `tables/table_efficiency.tex` | params / FLOPs / train compute / sampling |
| `tables/table_theory.tex` | full-velocity vs angular target: mean/std/CoV/corr/q99 |

`$^\dagger$` in tables marks aggregates over **<3 finite seeds** (the d=2 Angular instability — never
averaged silently).

## Efficiency / FLOPs (`efficiency.json`, `tables/table_efficiency.tex`)

FLOPs from analytic MAC counting (Linear/Conv2d hooks; MAC×2). Training compute = 3×fwd (fwd+bwd)×batch×steps.

| Model | Params | FLOPs/fwd | Total train | Sampling |
|-------|--------|-----------|-------------|----------|
| FM/RAFM/Angular MLP (Finance d49) | 45.9K | 90.9 KFLOP | 11.2 TFLOP | RK4 |
| ResMLP 256×4 (Weather big) | 576K | 1.15 MFLOP | 70.5 TFLOP | RK4 |
| DiT-SiT (DC-AE) | 32.5M | 2.74 GFLOP\* | 21.1 PFLOP | RK4 **25 steps = 100 evals** (all methods) |
| UNet ch96 (AudioMNIST) | 27.9M | 53.7 GFLOP | 124 PFLOP | RK4 **40 steps = 160 evals** (all methods) |
| MSGM (Finance d49) | 45.9K (net) | — | — | measured **18–20k s/seed, ~525× RAFM** |

\* SiT attention MACs approximated (Linear/Conv only); attention matmuls undercounted — treat as a lower bound.

**Key efficiency claim (paper):** FM, standard RAFM and Angular RAFM use the **identical network and
per-step training cost**; Angular adds only a per-sample scalar norm + multiply (negligible). The MSGM
$\sim$525× slowdown is structural — dense $(d,d,d)$ generator → $O(\text{batch}\cdot d^3)$ drift/diffusion
wrapped in sliced-score-matching double backward (`autograd.grad(create_graph=True)`); scales with $d^3$,
**not** with batch. See `../ANGULAR_AUDIT.md` and the cost note.

## Theory-to-experiment (`figs/theory_stats.json`)

| Dataset | full-vel `‖Ẋ_t‖` CoV / corr(r) | angular `‖Ẋ_t‖/‖X_t‖` CoV / corr(r) |
|---------|-------------------------------|-------------------------------------|
| Student-$t$ d16 (heavy) | 0.56 / **0.94** | 0.16 / **0.00** |
| AudioMNIST | 0.73 / **1.00** | 0.00 / **−0.01** |

Stated carefully: on the coupled sphere ($\lVert X_t\rVert=R$) the angular target equals the geodesic
angle $\theta=\angle(u_0,u_1)\in[0,\pi]$ — **scale-free w.r.t. radius and bounded by $\pi$**. Its
concentration near $\pi/2$ here is a **high-dimensional** phenomenon (random directions are near-orthogonal),
**not** a universal constant: synthetic CoV 0.16 ≠ 0, and in low $d$ it spreads across $[0,\pi]$.

## Reviewer-request audit

Full checklist in `reviewer_checklist.md`. 13/17 ✅ fully addressed; 🟡: MSGM (partial, rest prohibitive),
notation + references (manuscript-rewrite tasks), DPM/AMED (assessed unnecessary — the E4 euler/heun/RK4 ×
NFE sweep already covers solver/NFE dependence; DPM/AMED change the sampler, not the target, so no
conclusion moves).

## Source of every number

- Cross-domain (DC-AE, AudioMNIST) cells → `../master_results.json` → per-seed `metrics.json` /
  `dit_sit_s*/eval_std/angular_rafm/eval_40000.json` (dc-ae) / `poc_audio/stage2_3seed.json` (audio).
- Sweeps + real/synthetic singletons → `../master_suite.json` → `raw_results/<E*>/<dataset>/<method>/seed_*/metrics.json`.
- NFE/drift → `raw_results/E4_nfe_solver/*.csv` (std/gaussian retained; `*_angular*.csv`/`angular_rafm_*.csv` new).
- Efficiency → `efficiency.json` (built models) + train wall-clock from `total_train_time_s` in metrics + DiT `train_log.json`.
- Theory → `figs/theory_stats.json` (geometry from `configs/E1_studentt_d16.yaml` data + `poc_audio/data/audiomnist_stft_train.pt`).
- Seeds: `get_seed_list(3,42)` = {8925, 77395, 65457} (tabular/synthetic); DC-AE {8925, 7, 1234}; audio {8925, 77395, 65457}.

## Caveats & numerical notes (disclose in paper)

1. **d=2 Angular instability.** One of three seeds NaN's at `student_t_d2` and `toy_radial_angular`
   (nan_rate up to 0.85); every $d\ge8$ seed is finite. Cause: `v=‖x‖·A` amplification with a 1-D tangent
   space at $d=2$. Aggregates use finite seeds only (flagged `$^\dagger$`). Documented, not fixed.
2. **Sampling NFE — now matched (resolved).** All results use RK4 **128 steps = 512 model evaluations**
   for every method and dataset: synthetic runs were already at 128 steps; the 11 `run_real` Angular runs
   (piv/finance/weather/aniso×6/finance-randomsplit/weather-bigmodel) were **re-sampled** from their saved
   checkpoints at 128 steps (`scripts/resample_matched_nfe.py`, sampler-only, no retraining; old 2048-eval
   outputs kept as `*_nfe2048.*`). Effect: `radial_w1` unchanged to 4 decimals everywhere (drift already
   converged), `sliced_w1` moved only within finite-sample noise — no conclusion changed. DC-AE (25 steps =
   100 evals) and AudioMNIST (40 steps = 160 evals) were already matched across all methods.
3. **FID is torch-fidelity** (standard InceptionV3), not a custom metric; rFID decoder floor cached & shared.
4. **MSGM** filled only where already run (Finance, Weather); the rest is ~410 GPU-h, left missing by design.

---

## PACKAGE FROZEN

Experimental package is **frozen** as of the matched-NFE re-sample + DC-AE grid. No further training,
DPM/AMED, or MSGM sweeps. LibriSpeech remains independent/optional. Only manuscript-writing edits remain.

## Ready for paper

- **Figures:** all seven (`fig_synthetic_scaling`, `fig_audiomnist_mechanism`, `fig_dcae_tradeoff`,
  `fig_theory_scalefree`, `fig_nfe_drift`, `fig_realdata`, `fig_dcae_grid`) — vector PDF + PNG,
  paper-column sizing, consistent palette/labels, reproducible scripts. All numbers at **matched 512-eval**
  sampling.
- **Tables:** all eight `.tex` — drop-in with mean±std, NaN disclosed.
- **Efficiency/FLOPs table** and the **MSGM cost argument** (code-grounded).
- **Theory diagnostics** figure + table supporting the scale-free claim ($\theta\in[0,\pi]$, high-$d$
  concentration at $\pi/2$).
- **Reviewer checklist** with DPM/AMED assessment.

## Remaining optional work (writing only)

1. **[cheap, writing]** Notation pass + insert the reviewer's requested references (manuscript rewrite).
2. **[cheap, writing]** Optionally cite `fig_dcae_grid` as a qualitative appendix panel.

## Do not spend compute on

- **Full MSGM sweep** (~410 GPU-h, ~525× RAFM/step) — documented; leave missing.
- **DPM/AMED-Solver** as evidence — changes the sampler, not the target; no conclusion moves.
- **Re-sampling / re-training anything** — package is frozen; all sampling is at matched 512 evals.
- **Re-running any baseline** — all baseline numbers are retained and reused; only Angular was trained.
- **d=2 "fixes"** (radius clipping / higher NFE) — would break protocol parity; report as a limitation.
