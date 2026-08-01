# STATUS.md — RAFM rebuttal experiments

_Updated: 2026-07-27, round 2 (new real datasets + ablations)._

Branch `rebuttal-experiments` @ base `2e659c7`. Env: torch 2.6.0+cu124, RTX 2000 Ada. Added deps: zarr/xarray/numcodecs (for WeatherBench2).

Legend: DONE ✓ | RUNNING ▶ | REUSED ♻ | NOT-RUN — 

## Completed (round 1) — see FINAL_REPORT §5.1–5.8
E1 reproduction (d16/d32/gauss, within ~1 std; on-disk outputs used batch 256≠paper 4096), E2 attribution, E3 directional, E4 NFE/solver+drift (fixed source bug), E5 toy2d NaN, E6 dim-scaling (d2→256), E8 tail (df1.5→50), MSGM comparison.

## Completed (round 2)
| Exp | Status | Result |
|---|---|---|
| E10 PIV finalized | ✓ | RAFM best radial/KS (0.048 vs 0.19); ties source-only on sliced; path adds little on PIV directions. tables/E10_piv_d64, E3_angular/E10_piv_d64. |
| E-Finance (Ken French 49-Ind daily) | ✓ | NEW real, d=49, heavy-tail. Chronological split → radial shift floor (W1(train,test)=1.386): RAFM radial NEUTRAL (≈gaussian), but WINS all directional metrics. dataset_notes/finance_quick_check, tables/E_finance_ff49. |
| E-Weather (WeatherBench2 ERA5 AU wind) | ✓ | NEW real, d=96, LIGHT-tail but anisotropic (cond 1530). RAFM BEST on all metrics (radial 0.087 vs 0.18) — driven by dim+anisotropy, not tails. dataset_notes/weather_quick_check, tables/E_weather_au_wind. |
| E7 sample-size | ♻ (existing exp2) | RAFM needs n≥~5000 for radial edge; wins sliced from n≥1000. tables/E7_sample_efficiency. |
| E9 anisotropy sweep | ✓ | Gaussian base (light-tail), d32, cov cond 1→9·10⁴. RAFM edge grows with anisotropy alone: k300 radial 3.25 vs gaussian 37.4 (~11×); source-only degrades like gaussian → path essential. tables/E9_aniso, figures/E9_aniso.png. |

## Round 2 COMPLETE. Figures: E4_nfe_drift, E6_dim_scaling, E8_tail_scaling, E9_aniso, real_datasets_bars. FINAL_REPORT updated with §5.9–5.13 + dataset inventory + negative results + summary + repro commands. Git safety re-verified (only pre-existing user edits to tracked files; manuscript untouched).

## Not run (justified)
- WeatherAUS control: deprioritized — gridded weather (the *preferred* option per brief) succeeded, and WeatherAUS is a low-d heterogeneous tabular control needing Kaggle-auth; the brief says prefer gridded over low-d tabular. Documented, not run.
- Additional turbulence (JHTDB etc.): needs credentials/engineering; documented, not run.
- DINO/CLIP latents, omics: P2, not reached.
- Extra synthetic coverage (elliptical/mixtures/multimodal-radial): P2, partially covered by dim/tail/aniso sweeps.

## Next
1. Aggregate E9 → tables/figure; write §5.12.
2. Finalize FINAL_REPORT: dataset inventory, negative/neutral (finance), summary.
3. Refresh figures (dim/tail already; add finance/weather bars if time).

---
## Round 3 (direct reviewer answers) — 2026-07-28
| Exp | Status | Result |
|---|---|---|
| Higher-capacity (resMLP 256x4) on Weather | DONE | trend persists; RAFM radial arch-invariant (0.087); Gaussian-FM doesn't improve with capacity. tables/E_weather_bigmodel. |
| Finance chrono-vs-random split ablation | DONE | random floor 0.142 vs chrono 1.386; RAFM radial 1.42(chrono, neutral)->0.175(random, best). Proves neutrality = shift, not learning failure. tables/E_finance_randomsplit, diagnostics/finance_split_radial_shift.json. |
| NFE/inference curves real (PIV+Weather) | DONE | RAFM converges ~NFE 8; infer <0.05s. E4_nfe_solver/{piv,weather}_{rafm,gaussian}.csv. |
| MSGM finance+weather (matched budget) | RUNNING (bg, resumable) | 1 seed/dataset first (finance seed8925 → weather seed8925, ~12h total) to fill the table; extra 2 seeds deferred. 2.2s/step, checkpoints every 2000 steps. Aggregate via aggregate.py (picks up msgm/ dir). |
| WeatherAUS / turbulence JHTDB | NOT RUN | per instruction (gridded weather preferred; no credentials). |

Figures added: round3_nfe_capacity.png. Report §5.14–5.17 + updated negative-results/summary.
Git safety re-verified: only pre-existing user edits on tracked files; manuscript untouched.

## MSGM detached/resumable (2026-07-28)
- Resume verified: adapter reloads msgm_ckpt.pt (restarted at step 2000, not 0). ckpt_every reduced 2000->1000.
- Detached launcher: rebuttal_experiments/scripts/msgm_resume.bat (via PowerShell Start-Process, hidden) — survives terminal/session teardown; self-healing retry loop; exits when both seeds' metrics.json exist.
- To (re)start/continue MSGM anytime: double-click msgm_resume.bat (auto-resumes, skips completed). Log: logs/msgm_resume.log.
- Order: finance seed8925 (~5h) -> weather seed8925 (~6h). Aggregate with aggregate.py when metrics.json appear.

## MSGM COMPLETE (2026-07-30)
All 6 MSGM runs done (finance 3 seeds + weather 3 seeds, full budget 10k steps).
Finance MSGM: radial 1.417+/-0.033, ks 0.197, sliced 0.182 (tied w/ RAFM at shift floor).
Weather MSGM: radial 0.099+/-0.028, ks 0.033, sliced 0.120 (radial~tied RAFM, worst sliced).
Net: MSGM competitive on radial/KS, RAFM best on sliced/MMD + ~540x faster. Tables/figures/§5.10-5.11-5.17 updated. real_datasets_bars.png = MSGM 3-seed.

## Tables 2 & 3 with mean±std (2026-07-30)
- tables/E6_radialW1_by_dim.md — E6 dimension scaling radial_w1 mean±std (d2..d256). Report §5.7 updated.
- tables/E3_cr_sliced_w1.md — E3 common-radial cr_sliced_w1 mean±std (gaussian/source/msgm/rafm). Report §5.3 updated.
- tables/MASTER_results.md — consolidated all-benchmarks × all-methods (radial/sliced/ks/train).

## Single aggregate file (2026-07-30)
- tables/ALL_RESULTS.md — every aggregated results table in one file (19 sections). Built by scripts/build_all_results (inline).
