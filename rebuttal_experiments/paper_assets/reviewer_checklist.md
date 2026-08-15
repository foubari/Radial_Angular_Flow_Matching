# NeurIPS reviewer-request audit

Status per substantive request. ✅ fully addressed · 🟡 partial/indirect · ❌ absent.
Each remaining 🟡/❌ is tagged: **[add]** worth adding before submission · **[cheap]** cheap but optional ·
**[skip]** unnecessary given the stronger Angular RAFM story · **[expensive]** prohibitively expensive.

| # | Request | Status | Evidence / where |
|---|---------|--------|------------------|
| 1 | Higher-dimensional experiments | ✅ | E6 dim sweep $d\in\{2..256\}$; DC-AE latents (2048-d); AudioMNIST STFT (16254-d) |
| 2 | Tail-heaviness sweep | ✅ | E8 Student-$t$ dof $\in\{1.5..50\}$, 3 seeds, all methods + Angular |
| 3 | Anisotropy | ✅ | E9 condition number $\kappa\in\{1..300\}$ |
| 4 | Real datasets | ✅ | PIV, Weather, Finance (+ Finance random-split diagnostic) |
| 5 | Larger model / capacity ablation | ✅ | Weather ResMLP $256\times4$ (576K params, `E_weather_bigmodel`) |
| 6 | Source-only / matched-source ablations | ✅ | `source_only_{empirical,oracle}` everywhere; `matched_euclidean` on DC-AE + AudioMNIST |
| 7 | MSGM comparison | 🟡 **[expensive]** | Present on Finance + Weather; remaining ≈410 GPU-h (~525× RAFM), documented in `ANGULAR_AUDIT`/cost note. Not run |
| 8 | Non-radial / directional metrics | ✅ | sliced $W_1$, angular SW, FID/KID/precision/recall/coverage, digit accuracy |
| 9 | NFE / solver-step dependence | ✅ | E4 sweep: euler/heun/RK4 × NFE grid, std + Angular (`E4_nfe_solver`) |
| 10 | Tangent projection / radial drift | ✅ | E4 drift vs NFE, projection on/off; `fig_nfe_drift`; audit doc |
| 11 | Computational cost / runtime | ✅ | `efficiency.json` + `table_efficiency.tex`; MSGM measured 18–20k s/seed |
| 12 | FLOPs | ✅ | Analytic MAC counts, all backbones (`efficiency_profile.py`) |
| 13 | Image experiment | ✅ | DC-AE ImageNette, SiT, FID/KID/prdc via torch-fidelity |
| 14 | Limitations | ✅ | d=2 instability, heavy-tail radial cost, MSGM cost — `ANGULAR_AUDIT.md` |
| 15 | Notation clarity | 🟡 **[cheap]** | Scale-free target $A=\dot X_t/\lVert X_t\rVert$ defined + theory fig; final wording is a manuscript-rewrite task |
| 16 | Requested references | ❌ **[cheap]** | Needs the specific reviewer reference list; add during manuscript rewrite (no experiment) |
| 17 | DPM-Solver / AMED-Solver | 🟡 **[skip]** | See assessment below |

## DPM-Solver / AMED-Solver — assessment

**Recommendation: not needed for this paper.** Reasoning:

- DPM-Solver and AMED-Solver are *fast-sampling* solvers tailored to **diffusion SDEs / probability-flow
  ODEs of score models**. RAFM / Angular RAFM are **flow-matching ODEs**; the paper's contribution is the
  **training target** (scale-free angular regression), not the sampler.
- The reviewer concern behind "which solver / how many steps" is **NFE–quality dependence**, which is
  already characterized by the **E4 euler/heun/RK4 × NFE sweep** (`fig_nfe_drift`, `E4_nfe_solver`). It
  shows both std and Angular converge by ~100 evaluations and quantifies radial drift vs NFE.
- Adding DPM/AMED would change the *sampler*, not any conclusion about the *target*; std-RAFM and Angular
  would benefit equally, leaving all reported gaps unchanged. It would add engineering surface for no
  scientific delta.

If a reviewer explicitly insists, a DPM-Solver-style multistep run on the DC-AE flow is **[cheap]**
(sampler-only, hours) — but classify it as reviewer-appeasement, not evidence that changes the story.

## Summary of remaining items

- **Worth adding before submission [add]:** none experimentally. The two writing items (#15 notation, #16
  references) are done during the manuscript rewrite.
- **Cheap but optional [cheap]:** #15, #16 (writing); a single DPM-Solver DC-AE sampling run if demanded.
- **Unnecessary [skip]:** #17 DPM/AMED as evidence.
- **Prohibitively expensive [expensive]:** #7 full MSGM sweep (~410 GPU-h) — leave missing, documented.
