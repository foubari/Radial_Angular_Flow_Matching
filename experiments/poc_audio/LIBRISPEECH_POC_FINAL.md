# LibriSpeech PoC — final go/no-go

**Question.** Does the Angular RAFM mechanism (retains radial calibration of standard RAFM while
improving content) transfer from AudioMNIST to a harder real-speech task (LibriSpeech word-conditioned)?

Finished from the existing **5k** checkpoints. No redesign, no retrain from scratch.

## Setup used (exact)

- Checkpoints: `runs_libri5/matched_euclidean/ema_5000.pt`, `runs_libri5/rafm/ema_5000.pt`,
  `runs_libri5_ang/rafm/ema_5000.pt` (UNet, ch96, depth5, ncls20; angular flag read from each run's `meta.json`).
- Data: controlled `x = g·s`, reversible complex-STFT (2×129×63, D=16254), `‖x‖ ≡ g` (energy=radius).
  Test set 827 clips.
- Evaluation set: **n = 300**, class-balanced across the 20 words, fixed seed 0, identical across methods.
- Sampler: RK4, tangent-projected for spherical. **Two budgets** to rule out the sampling confound:
  - nfe = **6 steps = 24 model forward calls**;
  - nfe = **40 steps = 160 model forward calls** (AudioMNIST's budget), std + Angular.
- Content metric: pretrained Whisper-tiny.en word-presence (gain-robust), + on **real** reference clips.
- Runtime: ~4 min for the 3-method nfe6 pass; ~15 min for the 2-method nfe40 pass; diagnostics ~3 min. Single GPU.

## Radial / energy (5k)

| Method | radial $W_1$ ↓ | energy KS ↓ | PIT | cov$>$q95 | cov$>$q99 | cov$<$q10 | radial drift ↓ |
|--------|----------------|-------------|-----|-----------|-----------|-----------|----------------|
| Matched-Eucl. (nfe6) | 0.329 | 0.120 | 0.427 | 0.017 | 0.007 | 0.180 | 0.0874 |
| RAFM std (nfe6/nfe40) | 0.186 / 0.185 | 0.068 | 0.471 | 0.020 | 0.007 | 0.097 | 0.00052 |
| **Angular (nfe6/nfe40)** | 0.184 / 0.185 | 0.067 | 0.471 | 0.020 | 0.007 | 0.097 | 0.00102 |

→ **The geometric mechanism transfers cleanly:** Angular ≈ standard RAFM on every radial metric and on
drift ($\sim$0.001 vs 0.0005, both ≪ matched's 0.087), and both clearly beat matched-Euclidean
(radial $W_1$ 0.18 vs 0.33). Angular retains standard RAFM's radial calibration on LibriSpeech.

## Target-norm diagnostics (geometry on LibriSpeech data)

| target | corr with radius | CoV | median |
|--------|------------------|-----|--------|
| full-velocity $\lVert\dot X_t\rVert$ (std) | **1.00** | 0.724 | — |
| angular $\lVert\dot X_t\rVert/\lVert X_t\rVert$ | **0.006** | 0.005 | 1.571 ($\approx\pi/2$) |

→ The scale-free property holds on LibriSpeech: full-velocity scale is fully coupled to radius; the
angular target is radius-scale-free and bounded (here concentrating near $\pi/2$ as a high-$d$ effect;
the general bound is $\theta\in[0,\pi]$).

## Content (Whisper word-presence)

| | Matched | RAFM std | Angular | **Real reference** |
|--|---------|----------|---------|--------------------|
| word-presence @ nfe6 | 0.007 | 0.017 | 0.013 | **0.83** |
| word-presence @ nfe40 | — | 0.010 | 0.007 | 0.83 |
| word classifier acc (chance 0.05) | 0.053 | 0.053 | 0.043 | — |

→ **All three methods are essentially unintelligible** (word-presence ≈ 1% vs 83% on real clips; the
word classifier is at chance). The content floor is **robust to sampling budget** (nfe6 ≈ nfe40), so it
is not a sampling artefact. Angular vs standard cannot be separated on content because **both are at the
floor**.

## Uncurated examples

Per method: `eval_unet/<run>_<method>/step5000/spectrograms.png` + low/mid/high-energy `.wav` files
(3 fixed seeds each, selected purely by energy quantile — no curation). Spectrograms show energy/tail
structure consistent with the radial metrics but no coherent word formants at 5k.

## Diagnosis (which cause)

- **Evaluator limitation — ruled OUT.** Whisper scores **0.83** on real clips; it works.
- **Transport — ruled OUT.** Radial calibration and the scale-free target mechanism both transfer
  cleanly (Angular ≈ std RAFM; both beat matched). The failure is not in the RAFM/Angular transport.
- **Sampling budget — ruled OUT.** Content is at the floor at both nfe6 (24 calls) and nfe40 (160 calls).
- **Insufficient optimization (PRIMARY) + task difficulty.** 5k steps is far below what content needs:
  AudioMNIST (10 clean spoken digits) only reached 0.76 digit-accuracy at **24k**; LibriSpeech is a
  substantially harder real-speech task (20 words, real recordings, same small UNet). At 5k the model
  has learned the energy/radial law but not the fine spectral content, for **every** method.

## Decision

Per the pre-agreed rule ("if all generated content remains essentially unintelligible at 5k, do not
spend another day blindly training; return NO-GO at current scale and diagnose"): content is robustly at
the floor for all methods and all sampling budgets, so the **content**-improvement half of the mechanism
**cannot be observed at this scale**, and reaching intelligibility would require ≳24k steps on a harder
task — compute that is **not justified**, because the frozen main package already carries two strong
transfer results (AudioMNIST for audio, DC-AE ImageNette for image) plus the synthetic sweeps.

What LibriSpeech *does* confirm cheaply: the **geometric** mechanism (radial calibration + scale-free
target) transfers to real-speech data. What it does **not** provide at feasible PoC scale: a resolvable
content-quality advantage, because content is unintelligible for every method at 5k.

Not extended to 12k: the content floor is nfe-invariant and at chance (not "almost there"), so a 12k
point is unlikely to resolve an Angular-vs-std content signal and would be the "blind training" the rule
warns against. The three runs remain resumable (`ckpt.pt`) if a future, dedicated real-speech study wants
to push to 24k+.

---

**NO-GO — LibriSpeech does not currently provide sufficiently reliable additional evidence to justify more compute.**
