# PoC-B — controlled speech-energy (AudioMNIST): does RAFM help when magnitude IS the signal?

Tests the hypothesis from the DC-AE screening: RAFM should help precisely when the **global magnitude is
perceptually meaningful and the decoder is magnitude-sensitive** — the regime DC-AE lacked. Here the
magnitude is **clip energy/loudness**, by construction.

## Design

- **Data:** AudioMNIST (spoken digits, HF `gilkeyio/AudioMNIST`), resampled 8 kHz, 1 s. Reversible
  **complex STFT** (n_fft 256, hop 128 → `(2,129,63)`, D=16254); `torch.stft/istft` round-trip
  **rel-err 1.6e-07**.
- **Construction `x = g·s`:** `s` = unit-norm STFT (content = digit/speaker = DIRECTION); `g` = loudness
  (= global RADIUS), drawn from a **bimodal heavy-tailed** law (modes ≈1 and ≈4, 40 % loud). Then
  `‖x‖ ≡ g` exactly (verified 3e-06). No centering (keeps radius = energy). Gaussian baseline
  **moment-matched** (σ=E[g]/√D) for fairness.
- **Model:** digit-conditioned conv velocity net (0.5 M) on `(2,129,63)`, identical across methods, 12 k
  steps, EMA, CFG=1 sampling (RK4, 40 steps, tangent-projected for spherical). 1 seed (8925).
- **Eval:** energy-invariant digit classifier (log-magnitude, **95.5 %** on real content); n=2000 gen/method.

## Results (`eval/*/eval.json`, figure `figures/pocB_energy_vs_content.png`)

| method | energy W1 ↓ | energy **KS** ↓ | cov>q95 (t .05) | cov>q99 (t .01) | cov<q10 (t .10) | PIT (t .5) | **digit acc** (real .955) |
|---|---|---|---|---|---|---|---|
| gaussian_euclidean | 0.565 | 0.190 | 0.009 | 0.001 | 0.155 | 0.468 | 0.208 |
| matched_euclidean | 0.097 | 0.062 | 0.047 | 0.014 | 0.150 | 0.471 | 0.212 |
| fixed_spherical | 1.488 | 0.592 | 0.000 | 0.000 | 0.000 | 0.592 | 0.231 |
| **rafm** | **0.049** | **0.022** | **0.050** | **0.014** | **0.098** | **0.492** | 0.225 |

### Energy axis — RAFM wins decisively (mechanism CONFIRMED)

- **RAFM matches the bimodal heavy-tailed energy law best**: KS 0.022 (≈9× better than Gaussian's 0.190),
  tail coverage ≈ targets (q95 0.050, q99 0.014, low 0.098), PIT 0.492 ≈ 0.5.
- **Gaussian FM cannot reproduce the bimodal energy**: it collapses toward the mean, **severely
  under-covering the loud tail** (cov>q95 0.009 vs 0.05) and over-covering low energy (0.155 vs 0.10),
  PIT 0.47. A moment-matched Gaussian source is unimodal — structurally unable to produce a bimodal radius.
- **fixed_spherical** is a point mass (KS 0.59, zero tail spread).
- matched_euclidean (empirical radial source, Euclidean path) is also good — the **radial source** is what
  fixes energy; RAFM adds the spherical path and is best overall.

**This is the opposite of the DC-AE image result** (where RAFM's radial win was perceptually irrelevant).
Here the magnitude *is* the signal and the decoder (ISTFT) is magnitude-sensitive, so **RAFM's energy
calibration is real and the pro-RAFM hypothesis holds on the energy axis.**

### Content axis — limited by the generator, uniformly across methods (honest negative)

All four methods reach only **digit accuracy ≈ 0.21–0.23** (vs real 0.955; chance 0.10). Spectrograms
(`eval/*/spectrograms.png`) show digit-like formant structure with correct energy scaling but too noisy to
be cleanly recognizable. Because accuracy is **uniform across methods**, content quality is set by the
**backbone/budget** (0.5 M conv, 12 k steps, hard complex-STFT-with-phase target), **not** by the
transport. → RAFM's energy win comes at **no content cost relative to the baselines**, but the absolute
"preserves linguistic content" claim **cannot be made yet** at this generator scale.

## UPDATE — expressive UNet (27.9M) sweep (supersedes the pilot on content)

Same data / gain construction / 4-way / conditioning / eval; only the backbone is bigger: **UNet ch=96,
27.9 M params** (56× the 0.5 M pilot), batch 32, **24 000 steps = 768 k examples (matches pilot)**,
identical arch/init/optimizer/data-order across methods (seed 8925), CFG=1. Checkpoints 12/18/24k, n=2000.

**Digit classifier: 95.5 % on real content.** Results at 24k (`eval_unet/*/step24000/eval.json`,
figure `figures/pocB_unet_trajectory.png`):

| method | energy **KS** ↓ | cov>q95 (.05) | cov<q10 (.10) | PIT (.5) | **digit acc** | acc lo/mid/hi |
|---|---|---|---|---|---|---|
| gaussian_euclidean | 0.122 | 0.059 | 0.174 | 0.482 | 0.739 | .65/.74/.83 |
| matched_euclidean | 0.110 | 0.039 | 0.200 | 0.443 | 0.736 | .68/.76/.76 |
| fixed_spherical | 0.592 (point mass) | 0.000 | 0.000 | 0.592 | **0.805** | .75/.79/.88 |
| **rafm** | **0.022** | **0.050** | **0.098** | **0.493** | 0.692 | .67/.69/.72 |

**Content is solved by capacity:** digit accuracy jumped from the pilot's ~0.21 to **0.69–0.81** for all
methods. So content generation was an architecture/budget limitation, not a transport effect.

**Energy calibration (RAFM's expected advantage):** RAFM is best by a wide margin — KS **0.022** vs
matched 0.110 (5×) and gaussian 0.122; tails ≈ targets (q95 0.05, q99 0.014), PIT ≈ 0.5. Gaussian is
**not** structurally unable to model the bimodal energy — with capacity it reaches KS 0.12; it simply
calibrates worse than RAFM **under this architecture and budget**. (RAFM's energy metrics are
seed/step-stable because the generated radius is source-determined by the empirical-radial source.)

**RAFM vs matched-Euclidean (the key source-matched-path comparison):**
- **Energy: RAFM outperforms matched decisively** (KS 0.022 vs 0.110; PIT 0.493 vs 0.443; low-tail
  coverage 0.098≈target vs matched's 0.200 over-coverage). ✅ — the expected radial-calibration win holds
  with a strong generator.
- **Content: RAFM is modestly *below* matched, not fully competitive** — 0.692 vs 0.736 (~4-5 pts,
  consistent across 12/18/24k). RAFM's accuracy is also **flattest across energy** (.67/.69/.72) whereas
  the others rise with energy.

→ **This is a content↔energy trade-off, not a clean "win energy + tie content."** Diagnostic:
`fixed_spherical` (same spherical *path*, but *fixed* radius) has the **best** content (0.805) and the
**worst** energy — so the spherical **path** aids content, while RAFM's varying-radius empirical **source**
(what fixes the energy law) is what both calibrates energy and modestly costs content here.

## Verdict

- ✅ **Content criterion met:** digit accuracy improved substantially (≈0.21 → 0.69–0.81) with the larger
  backbone — content was a capacity limit.
- ✅ **RAFM's energy-calibration advantage is real and robust** with an expressive generator — decisively
  better than matched-Euclidean and gaussian (KS 0.022 vs 0.110 / 0.122). This validates
  [[rafm-pro-dataset-hypothesis]] on the energy axis and contrasts with the DC-AE image negative.
- ⚠️ **The joint claim is only partially met:** RAFM does **not** stay fully competitive with
  matched-Euclidean on content (0.692 vs 0.736); there is a **modest content cost** attached to RAFM's
  energy win in this setup. Honest statement: *RAFM buys a large, correct energy-tail calibration at a
  small but consistent content penalty vs the source-matched Euclidean baseline.*
- **Gaussian language (per protocol):** failed to match the bimodal energy **under this architecture/
  budget** (KS 0.12), NOT "structurally unable."

**Open follow-ups (not run):** more seeds for CIs on the ~4-pt content gap; longer training (content still
creeping up 0.64→0.69 for RAFM); check whether a lighter tangent-projection / path variant recovers the
content gap while keeping the energy win.

## Configs / seeds / compute / limits
1 seed (8925), 12 k steps × 4 methods ≈ 4.7 h (351 ms/it), eval ~40 min. Backbone 0.5 M. Data 12 k train /
3 k test. **Limitations:** single seed, small backbone, complex-STFT generation is hard; content result is
a capacity limitation, not a transport finding. Reversible/reproducible: `audio_data.py`, `audio_flow.py`,
`audio_classifier.py`, `audio_eval.py`, `launch_pocB.bat`.
