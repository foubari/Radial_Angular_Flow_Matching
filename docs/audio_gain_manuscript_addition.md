# Measured AudioMNIST manuscript addition

The text below uses the completed [three-seed evaluation](../outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json)
and preserves the [archived comparison rows](../experiments/poc_audio/stage2_3seed.json).
The measured artifact retains `status="baseline_mismatch"` and
`archived_baselines_reproduced=false`; all three paired inference checks passed.

## Replacement AudioMNIST discussion

AudioMNIST is constructed as $X=GS$, with unit-norm STFT content $S$ and
gain $G$ sampled independently of both content direction and digit identity.
We therefore evaluate a direct control: generate $Y$ using the original
fixed-spherical model, then return $X=GY/\|Y\|$. The gain uses the same
linearly interpolated empirical quantile law as RAFM, fitted only to the
10,200 generator-training norms. It is drawn independently of the generated
direction and class; the 3,000 external test gains are used only for evaluation.
We reuse the original 24,000-step EMA checkpoints for seeds 8925, 1234 and 7,
without retraining or changing their trained radius, initialization or ODE.
Each seed generates 2,000 clips, 200 per digit, with sampling seed 0 and
40 RK4 steps (160 network evaluations per trajectory).

The gain preserves every classifier prediction across all 6,000 generated
clips: correct counts before and after are 1605/2000, 1601/2000 and 1634/2000.
The measured accuracy is $0.8066667\pm0.00735225$, where uncertainty denotes
population standard deviation across the three training seeds. Energy KS
decreases from 0.5923333 to 0.0218333, and mass above the external-test 95th
and 99th percentiles changes from zero to 0.0500 and 0.0145. The gain control
therefore has a higher measured mean accuracy than the historical RAFM-Ang
$0.764\pm0.025$ and RAFM-Vel $0.711\pm0.014$ rows, with the same KS and
upper-tail values at the reported precision. This is a descriptive comparison
on a benchmark with explicit gain/content independence; it establishes neither
statistical significance nor general superiority of post-processing. The
RAFM-Ang versus RAFM-Vel comparison remains a separate comparison of regression
targets under matched-radius training and sampling. Independent gain preserves
the fixed-spherical model's content predictions rather than improving them.

All model seeds share one gain vector drawn with gain seed 0, paired to the
original RAFM quantile draws. Consequently, their radial Monte Carlo variation
is shared: zero observed seed standard deviation for KS and tail masses does
not imply zero uncertainty in the empirical gain law.

The recovered fixed-spherical checkpoints do not exactly reproduce the archived
accuracy: the published $0.810\pm0.013$ remains unchanged, while reevaluation
gives $0.8066667\pm0.00735225$ both before and after gain replacement. Every
seed differs from its archived accuracy, as recorded below; the archived
baseline's rounded KS, tail and PIT values are reproduced. The new row and
figure are explicitly labelled as measured results with a baseline discrepancy.
The original training commit is unknown. Source and checkpoint compatibility
were audited, but this does not establish training provenance or explain the
accuracy difference. A backend/environment difference is a possible cause,
not a demonstrated explanation.

| Training seed | Archived fixed-spherical accuracy | Reevaluated accuracy, before and after gain |
|---|---:|---:|
| 8925 | 0.805 | 0.8025 |
| 1234 | 0.798 | 0.8005 |
| 7 | 0.828 | 0.8170 |

## Table 5 addition and caption

Keep all original rows unchanged. The new row at Table 5 precision is:

```latex
Fixed spherical + empirical gain$^{\dagger}$ & $0.807 \pm 0.007$ & $0.0218 \pm 0.0000$ & $0.0500 \pm 0.0000$ & $0.0145 \pm 0.0000$ \\
```

AudioMNIST at 24,000 training steps: mean and population standard deviation
over three training seeds, with 2,000 generated clips per seed. The added
control applies an independent training-ECDF gain after original fixed-spherical
generation and preserves all 6,000 classifier predictions. Historical values
and the RAFM-Ang/RAFM-Vel comparison remain unchanged. †The original-checkpoint
reevaluation differs from the archived fixed-spherical accuracy; the added row
reports actual measurements with this discrepancy disclosed, and baseline
reproduction is not established. Shared gain draws account for the common
radial metrics across training seeds.

## Figure 2 caption

AudioMNIST content accuracy versus energy KS at 24,000 training steps. Hollow
markers show individual training seeds; filled markers and error bars show
means and population standard deviations. The independent empirical-gain
control uses all three original fixed-spherical checkpoints and 2,000 clips
per seed. Its classifier predictions are unchanged by gain replacement.
Historical points, including separate RAFM-Ang and RAFM-Vel results, are
preserved. The new gain points are measured results with an explicitly
disclosed discrepancy in original-checkpoint accuracy relative to the archive;
they do not establish baseline reproduction. Gains are shared across seeds,
so radial error bars do not represent independent gain-sampling uncertainty.

## Runtime and reproducibility paragraph

Evaluation used one AMD Instinct MI210 in FP32, with PyTorch 2.7.1+ROCm 6.3.
The measured evaluation scope—data loading, compatibility checks, three
generations, paired evaluation and per-seed serialization—took 2683.44 seconds.
Slurm job 763355 occupied 47 minutes 03 seconds. The earlier cache-failure
attempt, job 763350, occupied 4 minutes 26 seconds and is preserved separately;
it produced no samples. These are checkpoint-evaluation times, not training
times or a matched runtime comparison with historical methods. Input,
checkpoint, source and output fingerprints, protocol settings, and per-seed
timings accompany the measured artifact. The [source-compatibility audit](audio_release_compatibility.md),
[Slurm accounting](../outputs_audio_gain/release_verification/slurm_accounting.txt)
and [retained cache-failure record](../outputs_audio_gain/release_verification/cache_failure_and_retry.json)
document provenance and execution limits.
