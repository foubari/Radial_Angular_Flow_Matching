# Proposed manuscript additions

These passages describe the prepared experiments. They contain no measured
t-Flow or post-hoc-gain results, and are not evidence that those experiments
have run. Existing values in `RAFM_ICLR_2027.pdf` remain final, including rows
with temporary comments or colored text. Use the results paragraphs only after
the corresponding artifacts and complete seed coverage are available.

## Planned t-Flow baseline: methods paragraph

We will compare RAFM with the noise-prediction t-Flow formulation of Pandey
et al. The source is a multivariate Student-t random vector
\(N=sZ/\sqrt{K/\nu}\), where \(Z\sim\mathcal N(0,I_d)\) and
\(K\sim\chi^2_\nu\) are independent and a single scalar \(K\) is shared
across all coordinates of each example. The interpolant is
\(X_t=tY+(1-t)N\), and the network minimizes
\(\mathbb E\|f_\theta(X_t,t)-N\|_2^2\). Sampling integrates the associated
Euclidean field \((X_t-f_\theta(X_t,t))/t\). This objective is direct noise
regression; unweighted regression of the velocity label would define a
different objective. The source degree of freedom is selected independently
of the target degree of freedom.

The implementation follows the inspected equations and explicitly documents
its numerical conventions, including the positive initial time, Heun time
grid and treatment of the final endpoint. It is an independent implementation,
with a tunable isotropic source scale added for this comparison. It is not
presented as an unchanged authors' implementation or an exact reproduction of
undocumented endpoint conventions. The inspected source and implementation
limitations are recorded in [tflow_method.md](tflow_method.md).

## Planned matched-budget comparison: protocol paragraph

For every comparison, we will reuse the exact cached data and fixed splits,
the existing neural backbone and class-conditioning setup, and the complete
training budget listed in Appendix C. Source selection uses the nine settings
\(\nu\in\{3,5,7\}\) crossed with scale multipliers
\(\{0.5,1,2\}\). For each \(\nu\), the reference scale is

\[
s_{\mathrm{ref}}(\nu)
=\frac{\operatorname{median}_{Y\in\mathcal D_{\rm train}}\|Y\|}
{\sqrt{d\,F^{-1}_{d,\nu}(1/2)}},
\]

where \(F^{-1}_{d,\nu}\) is the quantile function of the F distribution.
This calibration uses training norms and does not rescale the data. Source
selection uses a separate seed, 46021, and validation data only. All nine
candidates train to 5% of the full update budget; the two best candidates
continue to 10% total budget. The selection score is the average of radial
KS and the mean KS over 64 fixed shared projections, with projection seed
61719. It is a tuning criterion, separate from the reported test metrics.

After selection, the chosen source parameters are frozen and three complete
training runs restart from scratch with the paper's model seeds. There is no
early stopping or test-set selection. Vector runs use 10,000 updates, audio
runs 24,000, and image runs 40,000. Reported evaluation budgets count actual
network calls: 512 for vectors, 160 for AudioMNIST and 100 for ImageNette.
The t-Flow sampler uses Heun integration at these budgets; the existing RAFM
results retain their RK4 protocol. The exact solver grid and positive start
time accompany each new result.

This protocol is proposed for 28 unique conditions: the complete Student-t
dimension and tail sweeps, six anisotropy settings, the Gaussian and 2D toy
controls, all four listed PIV dimensions, Finance, Weather, AudioMNIST and
ImageNette. Its estimated training cost is 15.4 full-run equivalents for
tuning plus 84 complete final runs. This is not a wall-clock estimate and
excludes evaluation and optional diagnostics. The unresolved cached-input,
batch and image-split provenance items in
[experiment_matrix.md](experiment_matrix.md) must be resolved before their
corresponding experiments are executable.

## Separate AudioMNIST control: methods paragraph

We will also evaluate fixed-radius spherical generation followed by an
independent empirical gain. For a nonzero generated STFT vector \(Y\), this
control returns \(X=G\,Y/\|Y\|\), where \(G\) is drawn independently of
the generated direction and digit label from the training-derived empirical
radius law. The existing fixed-spherical generator and final EMA checkpoints
are reused without retraining or changing their initial radius, learned
field, or integration path. The same generated examples are evaluated before
and after the gain transformation with the fixed content evaluator. The
transformation preserves direction algebraically; the implementation records
finite-precision differences and verifies the reproduced baseline against
the archived metrics.

This control addresses the independent-gain construction of AudioMNIST.
The RAFM-Ang versus RAFM-Vel comparison remains a separate contrast: both
methods use matched-radius training and sampling, while their regression
targets differ. The new control does not replace either method or establish
an outcome for that comparison in advance. Figure 2 and Table 5 should retain
both RAFM rows alongside the added control and, when available, t-Flow.

## Results insertion plan

- **Table 1:** add the t-Flow Sliced-Wasserstein column for Student-t d16/d32,
  PIV d64/d256, Finance and Weather. The pre-existing columns remain unchanged.
- **Table 2:** add the t-Flow ImageNette row with FID, KID, latent radial W1,
  precision and coverage.
- **Table 3:** add full t-Flow vector rows for Student-t d16/d32 and PIV
  d64/d256, including measured training time with hardware and precision.
- **Table 4:** add the complete t-Flow ImageNette row, including radial KS,
  recall and density.
- **Figure 2 and Table 5:** use the dedicated AudioMNIST renderer for the
  independent-gain control and optional measured t-Flow audio row. The full
  suite report retains the same audio seed records.
- **Appendix sweep and full-condition report:** retain every required
  condition and its expected seeds, including pending or failed conditions.
  Do not infer absent results from another dataset with the same dimension.

The standalone renderer is
`python -m experiments.tflow.render_results --manifest configs/tflow/suite_manifest.json --results-root OUTPUTS_TFLOW --output-dir NEW_REPORT_DIRECTORY`.
It consumes only new per-seed `result.json` envelopes and produces t-Flow-only
LaTeX additions, Markdown/CSV coverage, a full JSON report and per-condition
aggregate JSON files. It does not load or recompute historical baselines.
The output directory must be new or empty. A full result bundle is refused
unless every expected seed and condition is complete; `--pending` is an
explicit status-preview option, not a way to average surviving seeds.
Do not invoke result generation as an experiment step before approval.

## Results text held pending

No ranking, improvement, equivalence, significance, speedup or failure-rate
claim is supplied here. Once the artifacts are complete, results text should
state the measured metric and population standard deviation, all expected
seed outcomes, and the applicable protocol. Failed runs remain explicit and
are not silently removed from a three-seed mean. The newly added aggregates
are withheld when fewer than three valid final runs are available; this rule
does not alter the final historical paper rows or their documented treatment
of prior failures.

The historical `exploding_norm_rate` counts generated radii above 100 times
the generated median. Its `invalid_rate` combines this flag with the NaN rate.
A finite heavy-tailed sample can legitimately exceed that threshold. We will
retain both diagnostic values and include such runs in the complete three-seed
aggregate; rejecting them would selectively remove large finite observations
and bias the heavy-tail comparison. Nonzero `nan_rate` or `inf_rate`, nonfinite
measurements, and undefined required metrics still prevent completion. An
undefined angular bin is reported explicitly rather than averaged away.

Runtime comparisons must identify the actual accelerator, precision, update
budget and network-evaluation count. The existing RTX 2000 Ada and MI210
measurements describe different execution environments; a new cross-hardware
ratio is not reported as a matched timing result.
