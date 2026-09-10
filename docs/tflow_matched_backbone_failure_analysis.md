# t-Flow matched-backbone failure analysis

Audit date: 2026-09-10. This is an analysis of preserved `outputs_tflow_full/v2`
artifacts, not an altered method, a new tuning experiment, or a retrospective
selection rule. No checkpoint evaluation, training, or GPU computation was
performed for this audit. The only numerical calculation was a scalar,
standard-library verification of the algebra below.

## Conclusion and scope

The inspected implementation agrees with the published straight interpolant,
Student-t noise-prediction objective, and generative field. No sign error,
train/sample time mismatch, omitted published external identity skip, or Heun
arithmetic error was found. There is, however, a provable incompatibility between
the directly predicted noise and the benchmark's plain width-128 MLP when the
output dimension exceeds 128: a subspace of the initial noise cannot be learned
away, and the implemented field amplifies it by 1/t_min = 1,000.

This establishes a limitation of this matched-backbone adaptation. It does not
establish an intrinsic failure of published t-Flow, and it does not prove the
cause of observed failures in dimensions at most 128 or in U-Net/SiT models.
The existing runs, configurations, tuning selections, budgets, and failures
remain unchanged. Results must retain this qualification rather than supporting
a general claim that RAFM outperforms the published method.

## Published equations and the inspected implementation

[Appendix B, Eqs. 164–166 and Table 5](https://arxiv.org/html/2410.14171v2#A2)
specify a shared chi-square Student-t scale per example, straight interpolation,
noise-prediction squared error, and the field `(x - predicted_noise) / t`.
Table 5 gives `c_skip=0`, `c_out=1`, `c_in=1`, and unit loss weighting for t-Flow.
The published backbone is DDPM++, not this benchmark's vector MLP.

The corresponding code is:

- `baselines/tflow_core.py:121–131`: shared scalar mixture and source scale.
- `baselines/tflow_core.py:159–176`: uniform time, interpolation, noise target,
  and summed-coordinate/mean-batch squared error.
- `baselines/tflow_core.py:185–196`: the positive-time generative field.
- `baselines/tflow_core.py:294–303`: separate Heun predictor/corrector slopes.
- `experiments/tflow/run.py:259–280`: raw matched backbone and noise callback.
- `experiments/tflow/run.py:410–413, 489–504`: the same source configuration and
  callback convention in training and generation.

Source scale is already a declared benchmark adaptation and multiplies both
the noise labels and generation prior. It is not accidentally dropped from
either side. The actual network-call count is two per Heun interval.

The paper's Eq. 166 conditions on `t`, whereas its Table 5 and algorithms use
`sigma=1-t`. Our code consistently conditions on `t` in both phases. This is an
invertible time-label adaptation, not evidence of identical optimization to the
authors' implementation. The published algorithm/grid notation also leaves
endpoint and corrector ambiguities, documented in `docs/tflow_method.md`.
Our positive start, sigma-power grid, and final interval are disclosed choices.
The first-author/NVIDIA pages have not yielded a verified official t-Flow
implementation. A later t-EDM implementation is not treated as t-Flow evidence.
The final ICLR paper's indexed equations agree, but its PDF table layout and
the authors' actually executed sampler remain unverified; see the source/access
record in `docs/tflow_method.md:29–57`.

## Exact affine-output-subspace obstruction for d > 128

The plain MLP used here ends in `Linear(128, d)` with a bias and no output skip
(`rafm/models/mlp.py:53–58`). Regardless of preceding nonlinearities, a fixed
trained model therefore has

    epsilon_theta(x,t) = W h_theta(x,t) + b,    W in R^(d x 128).

Let `q` be a unit vector with `W^T q = 0`. Such vectors span a space of
dimension at least `d-128` when `d>128`. For every input and time,
`q^T epsilon_theta(x,t) = q^T b`. Consequently, with
`z(t) = q^T(x(t)-b)`, the implemented ODE is exactly

    dz/dt = z/t,    z(t) = (t/t_min) z(t_min).

Thus its final projected component is

    q^T(x(1)-b) = q^T(x(t_min)-b) / t_min.

This obstruction holds for any trained weights, including a global optimum
within this model class. It is not merely an insufficient-training hypothesis.
For a fixed model and an isotropic Student-t source of scale `s`, degrees of
freedom `nu>2`, and a nullspace projector `P` of rank `m`,

    Cov[P x(1)] = [s^2 nu/(nu-2) / t_min^2] P,
    E ||x(1)||^2 >= m s^2 nu/(nu-2) / t_min^2.

The bias cannot cancel this source variance. The formula uses the fresh-source
distribution and exact arithmetic; it is not a measured checkpoint projection.
For PIV d256, `m>=128`, `s=0.08311994486347801`, `nu=3`, and `t_min=.001`, the
RMS lower bound is **1628.8116189103416**. This is an RMS quantity, not radial W1
or a mean radius; it should not be equated to the observed radial W1 of 1462.05.

## Heun telescopes exactly for these components

For one interval from `t` to `t+h`, the scalar first slope is `z/t`. The Euler
predictor is `z_predict=z*(t+h)/t`, so the second slope is
`z_predict/(t+h)=z/t`. Heun therefore gives exactly

    z_next = z * (t+h)/t.

Multiplying across any positive increasing time grid telescopes to
`z_final/z_initial = 1/t_min`. The large first interval `.001 ->
.014142839201640722` is not necessary for this particular failure, and refining
the grid alone does not remove it. Float32 arithmetic can add numerical error,
but does not remove the representational obstruction.

A standard-library scalar calculation using the existing 256-interval
sigma-power grid, `b=.27` and `x_initial=.12`, returned
`x_final=-149.73000000000008`; the exact result is
`-149.73000000000002` (absolute difference `5.68e-14`). The measured scalar
amplification was `1000.0000000000005`. This checks the algebra, not the
trained checkpoint or its quality.

## Validation evidence preceding the final evaluation

`outputs_tflow_full/v2/tuning/piv_d256/selection.json` records nine 500-update
stage-one candidates and two 1,000-update finalists. **Every candidate and both
finalists had validation radial KS = 1.0.** Stage-one projected-KS means ranged
from `0.530140625` to `0.5610312500000001`. The frozen winner has `nu=3`,
`scale=0.08311994486347801`, radial KS `1.0`, projected KS `0.521203125`, and
combined selection score `0.7606015625`. These use 1,000 generated samples and
199 validation observations. The receipt states `test_data_used_for_selection:
false`. Selection identifies the minimum score among the candidates; it does
not certify that any candidate learned a usable distribution.

The already recorded PIV d256 seed-8925 final result has radial KS `1.0`, radial
W1 `1462.048828125`, sliced W1 `72.58660888671875`, finite generated samples,
and all four angular-bin scores plus their mean undefined. Its status is
`failed`, not a successful evaluation with omitted angular metrics. It retains
the 10,000-update checkpoint and measured training time `34.57014872133732 s`.
These final-test results motivated the audit; they were not used to choose a
new prior, sampler, architecture, or training budget.

All nine stage-one candidates and both finalists for PIV d64 and
`student_t_d128_df3.0_cor` also had validation radial KS `1.0`. The output-rank
proof does **not** explain those cases: a `d x 128` output layer can have full
row rank for `d<=128`. Direct-noise approximation error near zero is amplified
by `1/t`, and the omitted initial interval is approximate, but neither is a
measured causal attribution for those checkpoints. Additional diagnostics
would require separate work; none were run for this document.

## Validation coverage and interpretation boundary

The existing unit checks test the noise law, target and gradient identities,
Heun arithmetic on analytic fields, endpoints, finite-value handling, and
network-call counts. They do not show that the matched architecture can learn
the near-zero noise-identity boundary. Finite samples and a finite noise loss
alone are therefore insufficient evidence of a successful reproduction on a
new backbone. The inspected first PIV d256 training loss is also not negligible
relative to source energy: its source has `E||noise||^2=5.3060545797946554`.

Adding an identity or time-dependent output parameterization could change this
function class without adding trainable parameters, but would be a distinct
adaptation requiring explicit validation and reporting. No such change, solver
replacement, source retuning, or rerun is made here. In particular, uniformly
weighted velocity regression would change the published objective and must not
be silently substituted. Preserve every existing success/failure and describe
this study as an independent equation-based, matched-backbone adaptation.

## Audit identity

Observed repository HEAD: `cd2a857f2b8ddb5e083b71d49b7fbb786d550100`.
The source hashes below identify the actual inspected bytes independently of
that checkout. This HEAD is not newly asserted to be a training commit.

| Artifact | SHA-256 |
| --- | --- |
| `baselines/tflow_core.py` | `9c1d323fe3bb6ca357459f63e6fea3542e28645616d7981c314f5226f04b5a81` |
| `experiments/tflow/run.py` | `67b512f5607525814d08f60195846741089714e134bb65303904bbdea9849f0c` |
| `rafm/models/mlp.py` | `abe7b4e44e034a897e4772df0929053a24edaa92047eb48846a60667deb5a483` |
| `docs/tflow_method.md` | `fd49221480f2de61b0ee245073fb39ec6cfb227cc9355091c09bd57c622f8c22` |
| `tests/test_tflow_core.py` | `e26b045f91bd2fac920b2be52776a4311bcc3cc6a1f0c41d907602be8fee6782` |
| `outputs_tflow_full/v2/tuning/piv_d256/selection.json` | `8ae6c25fe6a5b8eca34695a57357bebf4148690ff59cc2f345fcd570f7b64085` |
| `outputs_tflow_full/v2/tuning/piv_d64/selection.json` | `a7d4f6b0004029cb861fcf2d13e04300c6e2ed0d76ff8358baf40282bede3d29` |
| `outputs_tflow_full/v2/final/piv_d256/seed_8925/result.json` | `5f31d47c1ffecea199e94cc7db13494d2f063cb9cf0d485ffd048f332b94c14c` |

The final-result receipt records checkpoint SHA-256
`255348cf2fad1641012fa8b9271f07d2d6f163c0376df36c7fe45ef7e7df63f3`
and generated-sample SHA-256
`9429fbb530f6e2e875e01fc3ede697e427e0151bea9f18c5ffd6adbe4860477c`.
Those two large artifacts were not loaded or rehashed in this bounded audit.
