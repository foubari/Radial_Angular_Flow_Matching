# t-Flow method and implementation provenance

`baselines/tflow_core.py` is an independent implementation of the noise-prediction
formulation in [Heavy-Tailed Diffusion Models, arXiv v2, Appendix B,
Eqs. 164–166](https://arxiv.org/html/2410.14171v2). It is not copied authors' code.
This document distinguishes the formulation from benchmark numerical choices.

## Formulation

For data `y`, draw independent standard Gaussian `z` and one scalar
`kappa ~ ChiSquare(nu) / nu` per example. All coordinates share that scalar.
With `n = z / sqrt(kappa)`, the paper uses

\[
x_t=t y+(1-t)n,\qquad t\sim U(0,1),\qquad
L=\mathbb E\|f_\theta(x_t,t)-n\|_2^2,\qquad
\dot x=(x-f_\theta(x,t))/t.
\]

The flow runs from noise at zero to data at one. Table 5 specifies no input/output
preconditioning, unit loss weighting, Heun integration, rho 7, noise levels
1 and 0.01, and reports NFE 25. Its source-df sweeps use 3, 5, 7 or 5, 7, 9.

The inspected v2 source has inconsistent algorithm notation: integer training
times versus continuous uniform times; network conditioning on sigma versus t;
an initial zero-time division; an overwritten first Heun slope; and an outer
grid exponent 1/rho that does not attain the stated endpoints. These do not
uniquely define a runnable sampler. The equations above determine this core.

## Final publication and code search status

On 2026-09-10, targeted search returned indexed text from the official
[final ICLR 2025 publication, pp. 30–31](https://proceedings.iclr.cc/paper_files/paper/2025/file/1d5b9233ad716a43be5c0d3023cb82d0-Paper-Conference.pdf).
Appendix B renumbers the straight interpolant, field and noise objective to
Eqs. **167–169**; their mathematical content agrees with the implementation.
The final-paper Figure 5 indexed algorithms retain the integer-time versus
continuous-time inconsistency and overwrite the first Heun slope while referring
to a separate primed slope. This supports using the equations and an explicit
standard Heun update. It does **not** establish the authors' executed sampler.

This is verification against indexed primary-source text, **not visual inspection
of the final PDF**. The web reader still rejects its 33,581,191-byte size, and the
cluster proxy still rejects a direct download with `Tunnel connection failed:
403 Forbidden`. Final-page images and the final table/schedule layout remain
unverified. The benchmark endpoint and grid remain declared adaptations.

Searches of the paper, arXiv metadata, author page and public GitHub references
did not establish a public authors' t-Flow implementation. The
[first author's page](https://kpandey008.github.io/) links the paper without code.
The [NVIDIA publication page](https://research.nvidia.com/labs/lpr/publication/pandey2024heavytailed/)
was rechecked on 2026-09-10 and likewise links arXiv without a code artifact.
[NVIDIA PhysicsNeMo v1.2.0 release notes](https://github.com/NVIDIA/physicsnemo/releases/tag/v1.2.0)
describe a later *adapted t-EDM* implementation. Its
[repository](https://github.com/NVIDIA/physicsnemo) is Apache-2.0 licensed.
The listing for
`physicsnemo/experimental/models/diffusion/preconditioning.py` was verified, but
its contents could not be fetched. Neither this path nor that release is treated
as verified t-Flow source. No t-Flow code was copied or t-Flow repository cloned. The separately pinned official SiT backbone is described below.

## Benchmark adaptations and restrictions

- **Source scale:** `TFlowSourceConfig(nu, scale)` allows `n = scale*z/sqrt(kappa)`.
  The paper's source has scale 1. Scale means the square root of the scale-matrix
  coefficient, not standard deviation. Use the same scale for training noise
  labels and the sampling prior. No variance matching, per-coordinate chi-square
  draws, truncation, rejection of extreme finite noise, or target-df matching is
  performed.
- **Source degrees of freedom:** finite `nu > 2` and finite `scale > 0` are
  enforced. Then `E||n||² = d*scale²*nu/(nu-2)`. The noise label therefore has a
  finite second moment even when the data do not. This is a sufficient L2-label
  condition, not a claim that every neural predictor has finite population loss.
  Student-t distributions exist for smaller positive nu, but their labels lack
  this guarantee. Source nu need not equal target nu. Finite-sample tail agreement
  does not establish equality of population moments.
- **Model interface:** `model(x, t)` receives a batch and times of shape `[B]`,
  and returns scaled noise predictions of the same shape, device and dtype.
  The caller owns preprocessing, architecture, model train/eval mode and RNG
  state. The core neither normalizes features nor changes geometry.
- **Loss reduction:** the default `reduction="batch_mean"` sums squared residuals
  over all non-batch coordinates and averages over examples. `"none"` returns
  per-example squared norms; `"sum"` sums them. Explicit `"mean"` averages over
  coordinates as well. That optional factor 1/d affects gradient scale and can
  interact with optimizer epsilon; it is not the default published reduction.
- **Precision and failures:** batches use float32 or float64. Source mixture
  arithmetic uses float64 and is checked after conversion. Inputs, source
  draws, predictions, squared residuals, loss, vector fields, predictors and
  corrected states must be finite. Nonpositive mixture denominators and time
  denominators raise errors. Nothing is silently clipped or repaired. These
  checks synchronize accelerators and are part of runtime unless the caller
  separately reports their overhead.
- **Randomness:** sampling uses public PyTorch distributions and the caller's
  global CPU/device RNGs. The runner must seed them and checkpoint their states.
  Training draws uniform times independently of data and noise. Explicit
  `times` and `noise` arguments are available for deterministic checks.

## Sampling convention

`heun_sample(model, x_initial, t_min=..., n_steps=..., ...)` requires an explicit
`0 < t_min < 1`. Supplying a fresh Student-t prior draw as `x_initial` treats it
as the state at `t_min`; this approximates the omitted interval `[0,t_min]`.
It is not exact initialization from the unknown intermediate marginal. The
noise-prediction error is amplified by `1/t`, so choosing a very small start
time does not alone guarantee better samples. No analytic initial drift based
on a potentially nonexistent target mean is assumed.

The default `grid="power_sigma"`, `rho=7`, `sigma_min=0.01` uses `N=n_steps`
knots (for `N>=2`):

\[
\sigma_i=\left((1-t_{\min})^{1/\rho}
 +\frac{i}{N-1}[\sigma_{\min}^{1/\rho}-(1-t_{\min})^{1/\rho}]\right)^\rho,
\quad t_i=1-\sigma_i,\quad i=0,\ldots,N-1.
\]

The final time `t_N=1` is appended explicitly. Thus the final interval runs from
0.99 to 1 with the default sigma minimum. This endpoint choice is an adaptation;
it is not asserted to reproduce an undocumented authors' endpoint convention.
For `N=1`, the only interval is directly from `t_min` to 1. The grid is calculated
in float64 and must remain positive and strictly increasing at model precision.
Using a power grid all the way to sigma zero can collapse late float32 times;
the explicit minimum and final interval avoid that usual case.

Alternative `linear` and `power_time` grids provide exactly `N` intervals with
`u=i/N`, respectively

\[
t(u)=t_{\min}+(1-t_{\min})u,
\qquad
t(u)=[t_{\min}^{1/\rho}+(1-t_{\min}^{1/\rho})u]^\rho.
\]

All grids preserve `t_min` and exactly 1. Power exponent placement, time
orientation, positive start, and final endpoint are explicit, reviewable choices.
No grid has been selected through test-set inspection or benchmark results.

Each interval computes separate first and second slopes:

```text
k1 = (x - model(x, t)) / t
x_predict = x + dt*k1
k2 = (x_predict - model(x_predict, t_next)) / t_next
x_next = x + dt*(k1 + k2)/2
```

Both calls occur on the final interval too. Returned `nfe` counts actual model
calls and equals `2*n_steps`. The reported paper value “NFE 25” is not silently
interpreted as 25 Heun intervals. Samples retain device and dtype; the return
record also includes the exact grid and its parameters. There is no terminal
denoising, source refresh, radius preservation or tangent projection.

## Why velocity MSE is different

If an interface defines `f(x,t)=x-t*v(x,t)`, then the interpolant identity gives

\[
\|f(x_t,t)-n\|^2=t^2\|v(x_t,t)-(y-n)\|^2.
\]

Ordinary velocity MSE drops this weight and changes the objective. The core
computes the direct noise residual instead, avoiding division by small times
and the potentially infinite-variance data velocity label during training.

## Validation boundary

`tests/test_tflow_core.py` contains deterministic shared-denominator checks,
loss/gradient identities, analytic linear and constant ODE checks, endpoint and
NFE assertions, RNG replay, and invalid-value handling. There are no training
loops, optimizer experiments, tuning sweeps or benchmark evaluations in this
module or its tests. Static parsing and numerical unit-test outcomes must be
reported separately; creating these tests does not imply they were executed.

## Downstream integration

`baselines/tflow_downstream.py` constructs existing raw backbones through
`build_backbone(model_cfg, dim)` and adapts their inputs through
`flat_noise_callback(model, model_cfg, labels)`. The callback is a plain function,
so model checkpoints retain original keys. It casts mixed-precision output back
to the input float32/float64 dtype before noise-residual computation.

Image configuration uses `kind="image_sit"`, absolute `source_dir`, hidden 384,
depth 12, heads 6, class dropout 0.1 and 10 classes unless explicitly configured.
The [official SiT repository](https://github.com/willisma/SiT) was cloned into
`/mnt/vast01/users/fouad.oubari/references/SiT-tflow` and checked out at the
workspace-pinned commit `cbde832a40b153ccc79603412409da9c9b0c568c`. The adapter checks
this commit and rejects changes to `models.py`. The upstream license is MIT;
source and license hashes are attached as plain metadata, outside the state dict.
Audio configuration uses `kind="audio_unet"` and the existing `UNetVel`; default
channel width is 96. Audio class dropout remains the caller's responsibility,
while SiT handles its own dropout during training.

Image evaluation accepts already generated flat CPU samples and held-out centered
test latents through `evaluate_image`. It requires the training-only mean and
explicit local decoder/Inception paths, an existing real PNG reference, its
expected manifest hash, and a new generated-image directory. Generated image
datasets belong under the workspace data root. It never downloads models,
regenerates the real reference, samples a flow, or changes the old evaluator.

The decoder and metric sequence follows
`experiments/image_latents/dit/dit_eval_sit.py`: undo centering and scale 0.41407,
decode DC-AE, clip decoded pixels to the existing [-1,1] range, save PNGs, compute
torch-fidelity FID/KID and PRDC k=5 from the same Inception 2048-dimensional
features. Latent metrics reuse the existing radial and 200-projection sliced-W1
implementations with the original seed 0. The report preserves unrounded metric
values and hashes the real PNG manifest, Inception weights and evaluator source.
An explicit final output directory receives only JSON reports. No backbone
forward pass, decoder execution or downstream evaluation was performed while
writing this adapter; these checks must be recorded separately when executed.
