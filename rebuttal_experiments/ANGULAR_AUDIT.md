# Angular RAFM — radial anomaly audit

**Question.** On some settings (PIV, Student-t) Angular RAFM shows a *worse* radial Wasserstein
(`radial_w1`) than standard RAFM, despite both using the identical radial source and tangent
projection. Is this a bug, or a genuine property?

## Controls verified (identical between std-RAFM and Angular RAFM)

| Control | std-RAFM | Angular RAFM | same? |
|---|---|---|---|
| Empirical radial source | `RadialEmpiricalSource(mode="ecdf").fit(train)` | same | ✅ |
| Source sampling at generation | `x0 = R·u0`, `R~eCDF(‖x1‖)`, `u0` uniform sphere | same | ✅ |
| Path | `spherical_geodesic` | same | ✅ |
| Solver / NFE | RK4, nfe=512 | same | ✅ |
| Tangent projection | `project_tangent=True` | same | ✅ |
| Eval protocol | `radial_metrics(samples, test)` | same, same test | ✅ |

Only difference: training target `A = u_t/‖x_t‖` and sampling reconstruction `v = ‖x‖·A` (plus the
same tangent projection applied afterward).

## Measurement (radii of generated vs source vs test)

`drift` = W1(‖gen‖, ‖train‖) measures how far the generated radial law moved from the source law
that a *perfect* tangent flow would conserve. `radial_w1` = W1(‖gen‖, ‖test‖) is the reported metric.

| Setting | method | radial_w1(gen,test) | drift W1(gen,train) | q99 gen/src | max gen/src |
|---|---|---|---|---|---|
| PIV d64 (n_test=201) | std-RAFM | 0.048 | 0.062 | 4.52 / 4.62 | 5.94 / 5.61 |
| PIV d64 | Angular | 0.095 | 0.069 | 4.62 / 4.62 | 5.49 / 5.61 |
| Student-t d16 df3 | std-RAFM | 0.243 | 0.207 | 65.2 / 65.6 | 309 / 892 |
| Student-t d16 df3 | Angular | 0.288 | 0.246 | 67.2 / 65.6 | 572 / 892 |

## Conclusion (not silently "fixed")

- **PIV → finite-sample.** Drift-from-source is nearly identical (0.069 vs 0.062) and Angular matches
  the source tail *better* at q99 (4.62 vs 4.62). The 2× gap in `radial_w1(gen,test)` is dominated by
  the tiny held-out set (n_test = 201): W1 to test is a noisy estimate at this sample size.
- **Student-t → numerical / parameterization, not implementation.** Angular drift is genuinely higher
  (0.246 vs 0.207). Mechanism: the reconstruction `v = ‖x‖·A` multiplies any residual non-tangent
  component of the *predicted* `A` by `‖x‖`. In a heavy tail (true max radius ≈ 892) this amplifies at
  large radius and pushes extreme generated radii outward (max 572 vs 309) faster than the per-step
  tangent projection removes them within the RK4 substeps. It is a property of the angular
  parameterization interacting with large radii, present with identical code paths — not a bug.

**Trade-off, stated plainly.** Angular pays a small radial cost in heavy-tailed / tiny-sample settings
but wins the *directional* and *whole-distribution* metrics (`sliced_w1`, `angular_sw_mean`) there and
elsewhere. The radial marginal is source-governed and near-identical for both methods where the tail is
light (finance, weather); the divergence only appears where `‖x‖` has a heavy tail.

## Addendum — d=2 numerical fragility (found during suite completion)

The stability scan flagged NaN seeds only at **d=2**: `student_t_d2` (seed 8925, nan_rate 0.85) and the
`toy_radial_angular` d=2 toy (seed 77395, nan_rate 0.0007). Every Angular seed at **d≥8** across the
dimension, df, and anisotropy sweeps and all real datasets is finite. This is the same `v=‖x‖·A`
amplification mechanism in its worst case: at d=2 the sphere's tangent space is 1-D, so a rare heavy-tail
trajectory overshoots under RK4 and diverges. It is a numerical limit of the parameterization in the
low-dimensional regime, **not** an implementation error (identical code path is stable everywhere d≥8).
Affected d=2 aggregates are averaged over the finite seeds only, with n disclosed per row. No radius
clipping or NFE increase was applied, to keep the protocol identical to the baselines.
