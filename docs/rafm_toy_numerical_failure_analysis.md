# RAFM toy numerical failures: preserved v1 evidence

This audit identifies a concrete numerical defect in the inherited spherical-path
implementation that can produce extremely large finite angular targets. It does
not prove which individual training examples caused the four recorded sampling
failures. These are scientific execution failures, not interrupted jobs that can
be recovered by a scheduler retry. No source changes, model imports, training,
sampling reruns, or protocol changes were performed for this audit.

## Recorded failures

All four runs below completed the prescribed 10,000 updates, saved their final
checkpoints, and reached evaluation on an AMD Instinct MI210 on
`auh7-3b-gpu-015` with Torch `2.7.1+rocm6.3`.

| Arm | Seed | Final logged loss | Maximum logged loss | Training time (s) | Worker stderr |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 65457 | 482.3936 | 1.37354027204608e14 | 39.5695 | `rafm_inputs-final-v1-765748_3.err` |
| B | 65457 | 168674.4375 | 1.37354027204608e14 | 38.7722 | `rafm_inputs-final-v1-765748_4.err` |
| B | 8925 | 144503.40625 | 3.35599114387456e14 | 39.2212 | `rafm_inputs-final-v1-765748_4.err` |
| C | 8925 | 131950.75 | 3.35599047278592e14 | 39.2740 | `rafm_inputs-final-v1-765748_5.err` |

Each `result.json` records `FloatingPointError: Nonfinite samples in batch
starting at 0` from `experiments/rafm_inputs/run.py:252`. The initial source
passed its finite check. The original ambient RK4 sampler completed the expected
512 network calls, after which its single 10,000-sample batch contained nonfinite
coordinates. The failure occurred before quality metrics and before writing a
generated-sample artifact. The number of invalid rows and the first divergent
integration step were not recorded; neither should be inferred from this error.

Checkpoints, configuration/data identities, training statistics, training logs,
and failure tracebacks remain under
`outputs_rafm_input_study/v1/final/toy_radial_angular/<arm>/seed_<seed>/`.
Worker stderr remains under `outputs_rafm_input_study/v1/logs/`.
All saved training loss rows in these four runs are finite. Finiteness alone
does not establish numerical correctness: the recorded spikes are enormous.
Successful toy runs also contain similarly large spikes, so the defect is not
limited to the four runs that eventually produced invalid samples.

## Inherited path behavior

The following references identify the audited source bytes, with line numbers
at the recorded hashes below.

1. `experiments/rafm_inputs/run.py:79` calls `sample_path`; line 80 separately
   calls `conditional_vector_field`. This preserves the original call sequence
   in `rafm/flow_matching/loss.py`. `SphericalGeodesicPath` delegates these calls
   to `slerp` and `slerp_velocity`.
2. `rafm/utils/sphere.py:41–45` and `85–89` each independently call
   `_handle_antipodal` when the angle is within `1e-3` of pi. The helper draws
   a new random perpendicular vector at line 138. Thus the state and velocity
   can use different perturbed starting directions. Near the antipodal branch,
   the velocity is not guaranteed to be the derivative of the sampled state.
3. The `1e-3` random perturbation at `sphere.py:143` can partially cancel an
   existing perpendicular displacement and move a near-antipodal pair closer
   to antipodal. The code does not enforce a minimum resulting separation.
4. In float32 the recomputed cosine can round to `-1`. The resulting stored
   angle is approximately `3.141592741`, whose sine is approximately
   `-8.74228e-8`. At `sphere.py:91`, this is replaced by the positive floor
   `1e-12`. The velocity formula at line 100 then multiplies its residual
   vector by approximately `pi / 1e-12`.
5. `slerp` renormalizes its interpolated state at `sphere.py:64`; the velocity
   routine does not apply the derivative of that renormalization. Dividing the
   resulting velocity by the state radius at
   `baselines/rafm_input_parameterization.py:157` removes the radius scale,
   but does not repair the inconsistent or excessively large angular target.

For an exact shortest geodesic on a nonzero fixed-radius sphere, parameterized
over unit time, angular speed is its central angle and is at most pi. This
mathematical statement does not imply the audited floating-point routine
satisfies the bound. The issues above concern its numerical implementation;
they do not establish a new theoretical limitation of the population method.

## Scalar illustration and limits

A lightweight calculation used Python's standard library (`math` and `struct`),
rounding elementary arithmetic to float32. It did not import Torch or execute
any model. Take radius one, `u1=(1,0)`,
`u0=(-cos(delta), sin(delta))`, `delta=0.0008`, and time `t=0.25`.
The negative perpendicular perturbation permitted by the helper gives a
normalized float32 direction approximately `(-1,-0.00020000018)` and a cosine
of exactly `-1`. Applying the audited velocity formula yields angular speed
approximately `4.44288672e8`. Against a small prediction, that single example
contributes approximately `4.81915098e13` to the batch-4096 sum-coordinate MSE.

This is a concrete counterexample to bounded numerical targets and shows a
mechanism on the scale of the observed loss spikes. It is not a GPU replay:
device transcendental arithmetic can differ, and the original offending
minibatch states, sampled perturbations, and per-example targets were not saved.
The failure logs alone cannot attribute every divergent checkpoint or sample
to this mechanism. No direct replay or alternative-protocol result is claimed.

The original near-antipodal tests in `tests/test_paths.py:93–130` check selected
finiteness and norm-preservation properties. They do not enforce the angular
speed bound or state/velocity consistency in that branch. The new runtime
regression check verifies agreement with the original loss and RNG behavior;
matching an inherited implementation also retains its numerical defects.

## Provenance and disposition

Read-only `git show` comparisons confirmed that the first four source files in
the table below are byte-identical to commit
`5b89ed5f4af8a47c3b57eb9d595203daafc6d2c4`. No B/C input-interface defect was
identified in this audit. This source identity does not prove that every
historical reported experiment used these exact bytes or encountered the same
numerical events.

| Source | SHA-256 |
| --- | --- |
| `rafm/utils/sphere.py` | `83d29211579a819576f5a80564328c83c0868d1d351803ae9335720a23c59dfb` |
| `rafm/paths/spherical_geodesic.py` | `aa1211671c19332718f97973e451d59b1c6bb00169f21a40ab22718a2a511e77` |
| `rafm/flow_matching/loss.py` | `dcc1cfdf9cffee5c8d64f4054a38d5d2ece4ed1f0b543c3db9fcb067e14a19df` |
| `rafm/flow_matching/sampler.py` | `6b3e79a81e2ce982e5467b8ca563decc457f2cd16f9e919c626f710ece224de0` |
| `baselines/rafm_input_parameterization.py` | `80242e47fde4210bace698b60ecb85ffe26a5f2baa2c93b00652286d65e823c8` |
| `experiments/rafm_inputs/run.py` | `58fdbc9ae54dab4c43278bb58c0b7ed2fbef1fbd94c7f56f0be3332e4d359818` |

| Preserved `result.json` relative to the toy output root | SHA-256 |
| --- | --- |
| `A/seed_65457/result.json` | `5cfe2fa04a552eb3d321c58e57ec7919359975d65088156fa1c3dbaa3408284f` |
| `B/seed_65457/result.json` | `bbdbd06a1246bacb37faeb7c01064068ecfb91a7e63d8b9a30ddf9df45993b00` |
| `B/seed_8925/result.json` | `272344ab09ebe6e606636eb57fd0820507dd3160716c9ae992cb4b6f55623d9a` |
| `C/seed_8925/result.json` | `01910570b8387511c139fdf08ff32519320c477adb6c35c723e3375e2bc0b6bc` |

The current four failures must remain failures in v1 reporting, with no
partial-seed aggregate presented as a complete three-seed comparison. They
must not be relabeled as scheduler-recoverable: all prescribed training updates
completed, and an unchanged checkpoint evaluation cannot repair training-target
errors. No toy fix or retraining is authorized under the current requirement to
preserve the implementation and protocol.

If a correction is separately authorized, it would require a documented
numerical-path version, tests of the angular bound and state/target consistency,
and fresh matched A/B/C runs on the same verified caches, seeds and per-method
budgets. It must not be introduced as a silent checkpoint resume, sampler
workaround, or replacement for the preserved v1 outcomes. Existing reported
paper results remain unchanged by this audit.
