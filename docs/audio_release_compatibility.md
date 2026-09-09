The recovered fixed-spherical AudioMNIST release is compatible with the audited evaluation source at the level of source bytes and metadata interpretation. The accompanying [machine-readable audit](../outputs_audio_gain/release_verification/source_compatibility.json) records the comparisons, original metadata hashes, and exact source excerpts. This audit imports no project modules and loads no tensors or checkpoints; strict state-dictionary compatibility and numerical checks remain compute-node responsibilities.

The release is `foubari/msgm-sparse-control`, tag `fixed-spherical-audiomnist-v1`. Its manifest identifies `github.com/foubari/radial_angular_FM`, branch `iclr-image-experiments`, commit `d3006dc8ee61b2ab6e470115d3f21c778fefcf81` as the checkout holding the recovered runs. The manifest explicitly says that metadata did not record a training commit. Accordingly, the audit records that revision under `artifact_checkout`, sets `training_commit` to `null`, and sets `training_commit_verified` to `false`. Source equality cannot establish an unrecorded training commit.

The following SHA256 values apply independently to both `git show d3006dc8ee61b2ab6e470115d3f21c778fefcf81:PATH` and the current worktree file. All eight direct byte comparisons passed.

| File | SHA256 of artifact source and worktree |
| --- | --- |
| `experiments/poc_audio/audio_flow.py` | `002401d0933edc1e090279428fbbfcf3d764910b1bf15a9840ba48d36e731dc0` |
| `experiments/poc_audio/audio_eval.py` | `ca4b1f0c788fcd3eef69b4970daf4a2cad540ee3fdcbe987e407f31a2f401004` |
| `experiments/poc_audio/audio_classifier.py` | `7af21994c8e7f428252ed566dbfae8274bf0b3368b96509dd76f7f65136f923a` |
| `experiments/poc_audio/audio_data.py` | `db1c5487a7ff6741265d3a3b74462dbe26ee2d8b906aaf9c6314b356bc5daf65` |
| `rafm/sources/radial_empirical.py` | `efcd2ff006d18ddf39b54bf33026b040b0c1badf9ec5b26edaefec7fb6d4d84e` |
| `rafm/utils/sphere.py` | `83d29211579a819576f5a80564328c83c0868d1d351803ae9335720a23c59dfb` |
| `rafm/flow_matching/sampler.py` | `6b3e79a81e2ce982e5467b8ca563decc457f2cd16f9e919c626f710ece224de0` |
| `rafm/paths/spherical_geodesic.py` | `aa1211671c19332718f97973e451d59b1c6bb00169f21a40ab22718a2a511e77` |

The released metadata omit `args.ncls` for seeds 8925, 1234, and 7. Seed 8925 also omits `args.angular`; the other two explicitly record `false`. Original `audio_eval.py:42–43` evaluates missing values using `meta.get("angular", False)` and `meta.get("ncls", 10)`. These are source-backed legacy evaluation defaults. `audio_flow.py:101–102` also gives `make_model` a default class count of 10, while trainer arguments at lines 122–124 define `--angular` as `store_true` and `--ncls` with default 10. Resolve only these absent fields, record each resolution separately, preserve the original metadata bytes, and reject conflicting explicit values. These defaults establish evaluation behavior without recovering the missing training revision.

All three metadata files match their manifest SHA256 and embedded configuration. They record `R0=2.2613651752471924`, `D=16254`, 27,896,802 parameters, UNet channels 96, depth 5, 24,000 training steps, batch size 32, learning rate 0.0002, EMA 0.999, and class dropout 0.1. Their hashes are recorded individually in the JSON audit. Full release verification must cover every downloaded asset, including manifest and metadata/config files, against authenticated release digests; source compatibility is an additional check.

`audio_flow.py:20–22` specifies the generator split exactly:

```python
g = torch.Generator().manual_seed(0)
p = torch.randperm(n, generator=g)
ntr = int(n * 0.85)
tr, te = p[:ntr], p[ntr:]
```

This is a local CPU Torch generator and a permutation of the nominal training file. For the configured 12,000 training examples, generator fitting uses 10,200 permuted indices. The remaining 1,800 are the internal held-out portion named `te` by `build`; they are distinct from the external 3,000-example `audiomnist_stft_test.pt`. The original data preparation uses a separate NumPy permutation to create the 12,000/3,000 files (`audio_data.py:41–43`). The generator split must be applied to the recovered nominal training tensor. This static audit did not recompute tensor indices or verify dataset counts numerically.

`build` computes `R0=float(x[tr].norm(dim=1).mean())` from raw generator training samples (`audio_flow.py:104–112`). For fixed-spherical training, it fits the source to `(R0*x/x.norm(...).clamp(min=1e-8))[tr]`; retain this source and its original sampled initial radii when generating Y. For RAFM, it fits the empirical radial source to raw `x[tr]`. The new gain control draws G from that same training-derived RAFM source after fitting, with no internal held-out or external test gains entering the fit. External test gains define evaluation references and tail thresholds only.

`RadialEmpiricalSource.fit` stores CPU norms of its fitted training tensors. In ECDF mode, `_sample_radii` draws `u=torch.rand(n)` on the CPU and applies `torch.quantile(training_norms, u)` with default linear interpolation, followed by a nonnegative clamp (`radial_empirical.py:50–61,85–93`). Its `sample` method draws these radii first and sphere directions second (lines 72–78). Consequently, an isolated CPU seed-0 `sample_radii` call reproduces the raw quantile draws used at the start of original RAFM evaluation. Replaying `sample(...).norm(...)` may introduce floating-point differences from sphere normalization; retain that diagnostic while using raw ECDF G for the intervention.

The gain draw uses a forked and restored CPU RNG. Original Y generation has a separate restored CPU/CUDA seed-0 scope. Original `audio_eval.py:46–50` first samples source radii on CPU, then generates normalized sphere directions on the evaluation device. For the authorized GPU evaluation, the separate CUDA direction stream is preserved. Neither generated directions, labels, classifier outputs, nor test gains enter the gain draw. Once fixed-spherical integration finishes, apply `X=G*Y/||Y||` without changing training or the initial radius.

The original evaluation defaults are 2,000 generated samples, 200 per digit, sample seed 0, batch size 128, and 40 RK4 steps. Each step calls the model four times, for 160 calls per trajectory. `audio_eval.py:20–29` projects the vector field tangentially for the spherical methods and does not renormalize intermediate states. Generation must preserve this integration behavior for all three released seeds.

The UNet consumes `(2,129,63)` inputs, hence D=16,254. Its constructor specifies channels 96, multipliers `(1,2,4)`, 10 classes, conditioning width 256, and attention from level 2 (`audio_flow.py:17,65–74`). Although released metadata say `depth=5`, `make_model` forwards only `ch` and `ncls` to `UNetVel`; it forwards depth as `K` only for `VelNet` (`audio_flow.py:101–102`). Depth therefore does not alter the selected UNet architecture.

Before generation, compute-node checks must strictly load all three EMA state dictionaries and the classifier, reject incompatible keys or shapes, verify input dataset hashes and shapes, and recompute the original radius using the exact split. Evaluate original Y and rescaled X together over all 2,000 labels. Require zero changed predictions, identical correct counts and accuracy, finite logits within the declared tolerance, unchanged normalized directions, and output norms matching G. Keep full-precision measurements alongside historical rounded comparisons; any invariance failure or unexplained baseline mismatch must remain explicit in the result status.
