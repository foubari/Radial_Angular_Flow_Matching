# Tabular MSGM sparse source audit

Audited commit: `eb80c6b5af1f3f47d35bf0f4d9ac0c83be9bce5a`.
This is a static source and serialization-metadata audit, not a completed runtime
verification. No scientific code or configuration was edited, no tensor computation
was performed, and no training or scheduler submission was executed by the auditor.

## Protocol retained

All seven `configs/exp1_sparse/*.yaml` dataset blocks match their corresponding
`configs/exp1/*.yaml` blocks. Their differences are the selected `msgm_sparse`
method/adapter and `output_dir: outputs_msgm_sparse`.

`configs/defaults.yaml` remains unchanged: hidden width 128, three hidden layers,
no premodule, Adam LR 0.001, batch 256, 10,000 training steps, checkpoints every
5,000 steps, 10,000 generated samples, RK4 with configuration `nfe: 128`, three
seeds from `get_seed_list(3, 42)` (8925, 77395, 65457), 500 sliced projections,
200 angular projections and four radial bins. Dataset splits are 60/20/20 with
split seed 0. Each config command executes its three seeds sequentially on one
device; the command has no per-seed CLI override.

| Config | Dataset output name | Dim | Input / total rows | Split sizes |
|---|---|---:|---|---|
| gaussian_d16 | gaussian_aniso_d16_cor | 16 | Synthetic Gaussian, 50,000 | 30,000 / 10,000 / 10,000 |
| studentt_d16 | student_t_d16_df3.0_cor | 16 | Synthetic Student-t(df=3), 50,000 | 30,000 / 10,000 / 10,000 |
| studentt_d32 | student_t_d32_df3.0_cor | 32 | Synthetic Student-t(df=3), 50,000 | 30,000 / 10,000 / 10,000 |
| toy2d | toy_radial_angular | 2 | Synthetic radial/angular mixture, 50,000 | 30,000 / 10,000 / 10,000 |
| piv_d16 | piv_d16 | 16 | Committed data/piv/piv_d16.pt, 998 | 598 / 199 / 201 |
| piv_d64 | piv_d64 | 64 | Committed data/piv/piv_d64.pt, 998 | 598 / 199 / 201 |
| piv_d256 | piv_d256 | 256 | Committed data/piv/piv_d256.pt, 998 | 598 / 199 / 201 |

PIV tensor metadata was read with standard-library ZIP/pickle-opcode inspection,
without deserializing or computing tensor values. The files contain CPU FP32
tensors of shapes (998,16), (998,64), and (998,256). The d16 tensor has stride
(32,1), a valid noncontiguous view. Actual finiteness must still be checked in an
allocated job. `rafm/data/piv.py:46-53` loads the existing file, selects its first
`dim` columns, applies the existing global mean subtraction in memory, and splits;
it does not write or regenerate PIV files.

## Data-draw and dense-stack limits

`experiments/exp1_main_benchmark.py:154-163` creates the dataset before any model
seed is applied. Synthetic dataset generation uses global Torch RNG state:
`rafm/data/toy_radial_angular.py:47-57`, `student_t.py:55`, and
`gaussian_aniso.py:50`. Split seed 0 fixes indices, and matrix seed 42 fixes the
mixing matrix; neither fixes the synthetic observations. Therefore the unchanged
fresh command shares its draw among its three model seeds but does not reproduce
the exact historical dense-run observations. Unchanged exp1 generates synthetic
data before model seeding; independent invocations may use different draws.
No new seed or alternative data generation was introduced by orchestration.

The sparse drift is the prescribed three-hidden-layer Swish MLP
`(d+1) -> 128 -> 128 -> 128 -> d`, with no preprocessing. `_MLPDrift` only coerces
the time argument to (batch,1). Optimizer, budgets, data access and metric calls
match the dense adapter/runner at the configured defaults. Sparse training logs
every 200 steps; the dense adapter logs every 500. Both checkpoint every 5,000
under the shared configuration.

The SDE stacks are distinct: sparse uses vendored `sdeflow-light` commit
590ec4b417a3cb4a136d4d80d37129fb52f6a241 and explicitly configures
`MSGMsde(denseTensor=False, beta_min=.1, beta_max=20, T=1, norm_sampler='ecdf',
estim_cst_norm_dens_r_T=False)` plus Rademacher SSM, `debias=False`,
`ssm_intT=False`. Dense imports `multiplicativeNoise` and `PluginReverseSDE` from
the absent `18727_Multiplicative_Diffusion_code` checkout. Its upstream defaults
and claimed verbatim MLP provenance cannot be independently compared from this
checkout alone. Do not describe these as a proven byte-identical stack with only
one Boolean changed.

Both MSGM adapters interpret `nfe=128` as 128 RK4 steps, hence 512 network calls.
The tabular FM sampler also runs 128 RK4 steps and reports 512 actual calls
(`rafm/flow_matching/sampler.py:79-87`). This differs from the separate
image/audio experiment's sampling-cost caveat. MSGM result JSONs omit NFE metadata.

## Completion and resume checks required

`baselines/msgm_sparse_runner.py:22` skips any existing `metrics.json` without
parsing it. At :40 it catches ImportError and returns, which can leave an overall
zero process exit code despite missing results. Successful process exit and file
existence are insufficient completion evidence.

Checkpoint resume restores model/optimizer/step/elapsed time only
(`msgm_sparse_adapter.py:95-99`): no RNG, configuration or dataset state is saved.
Synthetic restarts additionally recreate the unseeded dataset. Resume is not an
exact continuation of an uninterrupted run. At :133-134 the adapter deletes its
checkpoint immediately after training, before sampling/evaluation; an eval failure
then leaves no trained model to resume from. No trained model artifact survives a
successful run. Do not silently restart interrupted work while claiming exact RNG
or data continuation.

For every expected dataset/seed, independently require:

- `train_log.csv` with final step 10000 and finite logged losses/times;
- `samples.pt` with shape (10000,dim), real floating dtype, all entries finite;
- all 19 required numeric metric keys, all finite, and `nan_rate == 0`;
- valid seed/dataset identity from the output path and verified unchanged launch
  configuration, since metrics JSON has no seed, step or configuration metadata.

The exact 19 metric keys are:

```text
radial_w1 ks_stat q950_err q990_err q995_err tail_exc_95 tail_exc_99
sliced_w1 mmd
angular_sw_bin0 angular_sw_bin1 angular_sw_bin2 angular_sw_bin3 angular_sw_mean
nan_rate exploding_norm_rate invalid_rate
sample_time_s total_train_time_s
```

`rafm/metrics/angular.py:44-46` deliberately emits NaN for a bin with fewer than
10 generated or reference points. `json.dumps` in the runner allows NaN.
`stability_metrics` checks NaN, not every possible infinity. Therefore inspect raw
samples and all metric values rather than relying on stability fields alone.
Do not replace undefined metrics or change bins/sampling to force a passing report.

## I/O and resource containment

The selected sparse path explicitly writes only per-seed `train_log.csv`, its
temporary checkpoint, `samples.pt`, and `metrics.json` below the configured
`outputs_msgm_sparse/exp1_main_benchmark/<dataset>/msgm_sparse/seed_<seed>` path.
Diagnostic SDE plotting is disabled. Synthetic generators do not save their draws.
Do not run PIV preparation or unrelated experiment mains.

Library imports can create Python bytecode and plotting/runtime caches. Orchestration
must disable bytecode and direct caches/temp files/logs to the authorized output
tree. Use a fresh tabular process/PYTHONPATH and verify SDE/module origins to avoid
accidentally reusing imports from the running image/audio checkout.

Metrics run on CPU after samples return to CPU. Synthetic MMD materializes a
20,000-by-20,000 distance matrix (about 1.6 GB in FP32), plus masks, selected values,
median scratch space and kernel matrices. Allocate ample host RAM (at least roughly
8 GB, preferably more) and bounded CPU threads in the Slurm job; do not compute
these metrics on the login node. No additional scientific dependency is needed
beyond those handled by root's environment setup.
