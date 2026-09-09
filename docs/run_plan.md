# Prepared experiment plan — launch approval required

Repository: `foubari/Radial_Angular_Flow_Matching`.
Working branch: `experiments/tflow-empirical-gain`.
Base commit: `5b89ed5f4af8a47c3b57eb9d595203daafc6d2c4`.
Manuscript SHA-256: `97f709df2a45e4acf4ba186379c60e9c75120d95e09af782d7a18c8b56d00367`.

No training, tuning, benchmark generation, checkpoint evaluation or full sweep has
been launched for this extension. Resource access, implementation, static checks,
read-only input verification and lightweight correctness tests are authorized.
The commands below are prepared commands, not a record that they were executed.

## Implementation and provenance

- Direct t-Flow noise prediction, multivariate shared-scale Student-t source,
  straight interpolation and the corresponding field are in `baselines/tflow_core.py`.
- The existing MLP, U-Net and pinned official SiT are reused. SiT commit
  `cbde832a40b153ccc79603412409da9c9b0c568c` is checked out at
  `/mnt/vast01/users/fouad.oubari/references/SiT-tflow`.
- There is no verified authors' t-Flow code repository. This is an independent
  reproduction of the inspected arXiv-v2 equations, not an adaptation of t-EDM.
  The final ICLR PDF was located, but the cluster proxy returns HTTP 403
  `ERR_ACCESS_DENIED`; the web reader rejects its 33.6 MB size. Final-version
  algorithm differences remain unverified. See `tflow_method.md`.
- The sampler uses Heun with exact counts 512/160/100 calls per sample for
  vector/audio/image benchmarks. It starts at `t_min=0.001`, uses a rho 7 grid
  in sigma=1-t through sigma 0.01, then explicitly reaches t=1. These endpoint
  choices are documented numerical adaptations, not claimed verbatim pseudocode.
- Source/configuration/code/environment hashes bind checkpoints, selection and
  results. Failure records are preserved. Final aggregation requires all three
  finite seed results and never averages an unreported surviving subset.
- Existing paper tables and results are untouched. Standalone renderers append
  measured rows; pending previews cannot invent t-Flow or empirical-gain points.

## Stage A: requested first experiment approval

Approve only **GPU sanity on the five currently resolved conditions**:

| Condition | Sanity updates | Original batch | Generated sanity samples | Actual calls/sample |
|---|---:|---:|---:|---:|
| PIV64 |16|256|128|512|
| PIV256 |16|256|128|512|
| Finance |16|4096|128|512|
| Weather |16|2048|128|512|
| AudioMNIST |4|32|40, four per digit|160|

These are explicitly labelled sanity runs, never paper results. They use only
training/validation data, fixed prior nu 5 with training-median source scale,
the original optimizer/batch/backbone/precision, and no checkpoint reuse in
the final experiments. They check finite loss, backbone gradients and samples;
record peak memory, actual call counts and short-loop timing. Short-loop timing
is not presented as steady-state training throughput. Any failure stops that
case and is reported; no automatic LR/batch/solver workaround is allowed.

Use one MI210 GPU at a time on a compute node for this initial stage. The
planning allowance is **under one GPU-hour**, excluding scheduler delay;
this is an estimate, not a measured duration. The ImageNette sanity remains
blocked by the unresolved exact split/reference-disjointness record.

After Stage A, report the outcomes and request approval for the documented
tuning/final-run stage. Stage A approval does not authorize that later stage.

## Proposed source selection, not yet approved

For each resolved condition, use a separate tuning seed 46021 and nine candidates:
prior degrees of freedom `{3,5,7}` (finite-variance settings used in the paper)
crossed with source-scale multipliers `{0.5,1,2}`. For each nu,

`reference_scale = median(training radius) / sqrt(d * F(d,nu).ppf(0.5))`.

This matches the median of the multivariate-t source radius at multiplier 1.
It changes neither input preprocessing nor the meaning of the noise target.
No target degrees of freedom, target variance, test metric or test radius
selects the prior. The sampler equations use the selected scale consistently.

Every candidate trains to 5% of the original budget. The best two continue to
10% total. The budget per candidate/stage is 500→1000 vector steps,
1200→2400 audio steps, and 2000→4000 image steps. Score 1000 validation samples
using equal weights on radial KS and mean projected KS over 64 fixed directions
(projection seed 61719, sample seed 61717). This bounded CDF criterion is separate
from all reported test metrics. It remains meaningful for infinite-moment targets.
For class-conditioned benchmarks, the 1000 samples are exactly class-balanced.

Total tuning cost is `9*0.05+2*0.05=0.55` full training runs per condition.
Freeze the winning nu/scale and restart all three paper model seeds from
scratch for their complete original budgets. No early stopping or test-set
selection applies to those final runs.

The tuning pilot is deliberately short. It may not reliably rank configurations
that learn at different speeds; that limitation must accompany results. A larger
tuning budget requires a separately reviewed change, not an automatic extension.

## Full-suite resources: planning estimate only

The inventory contains 28 unique conditions: 26 vector conditions, AudioMNIST
and ImageNette. Three final seeds yield 84 complete training jobs. Including
the proposed tuning gives 99.4 condition-specific full-run equivalents:

| Family | Total optimizer updates including tuning | Assumed time/update | Estimated training GPU-hours |
|---|---:|---:|---:|
|26 vector conditions|923000|2–20ms|0.51–5.13|
|AudioMNIST|85200|100–400ms|2.37–9.47|
|ImageNette|142000|150–500ms|5.92–19.72|

These throughput intervals are **planning assumptions, not t-Flow measurements**.
They imply about 9–35 GPU-hours for training; allow roughly **10–45GPU-hours
including validation sampling and final evaluation**. The interval excludes
queue delay, failed attempts, old-baseline reruns and additional diagnostics.
We will replace it with measured estimates after approved compute runs.

For the eventual full sweep, use at most six independent single-GPU jobs on
one MI210 node, allocating only work that is ready. Reserve no multi-node or
distributed training resources. Image/audio have the longest individual jobs;
vector conditions can fill remaining slots. Historical paper timing hardware
and precision differ; no new speedup ratio is claimed from these assumptions.

## Artifact and protocol blockers

1. Three fixed-spherical AudioMNIST EMA24000 checkpoints and their `meta.json`
   are still absent. All four/optional angular Table 6 reference-method checkpoint
   groups are also absent. No substitute or retraining is authorized.
2. Exact cached inputs are absent for 12 Student-t conditions, six anisotropy
   conditions, the Gaussian control and the 2D toy. Model/split/matrix seeds alone
   cannot recover the historical random draws.
3. PIV32 input identity and the PIV16/32 historical batch provenance remain
   unresolved. A d32 backing-storage candidate in the d16 tensor is not yet an
   established historical input.
4. Student-t batch provenance conflicts between paper and retained configs.
   Main PIV64/256 were resolved using `RAFM_ANG_TABLE1_completion.json`; their
   exact tensor split hashes are checked separately.
5. ImageNette needs the actual generator train/validation/test row indices and
   reference-to-input mapping. The paper claims a reference-disjoint training
   split; retained extraction/training code instead splits combined official
   train+validation rows. We cannot silently select either reconstruction.
6. The RAFM repository exposes zero releases and zero workflow artifacts.
   `gh` has no authenticated session; the private MSGM release endpoint returns
   404 anonymously. No fixed-spherical checkpoint or synthetic cache was found
   in accessible repository history/branches or documented storage paths.
7. Main LaTeX sources remain absent. This does not block code or standalone
   replacement tables/figures/text; inserting them into the compiled manuscript
   is a later source-file dependency.

If the 20 missing synthetic caches cannot be recovered, a separately labelled
paired rerun can use new tensors cached once under the data root, with hashes
and shared splits for every rerun method. A minimal FM comparison would include
t-Flow, Gaussian FM, matched-source FM, RAFM-Vel and RAFM-Ang. That adds 240
baseline training runs beyond t-Flow for 20 conditions×four controls×three seeds.
Adding both dense and sparse MSGM would add another 120 runs. These comparisons
must remain separate from historical rows, and neither new data generation nor
these extra reruns is authorized by this plan. The comparison set and cost
require explicit approval if recovery fails.

## Reproducible commands after the corresponding approval

Run from `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions`. Python is the
existing ROCm environment at `../msgm-sparse-control/.venv/bin/python`.

```bash
# Stage A, one explicitly selected resolved condition; single GPU on compute.
sbatch --export=ALL,TFLOW_PHASE=sanity,TFLOW_CONDITION=piv_d64 tools/tflow_job.sh

# Later tuning approval; executes exactly the nine-candidate budget.
sbatch --export=ALL,TFLOW_PHASE=tuning,TFLOW_CONDITION=piv_d64 tools/tflow_job.sh

# Later full-run approval; repeat for all three recorded seeds/each ready case.
sbatch --export=ALL,TFLOW_PHASE=final,TFLOW_CONDITION=piv_d64,TFLOW_SEED=8925 tools/tflow_job.sh

# Static preparation and completeness-aware reporting do not launch experiments.
python3 -m experiments.tflow.prepare_configs
python3 tools/check_static_tflow.py
python3 -m experiments.tflow.render_results --results-root outputs_tflow \
  --output-dir outputs_tflow/report
```

The full list of conditions is `configs/tflow/suite_manifest.json`; all 28 draft
configs are under `configs/tflow/prepared/`. No script loops over blocked cases
or silently regenerates an input. Audio post-hoc checkpoint arguments and the
measured-only Figure 2/Table 5 renderer are documented in `audio_empirical_gain.md`.
