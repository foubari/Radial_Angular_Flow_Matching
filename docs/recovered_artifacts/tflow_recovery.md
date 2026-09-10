# Recovered t-Flow conditions

This completes the previously authorized t-Flow study for `imagenette_dcae`
and `piv_d32`. It does not repeat the 26 existing conditions or alter their
recorded successes and failures. The recovered A/B/C comparisons run separately.

The datasets, preprocessing, split indices and FID reference are the same
verified inputs described in this directory's release and materialization
receipts. In particular, the existing 3,925-image FID reference contains 2,339
training, 775 validation and 811 test images. This overlap is retained under the
user's instruction to defer reference-protocol changes. The archival checkout
`d3006dc8…` is not a verified training or extraction commit.

## Unchanged protocol

The recovered prepared configurations are reused byte for byte. ImageNette
uses the original SiT, batch 64, 40,000 updates, bfloat16 training, AdamW and
EMA, with seeds 8925, 1234 and 7. It generates 3,000 balanced samples with
sampling seed 0 and 100 actual network evaluations per trajectory. PIV32 uses
the original three-hidden-layer width-128 MLP, batch 256, 10,000 updates and
seeds 8925, 77395 and 65457, with 512 actual network evaluations per trajectory.
The configuration files record the full optimizer, conditioning and evaluator
settings. The published-method adaptations remain those documented for the
existing t-Flow implementation; no RAFM projection is imposed on t-Flow.

Source selection retains the agreed validation-only grid: prior degrees of
freedom 3, 5 and 7, crossed with training-median-calibrated scale multipliers
0.5, 1 and 2. Nine candidates receive 5% of a full training budget; the best
two continue to 10%. This totals 22,000 ImageNette updates and 5,500 PIV32
updates, or 0.55 full-run equivalents per condition. Selection uses validation
radial/projected CDF statistics, never test metrics or FID. Each of the three
final seeds then starts from scratch with the frozen selected source.

## Checks and launch

The original runtime and source pins are preserved from
`outputs_tflow_full/v2/launch/tasks_20260910T080611_91887409.json`.
The additive recovery launcher verifies those pins and requires the release,
cache, unit, backbone, sanity, image-evaluator and public-trainer resume checks.

Full-batch recovered smokes passed in jobs 766309 and 766310 on an AMD MI210.
The separate image public-trainer check passed in job 766324. It uses only 12 disposable optimizer
updates: a 4-to-6 update resume compared bitwise with an uninterrupted six-update
run. Its 40 generated examples are balanced across ten classes and use exactly
100 network calls. Its receipt is the launch gate, not a benchmark result.

```bash
PY=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python
"$PY" -B tools/launch_recovered_tflow.py \
  --node auh7-3b-gpu-030 --workers 6 --time-limit 04:00:00 \
  --extra-source tools/recovered_study_job.sh --submit
```

The launcher schedules two independent one-GPU selection workers, followed by
six independent one-GPU final workers. Immutable submission claims prevent
duplicate launches; failed attempts remain recorded. Check the submission
receipts before reusing this command. Results retain the existing paths under
`outputs_tflow_full/v2/{tuning,final}/{imagenette_dcae,piv_d32}/`.

The actual submission uses tuning array 766332 and final array 766333. After
PIV source selection completed, only final workers 3–5 had their scheduler
dependency narrowed to PIV tuning worker 766332_1. This lets the three PIV
seeds run while ImageNette source selection continues, without changing a task,
source setting or configuration. The explicit orchestration addendum is
`outputs_tflow_full/recovered/launch/piv_dependency_release.json`.
The minutely completion watcher runs as CPU-only step 765765.1 in an existing
allocation; its separate final report destination is
`outputs_rafm_input_study/complete_report_tflow/`.

The initial completion estimate is 1.5–2 hours after available compute starts,
including selection and final evaluation, excluding scheduler/startup delays.
This is an estimate based partly on the measured matched-backbone A/B/C image
training (about 1,752–1,787 seconds per 40,000 updates), not a measured t-Flow
completion time. It must be revised using the t-Flow training logs.
