# Shared t-Flow and RAFM-Ang input study launches

The full study is authorized. The launcher submits only after concrete unit,
condition-sanity and source-commit checks pass. It does not train on a login node
or change any experiment configuration. Running it without `--submit` writes a
reviewable immutable JSON plan and performs no scheduler submission.

## Allocation and task layout

| Study | Node | Tasks | Maximum simultaneous GPUs |
|---|---|---:|---:|
| t-Flow source selection | auh7-3b-gpu-008 | 26 logical tasks, 6 workers | 6 |
| t-Flow final seeds | same node, after the tuning array ends | 78 logical tasks, 6 workers | 6 |
| RAFM-Ang A/B/C | auh7-3b-gpu-015 | 234 logical tasks, 6 workers | 6 |

Each worker requests exactly one MI210, four CPU cores and 32 GB host memory in
`hermes-2`. Its disjoint round-robin task shard runs sequentially in isolated
child processes. Models are never sharded and concurrent experiments never share
one GPU. Persistent workers amortize the observed roughly 2.5-minute scheduler
prolog across many short vector runs. Resources return when each shard finishes;
no worker waits idle for another worker's tasks. The t-Flow final array uses `afterany` on the entire tuning array;
there is no overlap between the two t-Flow phases. This permits all three long
AudioMNIST final seeds to run together instead of serially after tuning.

A failed tuning condition does not cancel unrelated final tasks. Each final task
independently requires its own valid frozen source selection and immutable
`selection_receipt.json`. The receipt pins the selection bytes across all three
final seeds and is checked before and after each final run. No final model starts
with a failed, missing or changed selection. Audio is prioritized first, followed by
larger vector dimensions. ABC places A/B/C tasks adjacent for each seed.

The partition's `MaxTime=UNLIMITED` was inspected read-only on 2026-09-10.
The launch default is a 48-hour time limit per task, adjustable with
`--time-limit`; this is a scheduling limit, not a runtime estimate or an
extension to the prescribed optimizer-update budget.

## What is recorded and checked

The planner reads `configs/rafm_input_study/prepared/*.json` and requires exact
agreement with `materialization.json`. The two unresolved conditions remain
explicitly recorded: PIV d32 has no verified input realization, and ImageNette
DC-AE has unresolved split/evaluation provenance. They are not silently replaced
by another configuration or counted as completed.

Each task includes the full configuration, its canonical and file hashes, runtime
implementation hash and source-file hashes, exact command, source commit,
condition and model seed. Unit reports and passing condition-sanity evidence are
also pinned by SHA-256. ABC additionally requires all parameterization/runtime/preparation unit suites
and all nine A/B/C × MLP/U-Net/SiT rows in `full_backbone_checks.json`, with the
current checker-script hash and current model/backbone implementation hashes.
The planner uses only the Python standard library; on a compute node, the worker
compares its planned implementation hash against the runner's actual function
before any experiment command.

Submission requires the relevant implementation, configurations, launch tools
and independent sample auditor to match the recorded Git commit. Commit those
files before requesting submission. Subsequent documentation/results commits
are permitted when all frozen source/configuration bytes remain unchanged.
Changing frozen experiment files while tasks remain queued causes explicit
failure; it never silently changes the pending experiment.

Use the existing environment's Python for planning so package-version metadata
matches the compute environment. Planning does not import Torch or execute model
code. Every final successful train/evaluate command is followed by:

```bash
python tools/audit_study_samples.py --result /absolute/path/to/result.json
```

The independent auditor records actual generated sample count, class balance,
finite rows and sample-file identity in the sibling `sample_audit.json`.

## Commands

From the experiment repository:

```bash
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
STUDY_PYTHON=/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/bin/python

# Reviewable plans only; no jobs start.
"$STUDY_PYTHON" tools/launch_shared_study.py --study tflow
"$STUDY_PYTHON" tools/launch_shared_study.py --study rafm_inputs

# Run after the corresponding checks finish and source/configuration commit exists.
"$STUDY_PYTHON" tools/launch_shared_study.py --study tflow --submit
"$STUDY_PYTHON" tools/launch_shared_study.py --study rafm_inputs --submit
```

Each invocation writes a new manifest under the study's `v1/launch/` directory.
`--manifest /absolute/new/path.json` can choose the path; existing manifests are
never overwritten. A sibling `.submission.json` records exact `sbatch` commands,
job IDs and submission errors. An atomic
`submission_claim.json` prevents simultaneous planners from submitting duplicate
sweeps using different manifest filenames. Existing successful/partial submission
records also refuse a second full sweep. A claim remains after a submission
failure so the actual scheduler state can be inspected before any retry.

The scheduler shell uses the immutable manifest path and SHA-256. Each worker
writes `v1/executions/<job>_<array-index>_worker.json`; every logical task has its
own `..._task_<logical-index>.json` with commands, timestamps, hardware, source
identity, status and failures. An exclusive filesystem lock prevents two workers
from concurrently changing the same condition/arm/seed outputs. Logs are under `v1/logs/`. Separate
per-worker MIOpen kernel/performance, Inductor, Triton and plotting directories
live under `v1/cache/`. Sequential children reuse their worker's compiled kernels
without sharing writable caches with another worker.

## Failures and resumption

A failed condition/seed terminates that logical task only. The worker records
the failure and continues later unrelated tasks in its shard. It exits nonzero
after finishing its shard if any logical task failed.
The launcher never substitutes a different prior, solver, batch, seed, dataset,
precision or training budget. Nonfinite outcomes and metric failures remain in
the underlying runner's reports and the worker execution record.

Jobs do not automatically requeue. After a scheduler interruption, inspect the
execution and Slurm records, then resubmit only the interrupted worker array
indices using the same shell, phase, manifest and digest. Its already-completed
logical tasks are skipped and independently audited again; unfinished logical
tasks resume their own checkpoints. The original runner restores
its exact checkpoint, optimizer and RNG state. A completed run is skipped by the
runner and its samples are audited again. Existing scientific-failure records
continue to refuse automatic reruns. Do not resubmit the entire study.

If the second t-Flow array submission fails after tuning was submitted, its
partial submission record preserves the tuning job ID. Inspect that record and
submit only the missing final worker array with the recorded `0-5%6` range and
`afterany:<tuning-job-id>` dependency. Never repeat the first phase by blindly
rerunning `--submit`.

For a manual resume, retain the manifest's node, one-GPU request and `%6` throttle;
wait for active tasks or include a dependency so the two waves do not exceed six
concurrent jobs on the node. Reuse the exact reviewed shell command in the
submission record, changing only the interrupted worker array index selection and the
necessary dependency. No experiment configuration changes are involved.

## Validation of the launcher

Static Python compilation and shell syntax passed. Fifteen standard-library
checks passed using mocked scheduler calls and temporary fixtures. They cover
all 26 current t-Flow sanity hashes, logical task counts/order/disjoint shards,
login-node refusal, failed and incomplete unit suites, all nine required backbone
rows, stale checker/model hashes, altered sanity evidence, six-worker phase
barriers, partial-submission preservation, duplicate submission and atomic claim
rejection, frozen selection receipt changes, continuation after unrelated logical
failures, and filesystem lock exclusion. No actual `sbatch` or model execution
occurred in these tests or the initial dry plan.
