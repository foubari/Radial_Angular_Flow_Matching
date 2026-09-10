# Reporting the complete RAFM input study

The completion target is **252 A/B/C final runs**: 28 conditions, three arms and
three prescribed training seeds. Existing t-Flow results are retained as a
separate 84-run reference inventory. Missing recovered-condition t-Flow runs do
not prevent reporting completion of A/B/C.

Run the following from the repository root, on the login node for a lightweight
status read or in the final CPU scheduler job:

```bash
/usr/bin/python3 -B tools/report_rafm_completion.py
```

This freshly reads the complete configuration inventory, result files, dataset
manifest hashes and independent sample-audit receipts through the existing
standard-library collector. It does not import Torch, load tensors or models,
perform evaluations, query the scheduler, launch jobs or edit frozen runtime
files. It writes only its `abc_*` artifacts under
`outputs_rafm_input_study/complete_report/`, leaving the existing collector's
`report.json`, tables and `outputs_rafm_input_study/v1/report/` unchanged.

For a final CPU report job after both the original A/B/C and recovered A/B/C
arrays end, use:

```bash
/usr/bin/python3 -B tools/report_rafm_completion.py --require-finished
```

An unresolved A/B/C run causes exit code 2 **after** writing the report. An
inventory with every prescribed outcome recorded but scientific failures is
labelled `finished_with_failures`; exit code 0 means the reporting operation
completed, not that every scientific run passed. Successful metrics require
`status: complete` and `all_metrics_successful: true`.

An existing validated 28-condition collector snapshot can instead be rendered
with `--report outputs_rafm_input_study/complete_report/report.json`. The report
records that snapshot's path, hash and collection time; this mode does not
represent later-arriving results. The default command is preferred for final
reporting. A 26-condition report or changed configuration/plan hash is rejected.

Outputs:

- `abc_completion.md`: readable completion, comparisons, costs, all per-seed
  links, failures and interpretation limits. Display timestamps use Paris time.
- `abc_completion.json`: complete metadata, three-seed aggregates, B−A/B−C
  deltas and consistency, parameter overhead, radius drift, memory, runtime
  scopes, dataset identities and the qualified t-Flow/audio references.
- `abc_seed_status.csv`: exactly 252 prescribed A/B/C rows, including missing,
  partial, failed and incompatible runs.
- `abc_per_seed_metrics.csv`: all recorded A/B/C metrics; partial finite values
  retain the run's failure status.
- `abc_aggregate_metrics.csv`: mean and population SD only for three compatible
  audited final seeds. Incomplete groups have explicitly empty aggregates.
- `abc_tflow_reference_metrics.csv`: existing t-Flow results and missing/failure
  rows, kept separate from the A/B/C completion count.
- `abc_completion_files.json`: SHA-256 checksums for these report artifacts.

A directory without a final result is labelled `in_progress_or_interrupted`:
file existence is not evidence that a scheduler allocation is live. Scientific
toy failures are preserved, not treated as scheduler-recoverable. See the
[toy numerical audit](rafm_toy_numerical_failure_analysis.md) and
[t-Flow adaptation audit](tflow_matched_backbone_failure_analysis.md).
New shared synthetic realizations remain explicitly labelled. Recovered image
results preserve the authorized reference overlap, which is reported explicitly
and is not described as a disjoint held-out image reference.
