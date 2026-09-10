# Recovered t-Flow monitoring and final reporting

The additive `tools/monitor_recovered_tflow.py` reads the pinned two-condition launch manifest and submission receipt, frozen source/config checksums, tuning-selection records, six final results and their sample audits. It polls scheduler state and JSON logs every 60 seconds for at most eight hours. It never submits, cancels, retries, trains, samples or recomputes experiment metrics. A recorded scientific failure remains a failure.

Run on the existing CPU allocation, without requesting any GPU:

```bash
cd /mnt/vast01/users/fouad.oubari/msgm/rafm-additions
srun --overlap --exact --jobid=765765 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=1G --gres=none --unbuffered \
  /usr/bin/python3 -B tools/monitor_recovered_tflow.py \
  --manifest outputs_tflow_full/recovered/launch/tasks_20260910T153538_4845907e.json \
  --manifest-sha256 9c179f2d3686a28af35b60d409ad39f119ba2482d56d89deb76880cc621dca8c \
  --tuning-job 766332 --final-job 766333
```

Use the scheduler step's persistent stdout/stderr capture. The first `latest.json` with `monitor_mode: persistent`, its UTC/Paris timestamp, and a live CPU step establish that monitoring started; a single `--once` snapshot does not.

Outputs are atomic `outputs_tflow_full/recovered/monitor/latest.json`, append-only `events.jsonl`, and an exclusive once-only `reporter.json`. An exclusive process lock prevents duplicate monitors. Logs without elapsed times produce no throughput or ETA. Allocation/prolog state is kept distinct from actual logged updates.

When all six recovered t-Flow outcomes and all other prescribed outcomes are complete or explicitly failed, the watcher writes the separate `outputs_rafm_input_study/complete_report_tflow/` report. This cannot collide with the earlier ABC report job's `complete_report/`. Completion depends on audited artifacts, not Slurm cleanup. The pipeline calls the frozen shared collector with the complete 28-condition inventory and CPU-only plotting, then the ABC completion reporter, then the findings writer and final file manifest. The plotting command uses the existing project virtual environment; the other processes use stdlib Python. No frozen reporting or experiment source is modified.

The frozen collector's generated `manuscript_addition.md` still contains an obsolete PIV/image-blocker sentence. The additive `recovered_provenance_note.md` explicitly supersedes it. `findings.md` and `report.json` supply actual outcomes, all scientific failures, qualified t-Flow comparisons, the newly cached synthetic realization label and image-reference overlap limitations. The main manuscript is not edited.

Exit 0 means all reported outcomes succeeded. Exit 1 means the complete report includes recorded scientific failures, including failures from the earlier 26-condition study. Exit 2 means an execution/provenance blocker, three consecutive scheduler-query failures, unresolved results/audits 60 seconds after the final array ends, a reporting failure, or the bounded timeout. A blocked report attempt is preserved and never automatically retried.

## Controller purge and the authorized node-failure recovery

The initial monitor stopped at 18:10:44 Paris because Slurm had purged completed tuning array 766332 from its controller. Its `squeue` returned `Invalid job id specified` while `sacct` positively reported both tasks completed. The monitor now accepts precisely that controller-purge error only when accounting succeeds and every expected task is terminal. Authentication, network and incomplete-accounting errors remain failures.

The initial `monitor/` outputs, launch receipt and log remain preserved. The manually authorized replacement array 766523 runs only original final worker indices 1 and 2 (ImageNette seeds 1234 and 7), which had `NODE_FAIL` before creating output directories. Its separate recovery manifest and submission receipt record the intervention. This is infrastructure recovery, not a retry of a scientific failure.

Restart the command above with these additional arguments, keeping the original manifest and tuning/final IDs:

```bash
--additional-final-array 766523:2 \
--output outputs_tflow_full/recovered/monitor_retry001
```

The original and every explicitly listed replacement array must all be terminal before missing-outcome grace begins. Complete audited outcomes can still trigger the combined report without waiting for scheduler cleanup. This option only watches root-authorized jobs; it cannot create or retry them. The new launch receipt should pin both the watcher source and the separate recovery manifest/submission receipt.
