# Live loss dashboard

This service reads existing loss JSONL files and writes only separate TensorBoard events. It does not import Torch, change any experiment file, restart training, or request a GPU. TensorBoard and its dependencies live in the separate `outputs_monitoring/.venv` environment; exact versions are in `requirements.lock`.

The CPU-only scheduler job is 765765 on `auh7-3b-gpu-015`, with two CPUs, 4 GB RAM and a 24-hour limit. Its actual port and session are recorded in `service.json`. The preferred port is 6010, with 6011–6019 as fallbacks; port 6006 is never used.

The dashboard includes:

- `RAFM_inputs/<dataset>/A_original/seed_*`
- `RAFM_inputs/<dataset>/B_unit_radius/seed_*`
- `RAFM_inputs/<dataset>/C_unit_no_radius/seed_*`
- `tFlow/final/<dataset>/seed_*`
- `tFlow/tuning/<dataset>/candidate_*`

Select `loss/angular_mse` for comparisons between RAFM A/B/C. t-Flow uses the distinct `loss/noise_prediction_mse` tag: its raw numerical scale is a different objective and is not directly comparable to angular MSE. These are the saved instantaneous minibatch losses, not smoothed training averages. Failed evaluations retain their training curves; a decreasing loss does not establish successful sample quality.

All available history is backfilled. New files and appended rows are checked every 15 seconds; TensorBoard reloads every 15 seconds. Audio trainers emit a row every 500 updates, about every three minutes at observed throughput. Vector trainers log every 200 updates. Missing unlogged steps are never invented. Use the **Step** horizontal axis: event wall times record ingestion, not reconstructed historical training times. RAFM's saved cumulative training seconds are a separate scalar.

A repeated or decreasing step starts a separately named `resume_N` segment, preserving the earlier curve. Partial JSONL writes are deferred until a newline arrives. Bridge errors and the latest imported step for every source are recorded in `sessions/<job_id>/bridge_status.json`.

The initial checks use disposable fixtures under this monitoring directory. They verify TensorBoard can read the scalars, source files are untouched, polling does not duplicate points, partial rows are handled correctly, and resumed histories remain separate. Evidence is `bridge_checks.json`; the live HTTP checks are recorded separately.

To stop only this dashboard, use `scancel 765765`. Training remains independent. To start a later monitoring session after this one ends, run `sbatch tools/loss_tensorboard_job.sh` from the repository; a fresh session backfills existing loss files again.

## Connect

From your laptop, using your usual `m3` SSH host alias:

```bash
ssh -N -L 6010:auh7-3b-gpu-015:6010 m3
```

Keep the tunnel running and open [http://localhost:6010/#scalars](http://localhost:6010/#scalars). Filter runs with `RAFM_inputs/audiomnist_stft` to overlay the active A/B/C seeds. Other datasets appear as their trainers write logs.
