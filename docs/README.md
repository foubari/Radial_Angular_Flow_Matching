# RAFM extension preparation

Implementation branch: `experiments/tflow-empirical-gain` in
`foubari/Radial_Angular_Flow_Matching`, based on
`5b89ed5f4af8a47c3b57eb9d595203daafc6d2c4`.

**No training, tuning, checkpoint evaluation, or benchmark sampling has been
launched for this extension.** The next experiment approval is Stage A in
[the concrete run plan](run_plan.md). Passing correctness tests does not establish
benchmark performance or remove the documented artifact and protocol blockers.

- [Run plan and resource estimates](run_plan.md): five resolved sanity cases,
  validation-only source selection, full-suite scope, and explicit launch commands.
- [Experiment matrix](experiment_matrix.md) and
  [machine-readable inventory](../configs/tflow/suite_manifest.json): all 28
  conditions, historical protocols, hashes, and unresolved discrepancies.
- [t-Flow method](tflow_method.md): independent equation-based implementation,
  source provenance, endpoint choices, actual network-call budgets, and limitations.
- [Audio empirical-gain control](audio_empirical_gain.md): original checkpoint
  requirements, exact interpolated ECDF pairing, and complete invariance checks.
- [Proposed manuscript text](manuscript_additions.md),
  [pending AudioMNIST assets](proposed_audio/reporting_manifest.json), and
  [pending full-suite report](proposed_tables/tflow_results.md): existing results
  are preserved; pending additions contain no fabricated measurements.
- [Validation record](artifact_audit/validation_summary.json): executed static,
  CPU correctness, backbone-interface, and read-only input checks.
- [Access record](artifact_audit/access_checks.json): sources inspected and exact
  remaining access failures. A failed anonymous private-release lookup does not
  establish that an artifact is absent.

Prepared JSON configurations are under `configs/tflow/prepared/`. Unresolved
conditions refuse execution. Final model jobs require a frozen validation-only
source choice and an allocation with exactly one visible GPU. Reporting requires
all three final seeds and preserves failed or incomplete conditions explicitly.
Finite large-radius diagnostics are reported unchanged; they are not treated
as proof of numerical failure for a heavy-tailed distribution.

New generated tensor datasets belong under
`/mnt/vast01/users/fouad.oubari/data/tflow/`. Existing pinned input caches are read
in place. No historical input, checkpoint, result, FID reference, or paper table
is modified by this preparation.
