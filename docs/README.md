# RAFM experiment extension

Implementation branch: `experiments/tflow-empirical-gain` in
`foubari/Radial_Angular_Flow_Matching`, based on
`5b89ed5f4af8a47c3b57eb9d595203daafc6d2c4`.

**The user authorized the matched t-Flow and RAFM-Ang A/B/C study on
2026-09-10; it is running on separate cluster compute nodes.** See the
[launch/status record](../outputs_rafm_input_study/v1/STATUS.md),
[current result snapshot](../outputs_rafm_input_study/v1/report/README.md), and
[measured findings](../outputs_rafm_input_study/v1/report/findings.md).
Twenty-six of 28 requested conditions have verified inputs; native PIV d32 and
the image split/reference mapping remain blocked. No completed experiment or
quality improvement is implied by successful correctness checks.

The active reproducible configurations are `configs/rafm_input_study/prepared/`.
The [shared protocol](rafm_input_study_protocol.md),
[A/B/C parameterization](rafm_input_method.md), and
[launch commands and immutable manifests](shared_study_launch.md) supersede the
older preparation-only launch restrictions below. New synthetic realizations
are explicitly labelled shared comparisons; historical results are preserved.
The [t-Flow backbone failure analysis](tflow_matched_backbone_failure_analysis.md)
documents the interpretation limits of the observed failed runs, including the
direct-noise MLP's output-rank limitation. Failures are retained, not retuned.

The completed fixed-spherical audio gain control and its measured current-backend
reevaluation are separate references. Exact digit predictions, energy KS and tail
coverage agree; small PIT/radial reduction differences and the discrepancy from
historical paper accuracy remain recorded.

The following documents retain the earlier preparation and provenance audit:

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

Earlier prepared JSON configurations are under `configs/tflow/prepared/`. Unresolved
conditions refuse execution. Final model jobs require a frozen validation-only
source choice and an allocation with exactly one visible GPU. Reporting requires
all three final seeds and preserves failed or incomplete conditions explicitly.
Finite large-radius diagnostics are reported unchanged; they are not treated
as proof of numerical failure for a heavy-tailed distribution.

New generated tensor datasets belong under
`/mnt/vast01/users/fouad.oubari/data/tflow/`. Existing pinned input caches are read
in place. No historical input, checkpoint, result, FID reference, or paper table
is modified by this preparation.
