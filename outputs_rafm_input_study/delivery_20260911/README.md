# Completed RAFM input and t-Flow study

All 336 prescribed final outcomes across 28 conditions were recorded by
10 September 2026, 19:00 Europe/Paris. No training or evaluation jobs remain.
The two ImageNette scheduler startup failures were recovered on another MI210;
their failed allocations and replacement manifests remain in the archive.

| Method | Complete, audited outcomes | Failed evaluations | Missing |
|---|---:|---:|---:|
| A: original RAFM-Ang | 81 | 3 | 0 |
| B: unit input, radius conditioning | 82 | 2 | 0 |
| C: unit input, constant radius condition | 82 | 2 | 0 |
| t-Flow | 65 | 19 | 0 |
| Total | 310 | 26 | 0 |

The seven A/B/C failures produced nonfinite samples. The 19 t-Flow failed
evaluations contain undefined angular-bin metrics; their available metrics
remain recorded. Failed groups are never reduced to averages of surviving
seeds. A `partial_suite` or `incomplete` field in the original generated report
means incomplete valid metric coverage, not unfinished scheduler work.

## Selected final comparisons

Values are mean ± population standard deviation over the three prescribed
training seeds. Lower FID/sliced W1 and higher digit accuracy are better.

| Metric | A | B | C | t-Flow |
|---|---:|---:|---:|---:|
| ImageNette FID | 147.96 ± 0.38 | 311.40 ± 11.06 | 227.44 ± 67.72 | 223.80 ± 6.27 |
| AudioMNIST digit accuracy (%) | 78.83 ± 1.13 | 77.65 ± 1.35 | 78.68 ± 0.82 | 14.48 ± 0.69 |
| PIV d32 sliced W1 | 0.02023 ± 0.00157 | 0.02109 ± 0.00300 | 0.02063 ± 0.00125 | 0.89337 ± 0.61672 |

ImageNette t-Flow FIDs for seeds 8925, 1234 and 7 are 215.62454, 230.86925 and
224.89443. All evaluated checkpoints completed 40,000 updates. Original
RAFM-Ang has better FID, but t-Flow has lower mean KID: 0.09026 ± 0.00615 versus
0.09577 ± 0.00005. These three-seed descriptive differences do not establish
statistical significance. B has zero image recall in every seed; C's large FID
variance includes a poor seed-1234 result. None has been discarded or retuned.

The B input hypothesis is not consistently supported: B improves the primary
metric versus A in 10 of 26 eligible conditions and worsens it in 16. Seven
conditions improve in all three seeds. B improves versus C in 11 of 27 eligible
conditions and worsens in 16; again seven improve in all three seeds. The full
tables retain all radial, angular, tail, downstream, runtime and parameter
measurements available for each protocol. No pooled score across unlike
datasets is used.

## Reading the delivery

The archive preserves repository-relative paths. Start with
`outputs_rafm_input_study/complete_report_tflow/report.json`,
`per_seed_metrics.csv`, `aggregate_metrics.csv`, `abc_completion.md`, and
`findings.md`. That directory also contains standalone LaTeX tables and PDF/PNG
plots. `recovered_provenance_note.md` supersedes the frozen generator's obsolete
sentence claiming image/PIV artifacts remain blocked. The main manuscript is
unchanged, as requested.

Every prescribed run's `result.json` is included, together with its available
JSON metadata, sample audit and training log. Checkpoints and sampled tensors
remain at the original paths recorded in those results; large tensors are not
duplicated in this analysis archive. Source commits, immutable launch manifests,
prepared configurations, selection receipts, dataset/split hashes and audit
documents are included. `manifest.json` hashes every archived member, and
`archive_verification.json` records verification of the finished ZIP.

## Interpretation limits retained

- Synthetic comparisons using newly shared, explicitly seeded caches are
  labelled new matched realizations; they are not historical sample recovery.
- The unchanged ImageNette FID reference has 3,925 images, overlapping generator
  train/validation/test by 2,339/775/811. It is not a held-out reference.
- t-Flow has all metrics from the original ImageNette evaluator. The additional
  angular-bin diagnostics introduced for A/B/C image/audio comparisons are not
  present for t-Flow in those downstream result files. This diagnostic gap does
  not invalidate its measured FID/KID/PRDC, radial or downstream scores.
- `docs/rafm_toy_numerical_failure_analysis.md` documents inherited angular-path
  numerical issues, including finite but extreme losses. Finite results alone
  do not establish numerical reliability of that toy implementation.
- `docs/tflow_matched_backbone_failure_analysis.md` qualifies the behavior of
  this matched-backbone, direct-noise implementation. These measurements are
  not a general claim that the published t-Flow method is inferior.
- The archived checkout `d3006dc8…` identifies the location of recovered
  artifacts, not a verified historical training or extraction commit.
