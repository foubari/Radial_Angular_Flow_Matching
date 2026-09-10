# First completed recovered ImageNette comparison

Read-only audit on 2026-09-10, approximately 17:10 Europe/Paris. These are
full-budget seed-8925 results, not smoke measurements or a three-seed aggregate.

| Arm | FID | Recall | Latent sliced-W1 | Final logged angular loss |
| --- | ---: | ---: | ---: | ---: |
| A: original RAFM-Ang | 147.480651 | 0.697325 | 0.0470164 | 1.790236 |
| B: unit input plus radius | 324.626767 | 0 | 0.1515264 | 2.419777 |
| C: unit input, constant radius condition | 182.246091 | 0.647898 | 0.0549374 | 1.846691 |

All three independent sample audits passed. Checkpoint, sample, result and audit
hashes match their receipts; the 19 implementation-source pins match the audited
source bytes. Configurations, dataset/split hashes, centering, radius statistics,
and the original 3,925-image reference digest agree across arms.

Each model completed 40,000 updates at batch64 with AdamW LR0.0001, EMA0.9999,
bfloat16 training and float32 evaluation. Each evaluation used 3,000 balanced
samples, sampling seed0, CFG1 and 100 network calls per trajectory (600 total
across six generated batches). Decoded PNG directories are unique and contain
3,000 files each. B and C both have 32,522,176 parameters, 6,560 more than A.
No extra sampling projection or renormalization was introduced.

B's poor finite result is retained. No configuration/evaluation mix-up was found.
Its logged loss reached about 1.863 at update9,000, was about 1.965 at update10,000,
then rose to about 2.415 by update15,000 and stayed near2.4 through the final budget.
The cause is not established. No tuning, early stopping, checkpoint substitution
or rerun was performed in response to this result. Three-seed interpretation
must wait for the remaining evaluations.

The authoritative method identity is the top-level `method: rafm_ang_input`
and `arm` in each result. The reused evaluator's nested `method: tflow` label
is a metadata artifact of the shared helper; these checkpoints are not t-Flow
models, and these values must not be reported as t-Flow FIDs.

Raw results, training logs and sample-audit receipts:
`outputs_rafm_input_study/v1/final/imagenette_dcae/{A,B,C}/seed_8925/`.
The verified reference overlap remains unchanged and is documented in
`docs/recovered_artifacts/README.md`; this reference is not held out from training.
