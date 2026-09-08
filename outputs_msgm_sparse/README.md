# MSGM sparse tabular results

**Outcome: 20 valid seeds and one rejected result across seven datasets.** All 21 seeds reached 10,000 training steps. Student-t D32 seed 65457 generated one sample row containing NaN, which caused six nonfinite metrics. Its underlying numerical cause has not been established. The failing configuration was stopped without retrying, resampling, filtering samples, or changing settings.

| Dataset configuration | Valid seeds / attempted | Outcome |
|---|---:|---|
| toy2d | 3 / 3 | Complete |
| gaussian_d16 | 3 / 3 | Complete |
| studentt_d16 | 3 / 3 | Complete |
| studentt_d32 | 2 / 3 | Seed 65457 rejected |
| piv_d16 | 3 / 3 | Complete |
| piv_d64 | 3 / 3 | Complete |
| piv_d256 | 3 / 3 | Complete |

- [All 21 original per-seed result links and run outcome](_orchestration/run-outcome.md)
- [Raw transfer ZIP, including the rejected output](msgm_sparse_tabular_raw_results_with_failure.zip)
- [File manifest and SHA-256 hashes](raw_results_manifest.json), explicitly marked `incomplete_nonfinite`
- [Failure details](_orchestration/studentt-d32-failure.md) and [supervisor report](_orchestration/studentt_d32-validation.json)

The 21 original metrics files, training logs, manifest and ZIP are preserved byte-for-byte. The rejected metrics file retains its original `NaN` tokens; it is not strict JSON. Python's standard `json` reader accepts those tokens as floating-point NaN. This example loads every result, including the rejected one, without filtering or rewriting it (run from the repository root):

```python
import json
from pathlib import Path

base = Path("outputs_msgm_sparse")
manifest = json.loads((base / "raw_results_manifest.json").read_text())
results = {
    (item["dataset"], item["seed"]): json.loads(
        (base / item["relative_path"]).read_text()
    )
    for item in manifest["results"]
}
```

Use each manifest entry's `status` and `nonfinite_metric_keys` when interpreting results. No complete aggregate or dataset means are published. The strict collector was cancelled after the failed dependency. The manifest retains original cluster paths as provenance; its `relative_path` fields locate the committed files.

Scientific source commit: `eb80c6b5af1f3f47d35bf0f4d9ac0c83be9bce5a`. Seeds, configurations, training budgets and sampling settings were unchanged. Each configuration used one GPU for its three sequential seeds. [Protocol audit](_orchestration/source-audit.md) and [sanity report](_orchestration/sanity-report.md) provide details.

Unchanged exp1 generates synthetic data before model seeding; independent invocations may use different draws. Sparse SDE and historical dense implementations come from different codebases.
