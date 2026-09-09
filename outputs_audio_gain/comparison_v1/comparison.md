# AudioMNIST empirical-gain comparison: baseline_mismatch

Complete measured seeds: 3/3. Source status: `baseline_mismatch`.

Historical rows are preserved. New means use full-precision per-seed values and population standard deviations.

**Baseline mismatch:** the newly evaluated fixed-spherical baseline does not reproduce the archived rounded metrics. These measured rows are discrepancy-labelled and do not establish exact historical reproduction. The current paired before/after measurements remain directly comparable.

| Metric | Current fixed spherical | Current fixed spherical + gain |
|---|---:|---:|
| PIT | 0.592333 ± 0 | 0.492487 ± 7.85674e-08 |
| cov<q10 | 0 ± 0 | 0.0975 ± 0 |
| cov>q90 | 0 ± 0 | 0.1055 ± 0 |
| cov>q95 | 0 ± 0 | 0.05 ± 0 |
| cov>q99 | 0 ± 0 | 0.0145 ± 0 |
| data_energy_mean | 2.29787 ± 0 | 2.29787 ± 0 |
| digit_acc | 0.806667 ± 0.00735225 | 0.806667 ± 0.00735225 |
| energy_KS | 0.592333 ± 0 | 0.0218333 ± 0 |
| gen_energy_mean | 2.26141 ± 1.02953e-05 | 2.26982 ± 0 |
| mean_confidence | 0.82963 ± 0.0049476 | 0.82963 ± 0.00494757 |
| radial_w1 | 1.4876 ± 9.5047e-06 | 0.0487116 ± 3.93322e-09 |

| Seed | Status | Original accuracy | Gain accuracy | Changed predictions | Generation seconds | Paired evaluation seconds |
|---|---|---:|---:|---:|---:|---:|
| 8925 | complete | 0.8025 | 0.8025 | 0 | 916.9606634639204 | 6.913437947630882 |
| 1234 | complete | 0.8005 | 0.8005 | 0 | 869.8112407065928 | 0.08373824506998062 |
| 7 | complete | 0.817 | 0.817 | 0 | 870.1515833511949 | 0.08306964859366417 |

| Seed | Metric | Original full precision | Gain full precision |
|---|---|---:|---:|
| 8925 | PIT | 0.5923333333333335 | 0.4924866666666667 |
| 8925 | cov<q10 | 0.0 | 0.0975 |
| 8925 | cov>q90 | 0.0 | 0.1055 |
| 8925 | cov>q95 | 0.0 | 0.05 |
| 8925 | cov>q99 | 0.0 | 0.0145 |
| 8925 | data_energy_mean | 2.297874689102173 | 2.297874689102173 |
| 8925 | digit_acc | 0.8025 | 0.8025 |
| 8925 | energy_KS | 0.5923333333333334 | 0.021833333333333333 |
| 8925 | gen_energy_mean | 2.2614266872406006 | 2.269822359085083 |
| 8925 | mean_confidence | 0.8281312584877014 | 0.8281312584877014 |
| 8925 | radial_w1 | 1.4875863045851387 | 0.04871164358158898 |
| 1234 | PIT | 0.5923333333333335 | 0.4924865 |
| 1234 | cov<q10 | 0.0 | 0.0975 |
| 1234 | cov>q90 | 0.0 | 0.1055 |
| 1234 | cov>q95 | 0.0 | 0.05 |
| 1234 | cov>q99 | 0.0 | 0.0145 |
| 1234 | data_energy_mean | 2.297874689102173 | 2.297874689102173 |
| 1234 | digit_acc | 0.8005 | 0.8005 |
| 1234 | energy_KS | 0.5923333333333334 | 0.021833333333333333 |
| 1234 | gen_energy_mean | 2.261404275894165 | 2.269822359085083 |
| 1234 | mean_confidence | 0.8244608640670776 | 0.8244608640670776 |
| 1234 | radial_w1 | 1.4876066285371778 | 0.04871163436273726 |
| 7 | PIT | 0.5923333333333335 | 0.4924866666666667 |
| 7 | cov<q10 | 0.0 | 0.0975 |
| 7 | cov>q90 | 0.0 | 0.1055 |
| 7 | cov>q95 | 0.0 | 0.05 |
| 7 | cov>q99 | 0.0 | 0.0145 |
| 7 | data_energy_mean | 2.297874689102173 | 2.297874689102173 |
| 7 | digit_acc | 0.817 | 0.817 |
| 7 | energy_KS | 0.5923333333333334 | 0.021833333333333333 |
| 7 | gen_energy_mean | 2.2614054679870605 | 2.269822359085083 |
| 7 | mean_confidence | 0.8362985849380493 | 0.8362985253334045 |
| 7 | radial_w1 | 1.487606301665306 | 0.0487116365482409 |

All changed-prediction counts verified zero: `True`.

Published fixed-spherical accuracy remains **0.810 ± 0.013**. RAFM-Vel and RAFM-Ang remain separate historical comparisons.

| Historical row | Archived accuracy mean | Archived population SD |
|---|---:|---:|
| Fixed spherical | 0.8103 | 0.0128 |
| RAFM-Vel | 0.711 | 0.0139 |
| RAFM-Ang | 0.764 | 0.0249 |

## Baseline reproduction checks

```json
[
  {
    "seed": 8925,
    "metric": "digit_acc",
    "archived": 0.805,
    "recomputed_rounded": 0.803,
    "matches": false,
    "recorded_comparison": {
      "archived": 0.805,
      "recomputed": 0.803,
      "matches": false
    }
  },
  {
    "seed": 8925,
    "metric": "energy_KS",
    "archived": 0.5923,
    "recomputed_rounded": 0.5923,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.5923,
      "recomputed": 0.5923,
      "matches": true
    }
  },
  {
    "seed": 8925,
    "metric": "cov>q95",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 8925,
    "metric": "cov>q99",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 8925,
    "metric": "cov<q10",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 8925,
    "metric": "PIT",
    "archived": 0.5923,
    "recomputed_rounded": 0.5923,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.5923,
      "recomputed": 0.5923,
      "matches": true
    }
  },
  {
    "seed": 1234,
    "metric": "digit_acc",
    "archived": 0.798,
    "recomputed_rounded": 0.8,
    "matches": false,
    "recorded_comparison": {
      "archived": 0.798,
      "recomputed": 0.8,
      "matches": false
    }
  },
  {
    "seed": 1234,
    "metric": "energy_KS",
    "archived": 0.5923,
    "recomputed_rounded": 0.5923,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.5923,
      "recomputed": 0.5923,
      "matches": true
    }
  },
  {
    "seed": 1234,
    "metric": "cov>q95",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 1234,
    "metric": "cov>q99",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 1234,
    "metric": "cov<q10",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 1234,
    "metric": "PIT",
    "archived": 0.5923,
    "recomputed_rounded": 0.5923,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.5923,
      "recomputed": 0.5923,
      "matches": true
    }
  },
  {
    "seed": 7,
    "metric": "digit_acc",
    "archived": 0.828,
    "recomputed_rounded": 0.817,
    "matches": false,
    "recorded_comparison": {
      "archived": 0.828,
      "recomputed": 0.817,
      "matches": false
    }
  },
  {
    "seed": 7,
    "metric": "energy_KS",
    "archived": 0.5923,
    "recomputed_rounded": 0.5923,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.5923,
      "recomputed": 0.5923,
      "matches": true
    }
  },
  {
    "seed": 7,
    "metric": "cov>q95",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 7,
    "metric": "cov>q99",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 7,
    "metric": "cov<q10",
    "archived": 0.0,
    "recomputed_rounded": 0.0,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.0,
      "recomputed": 0.0,
      "matches": true
    }
  },
  {
    "seed": 7,
    "metric": "PIT",
    "archived": 0.5923,
    "recomputed_rounded": 0.5923,
    "matches": true,
    "recorded_comparison": {
      "archived": 0.5923,
      "recomputed": 0.5923,
      "matches": true
    }
  }
]
```

## Issues and failure

```json
{
  "issues": [],
  "failure": null
}
```

## Hardware

```json
{
  "host": "auh7-3b-gpu-019",
  "gpu": "AMD Instinct MI210",
  "vram_bytes": 68702699520,
  "hip_version": "6.3.42131-fa1d09cbd"
}
```

## Runtime

```json
{
  "total_evaluation_wall_s": 2683.439820148051,
  "scope": "data loading, compatibility checks, three generations, paired evaluation and per-seed serialization"
}
```

## Protocol

```json
{
  "training_seeds": [
    8925,
    1234,
    7
  ],
  "checkpoint_step": 24000,
  "n_gen": 2000,
  "class_counts": [
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200
  ],
  "sample_seed": 0,
  "gain_seed": 0,
  "rk4_steps": 40,
  "model_evaluations": 160,
  "generation_batch_size": 128,
  "classifier_batch_size": 2000,
  "tangent_projection": true,
  "state_renormalization": false,
  "cfg": 1,
  "gain_source": "RAFM training split norms; RadialEmpiricalSource ecdf quantile interpolation",
  "gain_rng": "isolated CPU generator; shared across training seeds",
  "gain_pairing": "sample seed 0 original RAFM quantile draws",
  "training_count": 10200,
  "external_test_count": 3000,
  "split_seed": 0,
  "torch_version": "2.7.1+rocm6.3",
  "cuda_version": null,
  "slurm_job_id": "763355",
  "device": "cuda:0",
  "aggregation": "archived per-seed rounding, population std"
}
```

## Provenance

```json
{
  "aggregate": {
    "path": "/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/outputs_audio_gain/fixed_spherical_empirical_gain_v1_attempt2/aggregate.json",
    "sha256": "5a42958f17e830c9b1e795f8c8ae1169db10122e4b3558cb73cddfb8204d1fb7"
  },
  "reference_aggregate": {
    "path": "/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/poc_audio/stage2_3seed.json",
    "sha256": "db71009221105b750acb9a2a4e5337f382a9cfc6ef9955ed14b6c3d8e4872adc"
  },
  "summary_script": {
    "path": "/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/tools/summarize_audio_gain.py",
    "sha256": "93735044b614c562e9666ed033799c4cba2d0475020496b8888fddc1934a2018"
  },
  "failure": null,
  "paper_sha256": "97f709df2a45e4acf4ba186379c60e9c75120d95e09af782d7a18c8b56d00367",
  "paper_source": "RAFM_ICLR_2027.pdf, page 26 Table 5; page 8 Figure 2"
}
```

Per-seed full-precision metrics, checkpoint/sample fingerprints, implementation sources and paired comparisons are retained in `comparison.json`. No figure is generated by this summary script.
