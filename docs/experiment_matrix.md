# Paper experiment matrix for the proposed t-Flow baseline

This inventory uses the supplied **RAFM_ICLR_2027.pdf** (39 pages; SHA-256 `97f709df2a45e4acf4ba186379c60e9c75120d95e09af782d7a18c8b56d00367`) and source at `5b89ed5f4af8a47c3b57eb9d595203daafc6d2c4`, including the paper-cited historical commits. All reported values are final; colors and temporary experiment comments do not remove rows or authorize replacements. Resource access and static implementation are authorized. **No experiment is authorized or launched.**

The machine-readable inventory is [suite_manifest.json](../configs/tflow/suite_manifest.json). It is a review document, not an executable launch config. A runner must refuse unresolved protocols and require pinned cached inputs. Dataset regeneration is prohibited.

## Coverage and training budget

**28 unique data conditions** cover eight main-paper conditions and 20 additional appendix conditions. Three final seeds per condition imply **84 complete training runs** before tuning or diagnostics. The overlapping d 16/df3 sweep point is counted once; d 16/df3 and d 32/df3 also appear in the main table. Exact archived inputs may differ between those historical contexts, so condition deduplication does not assert identical cached data.

The selected **proposal, still awaiting experiment approval**, tunes `nu ∈ {3,5,7}` crossed with source-scale multipliers `{0.5,1,2}`. For each nu, the reference scale is `median(train_norm) / sqrt(d * scipy.stats.f.ppf(0.5,d,nu))`; multiplying it by the candidate multiplier makes the middle candidate match the empirical median radius. The independent tuning seed is `46021`. All nine settings train to 5% budget, then the best two continue to 10% total budget: 500→1,000 vector steps, 1,200→2,400 audio steps, and 2,000→4,000 image steps. Selection uses validation CDF-based metrics and fixed shared sliced projections; test data never select a setting.

Tuning costs `9×0.05 + 2×(0.10−0.05) = 0.55` full-run equivalents per condition, or **15.4 equivalents** across 28 conditions. The winner is frozen and all three final full-budget seeds restart from scratch without early stopping: **84 final runs +15.4 tuning equivalents =99.4 total equivalents**. This is not a GPU-hour estimate: image, audio and vector budgets differ. Sampling, evaluation, source-scale fitting, optional diagnostics, failed runs and separate historical contexts with different cached inputs are excluded.

Vector model seeds are `8925,77395,65457`; downstream seeds are `8925,1234,7`. Vector training is 10,000 Adam steps (lr 0.001, width 128, three Swish hidden layers, scalar time concatenation, no EMA). All vector sampling uses 128 RK4 steps =512 model evaluations. Audio uses 24,000 steps, batch 32, UNet base 96, AdamW lr 0.0002, EMA 0.999 and 160 evaluations. Image uses 40,000 steps, batch 64, SiT 384/12/6, AdamW lr 0.0001, EMA 0.9999 and 100 evaluations. Full details and individual numeric fields are in the manifest.

## Required unique conditions

| ID | Paper scope / references | Batch | Split / generated count | Input and protocol status |
|---|---|---:|---|---|
| `student_t_d2_df3.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d8_df3.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d16_df3.0_cor` | main + appendix: Table 1, Table 3, Table 11, Figure 3, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d32_df3.0_cor` | main + appendix: Table 1, Table 3, Table 11, Figure 3, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d64_df3.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d128_df3.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d256_df3.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d16_df1.5_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d16_df2.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d16_df5.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d16_df10.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `student_t_d16_df50.0_cor` | appendix: Figure 3, B.1, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | synthetic_cache_missing; vector_batch_conflict |
| `gaussian_aniso_d16_cor` | appendix: Table 13, Table 14, C.4-C.9 | 4096 | random permutation ; generate 10,000 | synthetic_cache_missing |
| `aniso_k1` | appendix: Figure 3, B.1, C.5-C.9 | 4096 | contiguous iid order ; generate 10,000 | aniso_cache_missing |
| `aniso_k3` | appendix: Figure 3, B.1, C.5-C.9 | 4096 | contiguous iid order ; generate 10,000 | aniso_cache_missing |
| `aniso_k10` | appendix: Figure 3, B.1, C.5-C.9 | 4096 | contiguous iid order ; generate 10,000 | aniso_cache_missing |
| `aniso_k30` | appendix: Figure 3, B.1, C.5-C.9 | 4096 | contiguous iid order ; generate 10,000 | aniso_cache_missing |
| `aniso_k100` | appendix: Figure 3, B.1, C.5-C.9 | 4096 | contiguous iid order ; generate 10,000 | aniso_cache_missing |
| `aniso_k300` | appendix: Figure 3, B.1, C.5-C.9 | 4096 | contiguous iid order ; generate 10,000 | aniso_cache_missing |
| `toy_radial_angular` | appendix: Table 12, Table 14, Figure 5, C.5-C.9 | 4096 | random permutation ; generate 10,000 | synthetic_cache_missing |
| `piv_d16` | appendix: Table 13, Table 14, C.5-C.9 | unresolved (paper 256) | random permutation ; generate 10,000 | vector_batch_conflict |
| `piv_d32` | appendix: C.5 | unresolved (paper 256) | random permutation ; generate 10,000 | vector_batch_conflict; piv32_identity_missing |
| `piv_d64` | main + appendix: Table 1, Table 3, Table 14, C.5-C.9 | 256 | random permutation; generate 10,000 | resolved from completion JSON and main-table per-seed output; split-hash check pending |
| `piv_d256` | main + appendix: Table 1, Table 3, Table 14, C.5-C.9 | 256 | random permutation; generate 10,000 | resolved from completion JSON and main-table per-seed output; split-hash check pending |
| `finance_ff49` | main + appendix: Table 1, Table 7, Table 10, Table 15, C.5-C.9 | 4096 | chronological ; generate 2,871 | cached input pinned; static runner review pending |
| `weather_au_wind` | main + appendix: Table 1, Table 8, Table 15, Figure 5, C.5-C.9 | 2048 | chronological ; generate 1,170 | cached input pinned; static runner review pending |
| `audiomnist_stft` | main + appendix: Figure 2, Figure 4, Table 5, Table 6, C.10 | 32 | nested fixed random ; generate 2,000 | cached input pinned; static runner review pending |
| `imagenette_dcae` | main + appendix: Table 2, Table 4, Table 9, Figure 6, Figure 7, C.10 | 64 | unresolved ; generate 3,000 | image_split_conflict; historical_eval_provenance |

All Student-t conditions contain 50,000 examples and have 30,000/10,000/10,000 random splits(seed 0), with mixing matrix seed 42. Anisotropy has 50,000 raw Gaussian samples per kappa and contiguous 60/20/20 splits. PIV has 998 snapshots with 598/199/201 random splits(seed 0). Finance has8,609/2,869/2,871 chronological examples; Weather has3,508/1,169/1,170. Audio uses 10,200 of 12,000 nominal training clips,1,800 internal validation clips and 3,000 external test clips. The exact image split remains unresolved.

## Cached inputs

Input files were first hashed as bytes. Subsequent authorized CPU-only verification loaded the five resolved cached datasets, checked finite values and split counts, and exactly matched the archived PIV64/PIV256 train/test tensor hashes. No model checkpoint was loaded; see `artifact_audit/cached_input_checks.json`. Existing caches stay in place; any newly stored dataset must use `/mnt/vast01/users/fouad.oubari/data/`. The manifest contains absolute paths and hashes.

| Input | Shape / count | Location |
|---|---|---|
| `piv16` | [998, 16] | `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/data/piv/piv_d16.pt` |
| `piv64` | [998, 64] | `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/data/piv/piv_d64.pt` |
| `piv256` | [998, 256] | `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/data/piv/piv_d256.pt` |
| `finance` | [14349, 49] | `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/rebuttal_experiments/data/finance/finance_returns_raw.pt` |
| `weather` | [5847, 96] | `/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/rebuttal_experiments/data/weather/weather_au_wind.pt` |
| `audio_train` | [12000, 2, 129, 63] | `/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/poc_audio/data/audiomnist_stft_train.pt` |
| `audio_test` | [3000, 2, 129, 63] | `/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/poc_audio/data/audiomnist_stft_test.pt` |
| `audio_classifier` | metric_checkpoint; do not retrain | `/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/poc_audio/digit_classifier.pt` |
| `dcae_latents` | [13394, 2048] | `/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/image_latents/data/dcae_latents_scaled.pt` |
| `dcae_labels` | [13394] | `/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/image_latents/data/dcae_labels.pt` |
| `dcae_reference` | 3925 | `/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/data/experiments/image_latents/dit/eval_std/_real_ref` |

Audio preprocessing is fixed:8kHz one-second audio, complex Hann STFT(FFT 256, hop 128),2x129x 63 real/imag channels, unit-norm direction multiplied by an independent gain mixture(lognormal modes1/4, weights.6/.4, log-SD.25, lower clip.1). No centering is applied. The classifier is a fixed metric artifact and must not be retrained.

Image preprocessing uses the frozen `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers` autoencoder, Resize 256/CenterCrop 256,32x8x8 latents, scale 0.41407 and generator-training mean centering. Cached latents do not supply verified row-to-image/split identity. The 3925 reference PNGs need a pinned per-file inventory and disjointness verification.

## Unresolved discrepancies and execution blockers

- **`synthetic_cache_missing` (blocking).** Exact realized Student-t, Gaussian control and Toy2D tensors are not present in the audited input locations. Their constructors generate new values before the experiment runner sets the model seed. Reconstructing from matrix/split seeds does not recover historical data. Resolution: Obtain the exact archived data, splits and mixing matrices for each reported comparison; do not regenerate. Evidence: `rafm/data/student_t.py`, `rafm/data/gaussian_aniso.py`, `rafm/data/toy_radial_angular.py`, `experiments/exp1_main_benchmark.py`, `paper C.5-C.8`.
- **`aniso_cache_missing` (blocking).** All six expected aniso_gauss_d32_k{k}.pt inputs are missing; only meta.json is retained. Resolution: Obtain original cached files; generator seed 42 is not authorization to regenerate. Evidence: `rebuttal_experiments/scripts/gen_aniso.py`, `rebuttal_experiments/data/aniso/meta.json`, `paper B.1/C.5`.
- **`vector_batch_conflict` (blocking).** Paper B.1/C.7 specifies batch 256 for Student-t and PIV. Commit2e659c7 defaults are 4096; f 892493 defaults are 256 but E1/E6/E8/E 10 YAMLs remain 4096. The specific archived comparison configuration is unresolved. Resolution: Identify exact cached-data/config/result grouping for the reported values. Do not silently substitute a default; paper target is 256. Evidence: `2e659c7:configs/defaults.yaml`, `f892493:configs/defaults.yaml`, `f892493:rebuttal_experiments/configs/E6_dim16.yaml`, `paper C.1/C.7`. Later PIV d64/d256 completion metadata resolves both main comparisons to batch 256; this blocker no longer applies to those two conditions.
- **`piv32_identity_missing` (blocking).** Paper C.5 lists native8x4 PIV d 32; piv_d 32.pt is absent. d 16 backing storage is not verified provenance for this input. Resolution: Obtain verified native d 32 cached file or authoritative proof of storage identity before materializing any view. Evidence: `paper C.5, p33`, `data/piv/piv_d16.pt`, `configs/exp1/piv_d32.yaml`.
- **`piv64_protocol_conflict` (resolved archive selection).** E10 is a distinct older context. The later `outputs/exp1_main_benchmark/piv_d64/angular_rafm/` records and `rebuttal_experiments/RAFM_ANG_TABLE1_completion.json` match the paper Table 1/3 values and give batch 256, random split seed 0, 598/199/201 counts, 10,000 generated samples and 512 evaluations. Use these records for the main comparison. The same completion artifact resolves PIV d256 and stores train/test tensor MD5 values for a later authorized identity check.
- **`image_split_conflict` (blocking).** Paper C.10 says the 3925 official-validation reference images are disjoint from generator training and all FM methods use 3000 generated images. Local extraction concatenates official train+val 13394 images and trainer random-splits all rows 60/20/20. It saves no row-to-image split manifest. Angular launcher requests 5000. Resolution: Obtain exact paper training/validation/test row indices and image mapping, verify reference disjointness, pin 3000 FM evaluation count. Do not silently resplit. Evidence: `experiments/image_latents/scripts/extract_dcae_latents.py`, `experiments/image_latents/dit/dit_train_sit.py`, `experiments/image_latents/launch_angular_dit.bat`, `experiments/image_latents/dit/eval_phaseB.bat`, `paper C.10`.
- **`historical_eval_provenance` (blocking_comparison_alignment).** Original downstream per-seed eval JSONs are absent in the audited tree; aggregated JSONs survive. Published vector values also differ from retained generated table TeX. Resolution: Recover cited-commit per-seed records or preserve paper values as immutable reported references with explicit unmatched-input provenance. Evidence: `experiments/image_latents/results_phaseB.json`, `experiments/poc_audio/stage2_3seed.json`, `rebuttal_experiments/paper_assets/tables/table_realdata.tex`, `paper Tables1-8`.
- **`projection_identity_unpinned` (comparison_limitation).** Paper C.12 calls for shared projection directions within each comparison but no fixed projection seed; current evaluators draw from method-specific post-training RNG state. Exact old projection arrays/RNG state are absent. Resolution: Use documented common new projection arrays across newly evaluated methods; do not claim bit-identical historical projections. Evidence: `paper C.12`, `rafm/metrics/distributional.py`, `experiments/exp1_main_benchmark.py`.
- **`stale_nfe_config` (resolved_interpretation).** Some angular run configs say nfe 512 steps, but final metrics record 512 model evaluations; resample_matched_nfe.py resampled at 128 RK4 steps without changing YAML. Resolution: Use 128 RK4 steps/512 actual evaluations; retain original config and resampling provenance. Evidence: `rebuttal_experiments/scripts/resample_matched_nfe.py`, `rebuttal_experiments/raw_results/E9_aniso/aniso_k1/angular_rafm/seed_8925/metrics.json`, `paper C.9`.

## Published diagnostics and reuse of final checkpoints

| Artifact | Data conditions / protocol | Treatment |
|---|---|---|
| Figure 5, B.2, B.5 | student_t_d16_df3.0_cor, piv_d 64, weather_au_wind, toy_radial_angular; sampling_only_from_fixed_final_checkpoint | Paper panel uses RK4+projection. Source broader grid also retains Euler/Heun and projection on/off; not additional trained benchmarks. Preserve toy angular missing curve/failure. |
| Table 14 | gaussian_aniso_d16_cor, piv_d 16, piv_d 64, piv_d 256, student_t_d16_df3.0_cor, student_t_d32_df3.0_cor, toy_radial_angular; sampling_only;existing RAFM-Vel with/without tangent projection | Applies to spherical RAFM; do not change t-Flow into a spherical projected model to imitate this ablation. Add t-Flow baseline values only where scientifically meaningful. |
| Table 11 | student_t_d16_df3.0_cor, student_t_d32_df3.0_cor; existing empirical/oracle comparison | No new dataset; t-Flow baseline reuses corresponding condition. Historical oracle/empirical rows preserved. |
| Figure 4, B.2 | student_t_d16_df3.0_cor, audiomnist_stft; geometry diagnostics | Current make_audio_fig.py samples 20000 pairs; paper says 40000. Existing source can regenerate synthetic inputs. No such execution before approval or cache resolution. |
| Table 6 | audiomnist_stft; existing fixed-classifier control | Preserve gain-invariance and radius-replacement diagnostics; classifier weights fixed, no retraining. |
| Table 9, Figure 6, Figure 7 | imagenette_dcae; existing class/radius/style diagnostics and paired decoded grids | No additional trained benchmark; dataset identity and disjointness blocker applies to any new evaluation. |
| Table 10 | finance_ff 49; train-test radial-shift diagnostic | Same chronological Finance input; not a random-split experiment. |
| Tables 15, Table 16, Table 17, Table 18, Table 19, C.13 | ; runtime profiling of existing data conditions | No new accuracy dataset; historical timing hardware/precision/batches differ. Training-equivalent estimate above excludes timing bursts and evaluation cost. |

Figure 5 has actual RK4 NFEs `{4,8,20,48,100}`. It reuses one final checkpoint (seed 8925) and sampling seed 12345 with 10,000 generated samples. The original CSV source also includes Euler, Heun and projection on/off; those are separate optional evaluation coverage, not extra training conditions. A t-Flow baseline must retain its own Euclidean ODE and should not be modified into a projected spherical model to reproduce a RAFM-specific ablation.

The full vector metric set includes radial W1/KS,500-projection sliced W1, RBF MMD²,200-projection angular SW over four test-radius quantile bins, tail quantile relative errors(q.95/.99/.995), tail exceedance errors(q.95/.99), invalid-sample rates and training/sampling time. Audio adds frozen-classifier digit accuracy and gain calibration. Image uses torch-fidelity FID/KID and PRDC(k 5) on the same Inception feature space, plus latent calibration. The manifest preserves all method-specific count/NFE differences from the final sparse references.

## Repository-only variants kept outside paper scope

- **`finance_random_split`:** Repository-only diagnostic; not a reported paper benchmark. Paper Table 10 is chronological radial shift.
- **`weather_resmlp`:** Repository-only larger-backbone variant; paper Table 8 explicitly uses main three-hidden-layer MLP.
- **`gaussian_aniso_d32`:** Config exists but no reported paper result identified.
- **`sample_efficiency`:** No sample-efficiency result is reported in the supplied 39-page paper. Repository-only config remains explicit rather than silently omitted.
- **`solver_extended`:** Paper Figure 5 plots RK4 drift only; retain broader existing Euler/Heun/RK4 CSVs without assuming new t-Flow sweep authorization.
- **`training_trajectories`:** Archived downstream intermediates exist, but headline paper comparisons use final checkpoint. No extra training runs required.

In particular, the supplied paper contains **no sample-efficiency result**. The repository config `configs/exp2/sample_efficiency.yaml` proposes Student-t d 16/df3 with a 100,000-sample pool and training sizes 500/1,000/5,000/20,000/50,000; it is explicitly retained as optional and requires an exact cached pool. Finance random-split and Weather ResMLP are not substituted for the chronological main-paper protocols.

## Table and figure provenance

Original per-seed vector results live under `outputs/exp1_main_benchmark/` and `rebuttal_experiments/raw_results/`. Retained aggregation inputs are `master_suite.json`, `master_results.json`, audio `stage2_3seed.json` and image `results_phaseB.json`. Their numbers must not be assumed to equal the supplied PDF.

Aggregation uses `build_full_suite_report.py`, `build_master_aggregation.py`, audio `stage2_aggregate.py` and image `dit/aggregate_results.py`. Table generation is `rebuttal_experiments/paper_assets/tables/make_tables.py`; figure generation is `paper_assets/figs/make_figures.py`, `make_audio_fig.py`, `theory_diagnostics.py` and `experiments/image_latents/dit/make_dcae_grid.py`. Some figure scripts sample new geometry or construct datasets; they are not all static renderers. Preserve every existing artifact and create separate t-Flow outputs only after experiment approval.

Static checks for this inventory verify valid JSON,28 distinct condition IDs, three seeds each,84 final runs, complete issue/input references and available pinned-input paths. No ML imports, data deserialization, checkpoint loading, generation, training, sampling or scheduler submission occurred.

## Historical metric field mapping

Vector records use `radial_w1`, `ks_stat`, `sliced_w1`, `mmd`, `q950_err/q990_err/q995_err`, `tail_exc_95/tail_exc_99`, `angular_sw_bin0..3`, `angular_sw_mean`, `nan_rate`, `exploding_norm_rate`, `invalid_rate`, `nfe`, `sample_time_s` and `total_train_time_s`. Audio native records nest content under `content` and calibration under `energy`; their aggregate renames `energy.ks` to `energy_KS` and coverage keys to `cov>q95/cov>q99`. Image native records nest FID/KID/PRDC under `image` and radial/KS/sliced values under `latent`; the aggregate flattens these fields under `40000/<method>`. Full aliases and method names are in the manifest. Population standard deviations preserve archived failure handling and per-seed rounding.

The paper-matching PIV reference is the completion JSON plus the main-table output directories, not the E10 aggregate. Its Sliced W1 values are PIV d64 RAFM-Vel `0.02728±0.00290` and RAFM-Ang `0.02790±0.00249`; d256 RAFM-Vel `0.02418±0.00301` and RAFM-Ang `0.02268±0.00032`. Current chronological Finance and Weather per-seed files match the paper main Sliced W1 rows; their exact artifact mappings and aggregate metrics are retained in the manifest. Student-t E1 baseline records do not match the main paper values, although its angular rows match; they must not replace the original reported rows.

Rendered PDF pages 23, 31 and 35 were visually checked: complete sweep scope, control/projection rows and downstream training/reference requirements agree with the transcribed inventory. The prepared t-Flow numerical convention uses Heun with half as many intervals as actual neural evaluations, positive start `t_min=0.001`, `power_sigma` grid, rho 7 and `sigma_min=0.01`; this is distinct from the historical RAFM RK4 protocol.

## Completed lightweight verification

The five resolved cached inputs pass shape, split and finite-value checks. PIV64 and PIV256 match both archived split tensor hashes exactly. All3925 FID PNGs match the committed per-file SHA256 manifest; this establishes reference identity, not generator/reference disjointness. The frozen DC-AE decoder and Inception files are now also hashed in `configs/tflow/image_resources.json`.

MMD comparison limitation: C.12 describes a common kernel bandwidth, but the archived implementation chooses its median bandwidth separately from each method’s generated/test concatenation. These bandwidths were not archived. The new adapter preserves the existing evaluator; a claim of identical historical kernel bandwidth would be unsupported.
