# Student-t D32 failure

Job **758572**, config `studentt_d32`, seed **65457**, failed independent result validation after exactly 10,000 training steps. Training took 765.8992345333099 seconds; the final logged loss was finite at -9.327256202697754.

The unchanged benchmark output [metrics.json](../exp1_main_benchmark/student_t_d32_df3.0_cor/msgm_sparse/seed_65457/metrics.json) contains NaN for `radial_w1`, `q950_err`, `q990_err`, `q995_err`, `sliced_w1` and `mmd`. It reports `nan_rate` and `invalid_rate` of 9.999999747378752e-05 (approximately one in 10,000 generated samples), and an exploding-norm rate of zero. This is the recorded metric output; no sample tensor recomputation has been performed on the login node.

The external supervisor rejected the result and terminated only this configuration's child process group (child return code -15, Slurm exit code 1:0). The [validation report](studentt_d32-validation.json) preserves successful checks for seeds 8925 and 77395, and records the failure. No retry, seed replacement, resampling, parameter change or source change has been made.

The dependent all-results CPU collection job **758576** was automatically cancelled because its `afterok` dependency failed. A complete 21-valid-result manifest or ZIP must not be published. The other configurations continue under the user's instruction to stop only the failing configuration. All original artifacts remain in place.
