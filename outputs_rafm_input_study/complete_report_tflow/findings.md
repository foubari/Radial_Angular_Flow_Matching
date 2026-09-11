# Measured study findings

Scope: **partial_suite**; 19/28 conditions have all A/B/C/t-Flow methods complete.

Final seed status counts: {"complete": 310, "failed": 26}.

New synthetic realizations are new shared comparisons; published historical results remain unchanged.

**1.** Normalized input plus radius conditioning (B) versus original RAFM-Ang (A): among 26/28 eligible conditions, the primary metric improves in 10, worsens in 16, and ties in 0. All three seeds favor B in 7 conditions. These are condition-specific descriptive outcomes; individual effects follow below.

**2.** Explicit radius conditioning (B) versus identical zero-conditioned modules (C): among 27/28 eligible conditions, the primary metric improves in 11, worsens in 16, and ties in 0. All three seeds favor B in 7 conditions. These are condition-specific descriptive outcomes; individual effects follow below.

**3.** B-A: 7 conditions improve in every seed, 8 worsen in every seed, 0 tie in every seed, and 11 have mixed/tied seed outcomes; B-C: 7 conditions improve in every seed, 2 worsen in every seed, 0 tie in every seed, and 18 have mixed/tied seed outcomes. Coverage and individual deltas, including negative results, are reported without a cross-dataset pooled score.

**4.** Recorded hardware/precision permit descriptive timing ratios for 26 B/A and 27 B/C condition pairs. The parameter table reports absolute counts and the recorded conditioning overhead; B/C receive no additional tuning. t-Flow validation-only training and sampling seconds are separately reconstructed only from complete, consistent trial records.

## Paired primary effects

| Condition | Contrast | Metric (better) | Left mean ± SD | Right mean ± SD | Mean delta | Three seed deltas | Mean outcome / consistency |
|---|---|---|---|---|---|---|---|
| aniso_k1 | B-A | sliced_w1 (lower) | 0.02116546 ± 0.0003722794 | 0.02114721 ± 0.000228598 | 1.824213e-05 | -5.001202e-06, -0.000232216, 0.0002919436 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-A | sliced_w1 (lower) | 0.1540512 ± 0.03351665 | 0.1343387 ± 0.01337867 | 0.01971249 | 0.0563723, -0.01371326, 0.01647843 | worsened / mixed_or_tied_seeds |
| aniso_k100 | B-A | sliced_w1 (lower) | 1.201482 ± 0.1476257 | 1.476195 ± 0.2055364 | -0.2747125 | -0.2472599, -0.4574734, -0.1194041 | improved / improved_all_three |
| aniso_k3 | B-A | sliced_w1 (lower) | 0.04905627 ± 0.003722607 | 0.04692881 ± 0.001869119 | 0.002127457 | 0.0007687137, 0.0003211163, 0.005292542 | worsened / worsened_all_three |
| aniso_k30 | B-A | sliced_w1 (lower) | 0.3004518 ± 0.01201033 | 0.34036 ± 0.03601455 | -0.0399082 | -0.03195348, -0.0917176, 0.003946483 | improved / mixed_or_tied_seeds |
| aniso_k300 | B-A | sliced_w1 (lower) | 2.511904 ± 0.3429369 | 6.040783 ± 0.5216606 | -3.528878 | -3.864776, -3.704528, -3.01733 | improved / improved_all_three |
| audiomnist_stft | B-A | digit_acc (higher) | 0.7765 ± 0.01349691 | 0.7883333 ± 0.01133088 | -0.01183333 | -0.021, 0.0175, -0.032 | worsened / mixed_or_tied_seeds |
| finance_ff49 | B-A | sliced_w1 (lower) | 0.1819208 ± 0.002771369 | 0.1810953 ± 0.003159868 | 0.0008254647 | 0.003978595, -0.00157328, 7.107854e-05 | worsened / mixed_or_tied_seeds |
| gaussian_aniso_d16_cor | B-A | sliced_w1 (lower) | 0.1398223 ± 0.01980896 | 0.1223976 ± 0.01401081 | 0.01742475 | 0.03620982, -0.01249727, 0.0285617 | worsened / mixed_or_tied_seeds |
| imagenette_dcae | B-A | fid (lower) | 311.4028 ± 11.05801 | 147.9571 ± 0.3779735 | 163.4457 | 177.1461, 149.5761, 163.615 | worsened / worsened_all_three |
| piv_d16 | B-A | sliced_w1 (lower) | 0.01262365 ± 0.001102267 | 0.01202496 ± 0.00127226 | 0.0005986958 | 0.0005266815, 0.0008443883, 0.0004250174 | worsened / worsened_all_three |
| piv_d256 | B-A | sliced_w1 (lower) | 0.02435045 ± 0.001534214 | 0.02328675 ± 0.001292096 | 0.001063695 | 0.001744386, 0.001115985, 0.0003307126 | worsened / worsened_all_three |
| piv_d32 | B-A | sliced_w1 (lower) | 0.02108983 ± 0.003004296 | 0.02023155 ± 0.00157499 | 0.0008582833 | 0.00406542, -0.001218574, -0.0002719965 | worsened / mixed_or_tied_seeds |
| piv_d64 | B-A | sliced_w1 (lower) | 0.02908254 ± 0.003976575 | 0.02878804 ± 0.001971511 | 0.0002944979 | -0.002449688, 0.0005232319, 0.002809949 | worsened / mixed_or_tied_seeds |
| student_t_d128_df3.0_cor | B-A | sliced_w1 (lower) | 1.11209 ± 0.08964857 | 1.509971 ± 0.08400867 | -0.3978816 | -0.6300576, -0.3262579, -0.2373291 | improved / improved_all_three |
| student_t_d16_df1.5_cor | B-A | sliced_w1 (lower) | 1.865049 ± 0.02615053 | 2.117123 ± 0.1213077 | -0.2520739 | -0.1692873, -0.3871874, -0.1997471 | improved / improved_all_three |
| student_t_d16_df10.0_cor | B-A | sliced_w1 (lower) | 0.1851739 ± 0.005529706 | 0.1583266 ± 0.0152329 | 0.02684729 | 0.04198155, 0.02155584, 0.01700447 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | B-A | sliced_w1 (lower) | 0.5756264 ± 0.05167456 | 0.6273965 ± 0.01822375 | -0.05177011 | -0.03136802, -0.1003803, -0.02356201 | improved / improved_all_three |
| student_t_d16_df3.0_cor | B-A | sliced_w1 (lower) | 0.3234738 ± 0.035526 | 0.3104115 ± 0.03131256 | 0.01306228 | 0.003237575, 0.01505736, 0.0208919 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | B-A | sliced_w1 (lower) | 0.2131481 ± 0.03137122 | 0.2044502 ± 0.01999686 | 0.008697887 | -0.003341436, -0.01210567, 0.04154077 | worsened / mixed_or_tied_seeds |
| student_t_d16_df50.0_cor | B-A | sliced_w1 (lower) | 0.2007055 ± 0.0492691 | 0.1810999 ± 0.04547245 | 0.01960562 | 0.002621114, 0.01778673, 0.03840901 | worsened / worsened_all_three |
| student_t_d256_df3.0_cor | B-A | sliced_w1 (lower) | 1.211847 ± 0.05100692 | 1.764839 ± 0.07956659 | -0.5529917 | -0.5209283, -0.6085857, -0.5294611 | improved / improved_all_three |
| student_t_d32_df3.0_cor | B-A | sliced_w1 (lower) | 0.4184649 ± 0.01936759 | 0.4689223 ± 0.03662707 | -0.05045741 | -0.09470353, -0.01228169, -0.04438701 | improved / improved_all_three |
| student_t_d64_df3.0_cor | B-A | sliced_w1 (lower) | 0.7339074 ± 0.1158656 | 0.91761 ± 0.06660982 | -0.1837026 | 0.04013699, -0.2016504, -0.3895945 | improved / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-A | sliced_w1 (lower) | 0.1829337 ± 0.01304803 | 0.1977029 ± 0.02692466 | -0.01476921 | 0.002658024, -0.01450199, -0.03246365 | improved / mixed_or_tied_seeds |
| weather_au_wind | B-A | sliced_w1 (lower) | 0.0819225 ± 0.002815103 | 0.0766653 ± 0.0007494695 | 0.005257197 | 0.002613932, 0.005474441, 0.007683218 | worsened / worsened_all_three |
| aniso_k1 | B-C | sliced_w1 (lower) | 0.02116546 ± 0.0003722794 | 0.02115214 ± 0.0003762271 | 1.331978e-05 | 0.0002083369, -0.0003536679, 0.0001852904 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-C | sliced_w1 (lower) | 0.1540512 ± 0.03351665 | 0.1344633 ± 0.009847783 | 0.0195879 | 0.05476086, -0.005414158, 0.009417005 | worsened / mixed_or_tied_seeds |
| aniso_k100 | B-C | sliced_w1 (lower) | 1.201482 ± 0.1476257 | 1.258187 ± 0.09357823 | -0.05670486 | -0.1924658, 0.08683276, -0.0644815 | improved / mixed_or_tied_seeds |
| aniso_k3 | B-C | sliced_w1 (lower) | 0.04905627 ± 0.003722607 | 0.04770535 ± 0.00103814 | 0.001350913 | -3.990531e-05, -0.00173223, 0.005824875 | worsened / mixed_or_tied_seeds |
| aniso_k30 | B-C | sliced_w1 (lower) | 0.3004518 ± 0.01201033 | 0.3529205 ± 0.03106438 | -0.0524687 | -0.07936049, -0.05056337, -0.02748224 | improved / improved_all_three |
| aniso_k300 | B-C | sliced_w1 (lower) | 2.511904 ± 0.3429369 | 3.740247 ± 0.4055548 | -1.228342 | -0.7271662, -1.23703, -1.720831 | improved / improved_all_three |
| audiomnist_stft | B-C | digit_acc (higher) | 0.7765 ± 0.01349691 | 0.7868333 ± 0.008209074 | -0.01033333 | -0.004, -0.007, -0.02 | worsened / worsened_all_three |
| finance_ff49 | B-C | sliced_w1 (lower) | 0.1819208 ± 0.002771369 | 0.1929223 ± 0.003231769 | -0.01100154 | -0.0135159, -0.009726599, -0.009762123 | improved / improved_all_three |
| gaussian_aniso_d16_cor | B-C | sliced_w1 (lower) | 0.1398223 ± 0.01980896 | 0.1405654 ± 0.02014878 | -0.0007430688 | 0.01465532, -0.01225222, -0.004632309 | improved / mixed_or_tied_seeds |
| imagenette_dcae | B-C | fid (lower) | 311.4028 ± 11.05801 | 227.4368 ± 67.72443 | 83.96605 | 142.3807, -25.6024, 135.1199 | worsened / mixed_or_tied_seeds |
| piv_d16 | B-C | sliced_w1 (lower) | 0.01262365 ± 0.001102267 | 0.01230028 ± 0.001041908 | 0.0003233685 | 0.0006008726, 7.551629e-05, 0.0002937168 | worsened / worsened_all_three |
| piv_d256 | B-C | sliced_w1 (lower) | 0.02435045 ± 0.001534214 | 0.02483614 ± 0.001801346 | -0.0004856971 | -0.0007719658, -9.07965e-05, -0.0005943291 | improved / improved_all_three |
| piv_d32 | B-C | sliced_w1 (lower) | 0.02108983 ± 0.003004296 | 0.02063411 ± 0.001252197 | 0.0004557197 | 0.002856398, -0.0001353975, -0.001353841 | worsened / mixed_or_tied_seeds |
| piv_d64 | B-C | sliced_w1 (lower) | 0.02908254 ± 0.003976575 | 0.03915604 ± 0.001802111 | -0.0100735 | -0.01352261, -0.00942062, -0.007277269 | improved / improved_all_three |
| student_t_d128_df3.0_cor | B-C | sliced_w1 (lower) | 1.11209 ± 0.08964857 | 1.108545 ± 0.09175319 | 0.003544688 | 0.00803113, -0.0008511543, 0.003454089 | worsened / mixed_or_tied_seeds |
| student_t_d16_df1.5_cor | B-C | sliced_w1 (lower) | 1.865049 ± 0.02615053 | 1.80441 ± 0.03963706 | 0.06063954 | -0.0002036095, 0.09872878, 0.08339345 | worsened / mixed_or_tied_seeds |
| student_t_d16_df10.0_cor | B-C | sliced_w1 (lower) | 0.1851739 ± 0.005529706 | 0.1846423 ± 0.004755621 | 0.0005315095 | -0.0005529821, 0.001406848, 0.0007406622 | worsened / mixed_or_tied_seeds |
| student_t_d16_df2.0_cor | B-C | sliced_w1 (lower) | 0.5756264 ± 0.05167456 | 0.570111 ± 0.05265969 | 0.005515307 | -0.008599937, 0.003964275, 0.02118158 | worsened / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-C | sliced_w1 (lower) | 0.3234738 ± 0.035526 | 0.3196277 ± 0.03851336 | 0.003846059 | -0.01597115, 0.02030534, 0.007203996 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-C | sliced_w1 (lower) | 0.2131481 ± 0.03137122 | 0.229948 ± 0.01436158 | -0.0167999 | -0.03653954, -0.01943561, 0.005575448 | improved / mixed_or_tied_seeds |
| student_t_d16_df50.0_cor | B-C | sliced_w1 (lower) | 0.2007055 ± 0.0492691 | 0.2075806 ± 0.04576648 | -0.006875058 | -0.01008257, -0.001545221, -0.008997381 | improved / improved_all_three |
| student_t_d256_df3.0_cor | B-C | sliced_w1 (lower) | 1.211847 ± 0.05100692 | 1.201759 ± 0.04246641 | 0.01008809 | -0.002339721, 0.00756681, 0.02503717 | worsened / mixed_or_tied_seeds |
| student_t_d2_df3.0_cor | B-C | sliced_w1 (lower) | 0.4357766 ± 0.04706212 | 0.4726823 ± 0.008276605 | -0.03690574 | 0.0118013, -0.1099747, -0.01254383 | improved / mixed_or_tied_seeds |
| student_t_d32_df3.0_cor | B-C | sliced_w1 (lower) | 0.4184649 ± 0.01936759 | 0.41523 ± 0.02532527 | 0.003234923 | 0.00651437, 0.008376718, -0.005186319 | worsened / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-C | sliced_w1 (lower) | 0.7339074 ± 0.1158656 | 0.7282162 ± 0.1140678 | 0.005691171 | 0.0115853, -0.006797552, 0.01228577 | worsened / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-C | sliced_w1 (lower) | 0.1829337 ± 0.01304803 | 0.2424474 ± 0.01912146 | -0.05951367 | -0.05102471, -0.06262192, -0.06489439 | improved / improved_all_three |
| weather_au_wind | B-C | sliced_w1 (lower) | 0.0819225 ± 0.002815103 | 0.08133958 ± 0.001575853 | 0.000582916 | -0.001958691, -0.001540229, 0.005247667 | worsened / mixed_or_tied_seeds |
| aniso_k1 | tflow-A | sliced_w1 (lower) | 0.5047251 ± 0.04704663 | 0.02114721 ± 0.000228598 | 0.4835778 | 0.545352, 0.430737, 0.4746446 | worsened / worsened_all_three |
| aniso_k100 | tflow-A | sliced_w1 (lower) | 13.61448 ± 0.4672442 | 1.476195 ± 0.2055364 | 12.13828 | 11.95544, 11.63795, 12.82145 | worsened / worsened_all_three |
| aniso_k3 | tflow-A | sliced_w1 (lower) | 0.5166018 ± 0.1519085 | 0.04692881 ± 0.001869119 | 0.469673 | 0.343857, 0.3831429, 0.682019 | worsened / worsened_all_three |
| aniso_k30 | tflow-A | sliced_w1 (lower) | 4.830289 ± 0.4641929 | 0.34036 ± 0.03601455 | 4.489929 | 3.828757, 4.717698, 4.923333 | worsened / worsened_all_three |
| aniso_k300 | tflow-A | sliced_w1 (lower) | 191.4659 ± 20.27319 | 6.040783 ± 0.5216606 | 185.4252 | 209.4754, 185.7921, 161.008 | worsened / worsened_all_three |
| audiomnist_stft | tflow-A | digit_acc (higher) | 0.1448333 ± 0.006908127 | 0.7883333 ± 0.01133088 | -0.6435 | -0.654, -0.638, -0.6385 | worsened / worsened_all_three |
| finance_ff49 | tflow-A | sliced_w1 (lower) | 5.951689 ± 0.3055488 | 0.1810953 ± 0.003159868 | 5.770594 | 5.585619, 5.528837, 6.197325 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | tflow-A | sliced_w1 (lower) | 1.189184 ± 0.2825031 | 0.1223976 ± 0.01401081 | 1.066787 | 1.277747, 0.6614322, 1.261182 | worsened / worsened_all_three |
| imagenette_dcae | tflow-A | fid (lower) | 223.7961 ± 6.271902 | 147.9571 ± 0.3779735 | 75.839 | 68.14389, 82.88389, 76.48923 | worsened / worsened_all_three |
| piv_d16 | tflow-A | sliced_w1 (lower) | 0.06458287 ± 0.0214412 | 0.01202496 ± 0.00127226 | 0.05255791 | 0.04258835, 0.08390061, 0.03118478 | worsened / worsened_all_three |
| piv_d32 | tflow-A | sliced_w1 (lower) | 0.8933748 ± 0.6167153 | 0.02023155 ± 0.00157499 | 0.8731432 | 0.4043236, 0.468407, 1.746699 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | tflow-A | sliced_w1 (lower) | 271.1661 ± 39.45522 | 2.117123 ± 0.1213077 | 269.049 | 324.472, 247.4668, 235.208 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | tflow-A | sliced_w1 (lower) | 2.710411 ± 0.7391886 | 0.1583266 ± 0.0152329 | 2.552085 | 3.423086, 2.642407, 1.590761 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | tflow-A | sliced_w1 (lower) | 89.95651 ± 2.729531 | 0.6273965 ± 0.01822375 | 89.32911 | 92.63408, 89.39028, 85.96298 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | tflow-A | sliced_w1 (lower) | 4.80983 ± 1.310511 | 0.3104115 ± 0.03131256 | 4.499419 | 5.390807, 5.502507, 2.604942 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | tflow-A | sliced_w1 (lower) | 3.732092 ± 1.789478 | 0.2044502 ± 0.01999686 | 3.527642 | 6.075484, 2.435917, 2.071524 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | tflow-A | sliced_w1 (lower) | 3.138102 ± 1.392299 | 0.1810999 ± 0.04547245 | 2.957002 | 4.939093, 2.217263, 1.714649 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | tflow-A | sliced_w1 (lower) | 7.954707 ± 1.641373 | 0.4689223 ± 0.03662707 | 7.485785 | 9.727603, 6.81303, 5.91672 | worsened / worsened_all_three |
| student_t_d8_df3.0_cor | tflow-A | sliced_w1 (lower) | 5.252547 ± 1.119479 | 0.1977029 ± 0.02692466 | 5.054844 | 4.72357, 3.898385, 6.542578 | worsened / worsened_all_three |

Deltas are left minus right. Positive favors the left method for digit accuracy; negative favors it for sliced W1/FID. Standard deviations are population SD. Seed order is recorded per condition in findings.json.

## Conditional angular effects

| Condition | Contrast | Angular metric | Mean delta | Three seed deltas | Mean outcome / consistency |
|---|---|---|---|---|---|
| aniso_k1 | B-A | angular_sw_mean | 7.308457e-05 | 0.0001022283, 1.909921e-05, 9.792624e-05 | worsened / worsened_all_three |
| aniso_k1 | B-A | angular_sw_bin0 | 7.156034e-05 | 1.746137e-05, 0.0001072548, 8.996483e-05 | worsened / worsened_all_three |
| aniso_k1 | B-A | angular_sw_bin1 | 1.878881e-05 | 0.0001172582, -4.26136e-05, -1.827814e-05 | worsened / mixed_or_tied_seeds |
| aniso_k1 | B-A | angular_sw_bin2 | -1.421679e-05 | -2.724212e-05, -8.275872e-05, 6.735045e-05 | improved / mixed_or_tied_seeds |
| aniso_k1 | B-A | angular_sw_bin3 | 0.0002162059 | 0.0003014356, 9.451434e-05, 0.0002526678 | worsened / worsened_all_three |
| aniso_k10 | B-A | angular_sw_mean | 0.0006706554 | 0.001751888, -0.0003213319, 0.0005814104 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-A | angular_sw_bin0 | 0.001267488 | 0.002876977, 0.0003415663, 0.0005839225 | worsened / worsened_all_three |
| aniso_k10 | B-A | angular_sw_bin1 | 0.0007292616 | 0.001401108, 0.0003389223, 0.0004477547 | worsened / worsened_all_three |
| aniso_k10 | B-A | angular_sw_bin2 | 0.0004395548 | 0.001666818, -0.001032774, 0.0006846208 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-A | angular_sw_bin3 | 0.000246317 | 0.001062649, -0.0009330418, 0.0006093434 | worsened / mixed_or_tied_seeds |
| aniso_k100 | B-A | angular_sw_mean | -0.001167147 | -0.0006497796, -0.001835616, -0.001016044 | improved / improved_all_three |
| aniso_k100 | B-A | angular_sw_bin0 | -0.001426721 | -0.0008511348, -0.001149404, -0.002279625 | improved / improved_all_three |
| aniso_k100 | B-A | angular_sw_bin1 | -0.0003593229 | 0.0002814345, -0.001734493, 0.0003750902 | improved / mixed_or_tied_seeds |
| aniso_k100 | B-A | angular_sw_bin2 | -0.0009081999 | -0.000235552, -0.001561998, -0.0009270492 | improved / improved_all_three |
| aniso_k100 | B-A | angular_sw_bin3 | -0.001974342 | -0.001793866, -0.002896569, -0.001232591 | improved / improved_all_three |
| aniso_k3 | B-A | angular_sw_mean | 0.0003660947 | 0.0002997683, 0.0002838157, 0.0005147002 | worsened / worsened_all_three |
| aniso_k3 | B-A | angular_sw_bin0 | 0.0005053257 | 0.0002242066, 0.000390823, 0.0009009475 | worsened / worsened_all_three |
| aniso_k3 | B-A | angular_sw_bin1 | 0.0001620421 | 4.759664e-05, 0.000166852, 0.0002716775 | worsened / worsened_all_three |
| aniso_k3 | B-A | angular_sw_bin2 | 0.0001597169 | 0.000204761, -9.357836e-05, 0.0003679679 | worsened / mixed_or_tied_seeds |
| aniso_k3 | B-A | angular_sw_bin3 | 0.0006372943 | 0.0007225089, 0.000671166, 0.0005182079 | worsened / worsened_all_three |
| aniso_k30 | B-A | angular_sw_mean | -0.0003669038 | -0.0003007168, -0.0008238655, 2.387085e-05 | improved / mixed_or_tied_seeds |
| aniso_k30 | B-A | angular_sw_bin0 | -0.0004553813 | -0.0007325588, -0.0005129762, -0.0001206091 | improved / improved_all_three |
| aniso_k30 | B-A | angular_sw_bin1 | -4.634758e-05 | -4.347134e-05, -0.0005221171, 0.0004265457 | improved / mixed_or_tied_seeds |
| aniso_k30 | B-A | angular_sw_bin2 | -0.000442865 | -0.0002055038, -0.0004301486, -0.0006929426 | improved / improved_all_three |
| aniso_k30 | B-A | angular_sw_bin3 | -0.0005230214 | -0.0002213335, -0.00183022, 0.0004824894 | improved / mixed_or_tied_seeds |
| aniso_k300 | B-A | angular_sw_mean | -0.005743676 | -0.007655684, -0.005475069, -0.004100277 | improved / improved_all_three |
| aniso_k300 | B-A | angular_sw_bin0 | -0.00452963 | -0.005686947, -0.004735847, -0.003166097 | improved / improved_all_three |
| aniso_k300 | B-A | angular_sw_bin1 | -0.005835894 | -0.00875804, -0.005046089, -0.003703554 | improved / improved_all_three |
| aniso_k300 | B-A | angular_sw_bin2 | -0.005923168 | -0.009139535, -0.005570618, -0.003059351 | improved / improved_all_three |
| aniso_k300 | B-A | angular_sw_bin3 | -0.006686013 | -0.007038212, -0.006547721, -0.006472107 | improved / improved_all_three |
| audiomnist_stft | B-A | angular_sw_mean | -3.384552e-06 | 4.196481e-06, -7.411829e-06, -6.93831e-06 | improved / mixed_or_tied_seeds |
| audiomnist_stft | B-A | angular_sw_bin0 | -6.528882e-06 | -4.471513e-06, -2.534955e-05, 1.023442e-05 | improved / mixed_or_tied_seeds |
| audiomnist_stft | B-A | angular_sw_bin1 | 6.022262e-06 | 1.425075e-05, 2.070388e-05, -1.688785e-05 | worsened / mixed_or_tied_seeds |
| audiomnist_stft | B-A | angular_sw_bin2 | -3.211744e-06 | -5.493173e-06, -1.291919e-06, -2.850138e-06 | improved / improved_all_three |
| audiomnist_stft | B-A | angular_sw_bin3 | -9.819846e-06 | 1.249986e-05, -2.370973e-05, -1.824967e-05 | improved / mixed_or_tied_seeds |
| finance_ff49 | B-A | angular_sw_mean | 0.0001283568 | 0.0003109942, 0.0006558474, -0.0005817711 | worsened / mixed_or_tied_seeds |
| finance_ff49 | B-A | angular_sw_bin0 | 0.0006652248 | 0.001259435, 0.0008412646, -0.0001050252 | worsened / mixed_or_tied_seeds |
| finance_ff49 | B-A | angular_sw_bin1 | 0.0005394363 | 0.0004122425, 0.0006547403, 0.0005513262 | worsened / worsened_all_three |
| finance_ff49 | B-A | angular_sw_bin2 | -3.894667e-05 | 0.0002911985, -0.0005297512, 0.0001217127 | improved / mixed_or_tied_seeds |
| finance_ff49 | B-A | angular_sw_bin3 | -0.0006522872 | -0.0007188991, 0.001657136, -0.002895098 | improved / mixed_or_tied_seeds |
| gaussian_aniso_d16_cor | B-A | angular_sw_mean | 0.00112898 | 0.001996744, -0.000320704, 0.0017109 | worsened / mixed_or_tied_seeds |
| gaussian_aniso_d16_cor | B-A | angular_sw_bin0 | 0.001373145 | 0.001470816, 0.0006771842, 0.001971434 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | B-A | angular_sw_bin1 | 0.0006146853 | 0.0008377871, 0.000101041, 0.0009052278 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | B-A | angular_sw_bin2 | 0.001173781 | 0.002179737, -0.0007805321, 0.002122138 | worsened / mixed_or_tied_seeds |
| gaussian_aniso_d16_cor | B-A | angular_sw_bin3 | 0.00135431 | 0.003498636, -0.001280509, 0.001844802 | worsened / mixed_or_tied_seeds |
| imagenette_dcae | B-A | angular_sw_mean | 0.003774879 | 0.00226845, 0.002003331, 0.007052855 | worsened / worsened_all_three |
| imagenette_dcae | B-A | angular_sw_bin0 | 0.003493874 | 0.002196786, 0.001825387, 0.006459449 | worsened / worsened_all_three |
| imagenette_dcae | B-A | angular_sw_bin1 | 0.00355162 | 0.002298777, 0.001952841, 0.006403243 | worsened / worsened_all_three |
| imagenette_dcae | B-A | angular_sw_bin2 | 0.004162736 | 0.002659699, 0.00165246, 0.008176047 | worsened / worsened_all_three |
| imagenette_dcae | B-A | angular_sw_bin3 | 0.003891285 | 0.001918539, 0.002582636, 0.00717268 | worsened / worsened_all_three |
| piv_d16 | B-A | angular_sw_mean | 0.001165385 | 0.001495113, 0.0009078868, 0.001093157 | worsened / worsened_all_three |
| piv_d16 | B-A | angular_sw_bin0 | -0.001675664 | -0.0006630309, -0.003211126, -0.001152836 | improved / improved_all_three |
| piv_d16 | B-A | angular_sw_bin1 | -0.003100616 | -0.003285408, -0.0007415414, -0.005274899 | improved / improved_all_three |
| piv_d16 | B-A | angular_sw_bin2 | 0.0001442432 | 0.0009958968, -0.002916165, 0.002352998 | worsened / mixed_or_tied_seeds |
| piv_d16 | B-A | angular_sw_bin3 | 0.009293579 | 0.008932993, 0.01050038, 0.008447364 | worsened / worsened_all_three |
| piv_d256 | B-A | angular_sw_mean | 0.0001931381 | 0.000335566, 0.0004195918, -0.0001757434 | worsened / mixed_or_tied_seeds |
| piv_d256 | B-A | angular_sw_bin0 | 4.40885e-05 | 0.0003520902, 0.0006524576, -0.0008722823 | worsened / mixed_or_tied_seeds |
| piv_d256 | B-A | angular_sw_bin1 | 0.0002567774 | 0.0001613293, 0.001078721, -0.0004697181 | worsened / mixed_or_tied_seeds |
| piv_d256 | B-A | angular_sw_bin2 | -1.795559e-05 | -0.0003325678, -8.361787e-05, 0.0003623189 | improved / mixed_or_tied_seeds |
| piv_d256 | B-A | angular_sw_bin3 | 0.0004896422 | 0.001161412, 3.080629e-05, 0.000276708 | worsened / worsened_all_three |
| piv_d32 | B-A | angular_sw_mean | 0.000671527 | 0.003746992, -0.001469306, -0.0002631047 | worsened / mixed_or_tied_seeds |
| piv_d32 | B-A | angular_sw_bin0 | 0.0008182799 | 0.001267159, -0.0001890883, 0.001376769 | worsened / mixed_or_tied_seeds |
| piv_d32 | B-A | angular_sw_bin1 | -0.001148436 | 0.0003453977, -0.002769087, -0.00102162 | improved / mixed_or_tied_seeds |
| piv_d32 | B-A | angular_sw_bin2 | -0.0005425972 | 0.002130732, -0.002855476, -0.0009030476 | improved / mixed_or_tied_seeds |
| piv_d32 | B-A | angular_sw_bin3 | 0.003558862 | 0.01124468, -6.357208e-05, -0.0005045198 | worsened / mixed_or_tied_seeds |
| piv_d64 | B-A | angular_sw_mean | 0.0003778149 | -0.0001260079, 0.0003003436, 0.000959109 | worsened / mixed_or_tied_seeds |
| piv_d64 | B-A | angular_sw_bin0 | -0.0005226855 | 0.0002474226, -0.0001300368, -0.001685442 | improved / mixed_or_tied_seeds |
| piv_d64 | B-A | angular_sw_bin1 | -0.0002974924 | -0.0007266663, -0.0004586149, 0.0002928041 | improved / mixed_or_tied_seeds |
| piv_d64 | B-A | angular_sw_bin2 | -0.0001158547 | -0.0004111305, -8.381903e-06, 7.194839e-05 | improved / mixed_or_tied_seeds |
| piv_d64 | B-A | angular_sw_bin3 | 0.002447292 | 0.0003863424, 0.001798408, 0.005157126 | worsened / worsened_all_three |
| student_t_d128_df3.0_cor | B-A | angular_sw_mean | -0.00180138 | -0.00244499, -0.001651059, -0.00130809 | improved / improved_all_three |
| student_t_d128_df3.0_cor | B-A | angular_sw_bin0 | -0.00156434 | -0.002285548, -0.000877033, -0.001530441 | improved / improved_all_three |
| student_t_d128_df3.0_cor | B-A | angular_sw_bin1 | -0.001905965 | -0.002461371, -0.001765915, -0.001490608 | improved / improved_all_three |
| student_t_d128_df3.0_cor | B-A | angular_sw_bin2 | -0.002029219 | -0.002912124, -0.001655381, -0.001520153 | improved / improved_all_three |
| student_t_d128_df3.0_cor | B-A | angular_sw_bin3 | -0.001705994 | -0.002120916, -0.002305906, -0.0006911592 | improved / improved_all_three |
| student_t_d16_df1.5_cor | B-A | angular_sw_mean | -0.003056091 | -0.0007082697, -0.006070071, -0.002389933 | improved / improved_all_three |
| student_t_d16_df1.5_cor | B-A | angular_sw_bin0 | -0.001695553 | 0.001725651, -0.004241897, -0.002570415 | improved / mixed_or_tied_seeds |
| student_t_d16_df1.5_cor | B-A | angular_sw_bin1 | -0.002613999 | -0.0007130643, -0.004996226, -0.002132706 | improved / improved_all_three |
| student_t_d16_df1.5_cor | B-A | angular_sw_bin2 | -0.004098657 | -0.0006554872, -0.006677988, -0.004962496 | improved / improved_all_three |
| student_t_d16_df1.5_cor | B-A | angular_sw_bin3 | -0.003816156 | -0.003190178, -0.008364173, 0.0001058839 | improved / mixed_or_tied_seeds |
| student_t_d16_df10.0_cor | B-A | angular_sw_mean | 0.001742385 | 0.002436832, 0.001603677, 0.001186646 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | B-A | angular_sw_bin0 | 0.00154492 | 0.001908212, 0.0019316, 0.0007949471 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | B-A | angular_sw_bin1 | 0.001344413 | 0.002551205, 0.001140047, 0.0003419891 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | B-A | angular_sw_bin2 | 0.002131046 | 0.002276057, 0.001777542, 0.002339539 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | B-A | angular_sw_bin3 | 0.00194916 | 0.003011853, 0.001565521, 0.001270108 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | B-A | angular_sw_mean | -0.001181489 | -0.0006280029, -0.001978919, -0.000937545 | improved / improved_all_three |
| student_t_d16_df2.0_cor | B-A | angular_sw_bin0 | -0.001329508 | -0.0002360791, -0.0006217277, -0.003130718 | improved / improved_all_three |
| student_t_d16_df2.0_cor | B-A | angular_sw_bin1 | -0.001206144 | -0.0002581365, -0.003263358, -9.693671e-05 | improved / improved_all_three |
| student_t_d16_df2.0_cor | B-A | angular_sw_bin2 | -0.00170008 | -0.0005963892, -0.001595707, -0.002908143 | improved / improved_all_three |
| student_t_d16_df2.0_cor | B-A | angular_sw_bin3 | -0.0004902246 | -0.001421407, -0.002434885, 0.002385618 | improved / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-A | angular_sw_mean | 0.0007673673 | 0.0004436977, 0.0005193809, 0.001339023 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | B-A | angular_sw_bin0 | 0.0009407817 | 0.001962409, -0.0004267134, 0.001286649 | worsened / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-A | angular_sw_bin1 | 0.0006026455 | -8.640811e-05, 0.0003485605, 0.001545784 | worsened / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-A | angular_sw_bin2 | -1.374632e-05 | -0.0008722786, 0.0001936499, 0.0006373897 | improved / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-A | angular_sw_bin3 | 0.001539789 | 0.000771068, 0.001962027, 0.001886271 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | B-A | angular_sw_mean | 0.0009598892 | 0.0004552188, -0.0002502389, 0.002674688 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-A | angular_sw_bin0 | 0.00181239 | 0.001403786, 0.0005509686, 0.003482416 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | B-A | angular_sw_bin1 | 0.00135921 | 0.0005121334, -0.0001556696, 0.003721167 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-A | angular_sw_bin2 | 0.0009552377 | -5.041435e-05, -0.0001540119, 0.003070139 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-A | angular_sw_bin3 | -0.0002872817 | -4.462991e-05, -0.001242243, 0.0004250277 | improved / mixed_or_tied_seeds |
| student_t_d16_df50.0_cor | B-A | angular_sw_mean | 0.001561415 | 0.0008626832, 0.00163029, 0.002191271 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | B-A | angular_sw_bin0 | 0.00209787 | 0.00161032, 0.002552489, 0.002130801 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | B-A | angular_sw_bin1 | 0.001447281 | 0.0004805485, 0.002061877, 0.001799416 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | B-A | angular_sw_bin2 | 0.001170701 | -0.0003178148, 0.001338338, 0.002491578 | worsened / mixed_or_tied_seeds |
| student_t_d16_df50.0_cor | B-A | angular_sw_bin3 | 0.001529807 | 0.001677679, 0.0005684551, 0.002343288 | worsened / worsened_all_three |
| student_t_d256_df3.0_cor | B-A | angular_sw_mean | -0.0009385826 | -0.0009773214, -0.0009220817, -0.0009163448 | improved / improved_all_three |
| student_t_d256_df3.0_cor | B-A | angular_sw_bin0 | -0.001265174 | -0.001028291, -0.001506992, -0.00126024 | improved / improved_all_three |
| student_t_d256_df3.0_cor | B-A | angular_sw_bin1 | -0.0009013306 | -0.0009697902, -0.0008465163, -0.0008876852 | improved / improved_all_three |
| student_t_d256_df3.0_cor | B-A | angular_sw_bin2 | -0.0009287517 | -0.001098185, -0.0006577824, -0.001030288 | improved / improved_all_three |
| student_t_d256_df3.0_cor | B-A | angular_sw_bin3 | -0.0006590739 | -0.0008130195, -0.0006770359, -0.0004871662 | improved / improved_all_three |
| student_t_d32_df3.0_cor | B-A | angular_sw_mean | -0.0007287489 | -0.001221617, -0.0004278824, -0.0005367475 | improved / improved_all_three |
| student_t_d32_df3.0_cor | B-A | angular_sw_bin0 | -5.96242e-05 | -1.323689e-05, -0.0001193779, -4.625786e-05 | improved / improved_all_three |
| student_t_d32_df3.0_cor | B-A | angular_sw_bin1 | -0.0004851303 | -0.0002456876, -0.0005481811, -0.0006615222 | improved / improved_all_three |
| student_t_d32_df3.0_cor | B-A | angular_sw_bin2 | -0.001019585 | -0.001989566, -0.0001651458, -0.0009040423 | improved / improved_all_three |
| student_t_d32_df3.0_cor | B-A | angular_sw_bin3 | -0.001350656 | -0.002637977, -0.0008788249, -0.0005351678 | improved / improved_all_three |
| student_t_d64_df3.0_cor | B-A | angular_sw_mean | -0.001137848 | 0.0005529399, -0.001329454, -0.00263703 | improved / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-A | angular_sw_bin0 | -0.0006935854 | 0.0009810235, -0.0007109474, -0.002350832 | improved / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-A | angular_sw_bin1 | -0.0005720778 | 0.001352918, -0.0006585335, -0.002410618 | improved / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-A | angular_sw_bin2 | -0.001002593 | 0.0004640371, -0.001204297, -0.002267519 | improved / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-A | angular_sw_bin3 | -0.002283136 | -0.0005862191, -0.00274404, -0.003519149 | improved / improved_all_three |
| student_t_d8_df3.0_cor | B-A | angular_sw_mean | -0.0001092963 | 0.0009906366, 0.0004585683, -0.001777094 | improved / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-A | angular_sw_bin0 | 0.001831684 | 0.002914198, 0.003061747, -0.000480894 | worsened / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-A | angular_sw_bin1 | -0.0002195152 | 0.0008326471, -3.805943e-05, -0.001453133 | improved / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-A | angular_sw_bin2 | -0.0008885628 | -0.0005040253, 0.0003751218, -0.002536785 | improved / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-A | angular_sw_bin3 | -0.001160791 | 0.000719727, -0.001564536, -0.002637563 | improved / mixed_or_tied_seeds |
| weather_au_wind | B-A | angular_sw_mean | 0.0002278842 | 0.0004995156, -0.0002339147, 0.0004180518 | worsened / mixed_or_tied_seeds |
| weather_au_wind | B-A | angular_sw_bin0 | -0.0009560691 | -0.0008808374, -0.001127252, -0.0008601174 | improved / improved_all_three |
| weather_au_wind | B-A | angular_sw_bin1 | 0.0007903588 | 0.001537347, -0.0003693402, 0.001203069 | worsened / mixed_or_tied_seeds |
| weather_au_wind | B-A | angular_sw_bin2 | 0.0006373283 | 0.001180933, 0.0005556364, 0.0001754155 | worsened / worsened_all_three |
| weather_au_wind | B-A | angular_sw_bin3 | 0.0004399189 | 0.0001606196, 5.297363e-06, 0.00115384 | worsened / worsened_all_three |
| aniso_k1 | B-C | angular_sw_mean | 0.0001685809 | 0.0001948711, 0.0001397039, 0.0001711677 | worsened / worsened_all_three |
| aniso_k1 | B-C | angular_sw_bin0 | 0.0001188141 | 1.885928e-06, 0.0002244851, 0.0001300713 | worsened / worsened_all_three |
| aniso_k1 | B-C | angular_sw_bin1 | 0.0001178151 | 0.0002237605, 6.230827e-05, 6.737653e-05 | worsened / worsened_all_three |
| aniso_k1 | B-C | angular_sw_bin2 | -3.597637e-05 | -4.05591e-05, -0.0001106481, 4.327809e-05 | improved / mixed_or_tied_seeds |
| aniso_k1 | B-C | angular_sw_bin3 | 0.0004736707 | 0.0005943971, 0.0003826702, 0.0004439447 | worsened / worsened_all_three |
| aniso_k10 | B-C | angular_sw_mean | -3.391403e-05 | 0.001122696, -0.0007603692, -0.0004640687 | improved / mixed_or_tied_seeds |
| aniso_k10 | B-C | angular_sw_bin0 | 0.0004730814 | 0.001988997, 4.98658e-05, -0.0006196182 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-C | angular_sw_bin1 | 0.0002763389 | 0.0008593481, -2.089236e-05, -9.438954e-06 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-C | angular_sw_bin2 | 0.0003218077 | 0.001688046, -0.0009613195, 0.0002386961 | worsened / mixed_or_tied_seeds |
| aniso_k10 | B-C | angular_sw_bin3 | -0.001206884 | -4.56078e-05, -0.002109131, -0.001465914 | improved / improved_all_three |
| aniso_k100 | B-C | angular_sw_mean | -0.001549887 | -0.001986417, -0.001136838, -0.001526407 | improved / improved_all_three |
| aniso_k100 | B-C | angular_sw_bin0 | -0.003395253 | -0.003673816, -0.003583573, -0.002928372 | improved / improved_all_three |
| aniso_k100 | B-C | angular_sw_bin1 | -0.0003229703 | -3.762264e-05, -0.0006875107, -0.0002437774 | improved / improved_all_three |
| aniso_k100 | B-C | angular_sw_bin2 | 0.0002097638 | 0.0002395203, 0.0007050605, -0.0003152895 | worsened / mixed_or_tied_seeds |
| aniso_k100 | B-C | angular_sw_bin3 | -0.002691089 | -0.00447375, -0.0009813281, -0.002618189 | improved / improved_all_three |
| aniso_k3 | B-C | angular_sw_mean | 0.0002433504 | 0.0001784019, 0.0001294005, 0.0004222487 | worsened / worsened_all_three |
| aniso_k3 | B-C | angular_sw_bin0 | 0.000599985 | 3.462937e-05, 0.0005929451, 0.001172381 | worsened / worsened_all_three |
| aniso_k3 | B-C | angular_sw_bin1 | 0.0003878125 | 0.0002386607, 0.0003756359, 0.0005491409 | worsened / worsened_all_three |
| aniso_k3 | B-C | angular_sw_bin2 | 0.0001227475 | 0.0001850133, -0.0002029864, 0.0003862157 | worsened / mixed_or_tied_seeds |
| aniso_k3 | B-C | angular_sw_bin3 | -0.0001371435 | 0.0002553044, -0.0002479926, -0.0004187422 | improved / mixed_or_tied_seeds |
| aniso_k30 | B-C | angular_sw_mean | -0.001628803 | -0.001811056, -0.001639757, -0.001435597 | improved / improved_all_three |
| aniso_k30 | B-C | angular_sw_bin0 | -0.002501218 | -0.002866858, -0.001963672, -0.002673123 | improved / improved_all_three |
| aniso_k30 | B-C | angular_sw_bin1 | -0.0005267776 | -0.000492515, -0.0005708421, -0.0005169758 | improved / improved_all_three |
| aniso_k30 | B-C | angular_sw_bin2 | -1.949041e-05 | -0.000212675, -7.855752e-05, 0.0002327613 | improved / mixed_or_tied_seeds |
| aniso_k30 | B-C | angular_sw_bin3 | -0.003467728 | -0.003672176, -0.003945955, -0.002785052 | improved / improved_all_three |
| aniso_k300 | B-C | angular_sw_mean | -0.003327243 | -0.002856665, -0.00315959, -0.003965474 | improved / improved_all_three |
| aniso_k300 | B-C | angular_sw_bin0 | -0.004630181 | -0.005122524, -0.00413741, -0.00463061 | improved / improved_all_three |
| aniso_k300 | B-C | angular_sw_bin1 | -0.001415514 | -0.001388499, -0.00104435, -0.001813693 | improved / improved_all_three |
| aniso_k300 | B-C | angular_sw_bin2 | -0.0009524284 | -0.0003095954, -0.0004137917, -0.002133898 | improved / improved_all_three |
| aniso_k300 | B-C | angular_sw_bin3 | -0.006310849 | -0.004606043, -0.007042807, -0.007283696 | improved / improved_all_three |
| audiomnist_stft | B-C | angular_sw_mean | -1.263237e-06 | 3.716908e-06, -7.420938e-06, -8.568168e-08 | improved / mixed_or_tied_seeds |
| audiomnist_stft | B-C | angular_sw_bin0 | -3.624339e-06 | -2.25252e-06, -1.253351e-05, 3.91301e-06 | improved / mixed_or_tied_seeds |
| audiomnist_stft | B-C | angular_sw_bin1 | -2.651281e-06 | 6.319897e-06, -7.406983e-06, -6.866758e-06 | improved / mixed_or_tied_seeds |
| audiomnist_stft | B-C | angular_sw_bin2 | 5.64106e-06 | 2.964458e-06, 1.853332e-06, 1.210539e-05 | worsened / worsened_all_three |
| audiomnist_stft | B-C | angular_sw_bin3 | -4.418388e-06 | 7.835799e-06, -1.15966e-05, -9.494368e-06 | improved / mixed_or_tied_seeds |
| finance_ff49 | B-C | angular_sw_mean | -0.002831384 | -0.002848838, -0.002101074, -0.00354424 | improved / improved_all_three |
| finance_ff49 | B-C | angular_sw_bin0 | -0.004141612 | -0.003424622, -0.004752176, -0.004248037 | improved / improved_all_three |
| finance_ff49 | B-C | angular_sw_bin1 | 2.845501e-06 | -0.0007520672, 0.0003396664, 0.0004209373 | worsened / mixed_or_tied_seeds |
| finance_ff49 | B-C | angular_sw_bin2 | 0.0008988958 | 0.0008027945, 0.0007973611, 0.001096532 | worsened / worsened_all_three |
| finance_ff49 | B-C | angular_sw_bin3 | -0.008085666 | -0.008021457, -0.004789148, -0.01144639 | improved / improved_all_three |
| gaussian_aniso_d16_cor | B-C | angular_sw_mean | -0.000936318 | -0.0002537202, -0.001752359, -0.0008028748 | improved / improved_all_three |
| gaussian_aniso_d16_cor | B-C | angular_sw_bin0 | -0.001906555 | -0.001861605, -0.002746678, -0.001111384 | improved / improved_all_three |
| gaussian_aniso_d16_cor | B-C | angular_sw_bin1 | 0.0003176664 | 0.0004795799, 0.0002559358, 0.0002174834 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | B-C | angular_sw_bin2 | 0.000459138 | 0.001687055, -0.0002719276, -3.771391e-05 | worsened / mixed_or_tied_seeds |
| gaussian_aniso_d16_cor | B-C | angular_sw_bin3 | -0.002615521 | -0.001319911, -0.004246766, -0.002279885 | improved / improved_all_three |
| imagenette_dcae | B-C | angular_sw_mean | 0.003232312 | 0.001862622, 0.001218709, 0.006615605 | worsened / worsened_all_three |
| imagenette_dcae | B-C | angular_sw_bin0 | 0.002852222 | 0.001845515, 0.0004177843, 0.006293367 | worsened / worsened_all_three |
| imagenette_dcae | B-C | angular_sw_bin1 | 0.003175587 | 0.002161468, 0.001159507, 0.006205784 | worsened / worsened_all_three |
| imagenette_dcae | B-C | angular_sw_bin2 | 0.003713461 | 0.002306206, 0.001025032, 0.007809146 | worsened / worsened_all_three |
| imagenette_dcae | B-C | angular_sw_bin3 | 0.003187978 | 0.001137296, 0.002272514, 0.006154123 | worsened / worsened_all_three |
| piv_d16 | B-C | angular_sw_mean | -0.003373864 | -0.002418953, -0.004478152, -0.003224487 | improved / improved_all_three |
| piv_d16 | B-C | angular_sw_bin0 | -0.01875083 | -0.01978858, -0.01769115, -0.01877277 | improved / improved_all_three |
| piv_d16 | B-C | angular_sw_bin1 | -0.01660032 | -0.01668647, -0.01342671, -0.01968777 | improved / improved_all_three |
| piv_d16 | B-C | angular_sw_bin2 | 0.003533963 | 0.003003724, 0.003650527, 0.003947638 | worsened / worsened_all_three |
| piv_d16 | B-C | angular_sw_bin3 | 0.01832173 | 0.02379552, 0.009554721, 0.02161495 | worsened / worsened_all_three |
| piv_d256 | B-C | angular_sw_mean | -0.0003423633 | -0.0003796029, -0.000114698, -0.0005327892 | improved / improved_all_three |
| piv_d256 | B-C | angular_sw_bin0 | -7.242709e-05 | -0.0006522303, 0.0004958408, -6.089173e-05 | improved / mixed_or_tied_seeds |
| piv_d256 | B-C | angular_sw_bin1 | -0.000202467 | -0.0003111809, 0.0002217069, -0.0005179271 | improved / mixed_or_tied_seeds |
| piv_d256 | B-C | angular_sw_bin2 | -0.0006769852 | -0.0006915247, -0.0003043162, -0.001035115 | improved / improved_all_three |
| piv_d256 | B-C | angular_sw_bin3 | -0.000417574 | 0.0001365244, -0.0008720234, -0.000517223 | improved / mixed_or_tied_seeds |
| piv_d32 | B-C | angular_sw_mean | -0.001846435 | 0.0008740425, -0.002508987, -0.003904359 | improved / mixed_or_tied_seeds |
| piv_d32 | B-C | angular_sw_bin0 | -0.005224419 | -0.004662782, -0.004715813, -0.00629466 | improved / improved_all_three |
| piv_d32 | B-C | angular_sw_bin1 | -0.005623758 | -0.004428372, -0.005735658, -0.006707244 | improved / improved_all_three |
| piv_d32 | B-C | angular_sw_bin2 | -0.001758323 | 2.782792e-06, -0.002568327, -0.002709426 | improved / mixed_or_tied_seeds |
| piv_d32 | B-C | angular_sw_bin3 | 0.005220761 | 0.01258454, 0.002983849, 9.389222e-05 | worsened / worsened_all_three |
| piv_d64 | B-C | angular_sw_mean | -0.005019479 | -0.005673275, -0.005004739, -0.004380424 | improved / improved_all_three |
| piv_d64 | B-C | angular_sw_bin0 | -0.003571595 | -0.003886787, -0.003086451, -0.003741547 | improved / improved_all_three |
| piv_d64 | B-C | angular_sw_bin1 | -0.001642903 | -0.001956586, -0.001560681, -0.001411442 | improved / improved_all_three |
| piv_d64 | B-C | angular_sw_bin2 | -0.001292581 | -0.0009155143, -0.0007810108, -0.002181219 | improved / improved_all_three |
| piv_d64 | B-C | angular_sw_bin3 | -0.01357084 | -0.01593421, -0.01459081, -0.01018749 | improved / improved_all_three |
| student_t_d128_df3.0_cor | B-C | angular_sw_mean | 5.948466e-05 | 4.957383e-05, 6.226776e-05, 6.661238e-05 | worsened / worsened_all_three |
| student_t_d128_df3.0_cor | B-C | angular_sw_bin0 | 0.000174438 | -1.911633e-05, 0.0004942631, 4.816707e-05 | worsened / mixed_or_tied_seeds |
| student_t_d128_df3.0_cor | B-C | angular_sw_bin1 | 8.111199e-05 | 6.841635e-05, 0.0002076193, -3.269967e-05 | worsened / mixed_or_tied_seeds |
| student_t_d128_df3.0_cor | B-C | angular_sw_bin2 | -8.931383e-07 | 2.544094e-05, -1.646765e-05, -1.165271e-05 | improved / mixed_or_tied_seeds |
| student_t_d128_df3.0_cor | B-C | angular_sw_bin3 | -1.671817e-05 | 0.0001235544, -0.0004363437, 0.0002626348 | improved / mixed_or_tied_seeds |
| student_t_d16_df1.5_cor | B-C | angular_sw_mean | 0.0002160058 | 0.0001882436, -0.0003126576, 0.0007724313 | worsened / mixed_or_tied_seeds |
| student_t_d16_df1.5_cor | B-C | angular_sw_bin0 | 0.0004882213 | 0.001778951, -0.0006710161, 0.0003567291 | worsened / mixed_or_tied_seeds |
| student_t_d16_df1.5_cor | B-C | angular_sw_bin1 | 0.0006372062 | 0.0004187953, 0.0003002696, 0.001192554 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | B-C | angular_sw_bin2 | 0.0002441118 | 9.400956e-05, -8.244067e-05, 0.0007207664 | worsened / mixed_or_tied_seeds |
| student_t_d16_df1.5_cor | B-C | angular_sw_bin3 | -0.0005055163 | -0.001538781, -0.0007974431, 0.0008196756 | improved / mixed_or_tied_seeds |
| student_t_d16_df10.0_cor | B-C | angular_sw_mean | -0.0005826435 | -0.0002900101, -0.0008217329, -0.0006361874 | improved / improved_all_three |
| student_t_d16_df10.0_cor | B-C | angular_sw_bin0 | -0.00164478 | -0.0006608702, -0.001631842, -0.002641627 | improved / improved_all_three |
| student_t_d16_df10.0_cor | B-C | angular_sw_bin1 | 7.330161e-05 | 0.0008562198, -0.0001672981, -0.0004690168 | worsened / mixed_or_tied_seeds |
| student_t_d16_df10.0_cor | B-C | angular_sw_bin2 | 0.0005272481 | 0.0005271705, 0.0006709918, 0.000383582 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | B-C | angular_sw_bin3 | -0.001286344 | -0.001882561, -0.002158783, 0.000182312 | improved / mixed_or_tied_seeds |
| student_t_d16_df2.0_cor | B-C | angular_sw_mean | -0.0002719467 | -0.0004439489, -0.0003736804, 1.789303e-06 | improved / mixed_or_tied_seeds |
| student_t_d16_df2.0_cor | B-C | angular_sw_bin0 | -0.0003868019 | -0.0005914364, 0.0005427497, -0.001111719 | improved / mixed_or_tied_seeds |
| student_t_d16_df2.0_cor | B-C | angular_sw_bin1 | 9.781991e-07 | 0.0003745481, -0.0005295631, 0.0001579495 | worsened / mixed_or_tied_seeds |
| student_t_d16_df2.0_cor | B-C | angular_sw_bin2 | -0.0003690487 | -0.0004172865, -0.0002870392, -0.0004028203 | improved / improved_all_three |
| student_t_d16_df2.0_cor | B-C | angular_sw_bin3 | -0.0003329143 | -0.001141621, -0.001220869, 0.001363747 | improved / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-C | angular_sw_mean | 9.80217e-06 | -5.511125e-05, 0.0002615773, -0.0001770596 | worsened / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-C | angular_sw_bin0 | -0.0006089425 | 0.0003469642, -0.00140022, -0.0007735714 | improved / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-C | angular_sw_bin1 | 8.460134e-05 | 0.0002502017, 0.0001873141, -0.0001837118 | worsened / mixed_or_tied_seeds |
| student_t_d16_df3.0_cor | B-C | angular_sw_bin2 | -0.0004859334 | -0.0007587345, -0.0006483831, -5.068257e-05 | improved / improved_all_three |
| student_t_d16_df3.0_cor | B-C | angular_sw_bin3 | 0.001049483 | -5.887635e-05, 0.002907598, 0.0002997275 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-C | angular_sw_mean | -0.0007032825 | -0.001337551, -0.001145672, 0.0003733756 | improved / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-C | angular_sw_bin0 | -0.0006428044 | -0.0008541532, -0.00246051, 0.001386249 | improved / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-C | angular_sw_bin1 | 0.0009137516 | 0.0004284801, -0.0001706565, 0.002483431 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-C | angular_sw_bin2 | 0.0002607896 | -0.0004006233, 0.0001169378, 0.001066054 | worsened / mixed_or_tied_seeds |
| student_t_d16_df5.0_cor | B-C | angular_sw_bin3 | -0.003344867 | -0.004523908, -0.00206846, -0.003442232 | improved / improved_all_three |
| student_t_d16_df50.0_cor | B-C | angular_sw_mean | -0.0008827885 | -0.0009002052, -0.0007872554, -0.0009609049 | improved / improved_all_three |
| student_t_d16_df50.0_cor | B-C | angular_sw_bin0 | -0.001209921 | -0.0007581124, -0.001211641, -0.001660009 | improved / improved_all_three |
| student_t_d16_df50.0_cor | B-C | angular_sw_bin1 | 0.0007351534 | 0.0006058039, 0.0008993484, 0.000700308 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | B-C | angular_sw_bin2 | -0.0003114349 | -0.0006759148, 0.0002017152, -0.000460105 | improved / mixed_or_tied_seeds |
| student_t_d16_df50.0_cor | B-C | angular_sw_bin3 | -0.002744952 | -0.002772598, -0.003038444, -0.002423814 | improved / improved_all_three |
| student_t_d256_df3.0_cor | B-C | angular_sw_mean | 7.641511e-05 | 4.649156e-05, 8.086802e-05, 0.0001018858 | worsened / worsened_all_three |
| student_t_d256_df3.0_cor | B-C | angular_sw_bin0 | 0.0001648783 | 0.0001515003, 0.0002044924, 0.0001386423 | worsened / worsened_all_three |
| student_t_d256_df3.0_cor | B-C | angular_sw_bin1 | 5.690435e-05 | 7.444993e-06, 0.0001008492, 6.241887e-05 | worsened / worsened_all_three |
| student_t_d256_df3.0_cor | B-C | angular_sw_bin2 | 2.88873e-06 | 1.72148e-05, -2.995785e-05, 2.140924e-05 | worsened / mixed_or_tied_seeds |
| student_t_d256_df3.0_cor | B-C | angular_sw_bin3 | 8.098905e-05 | 9.806128e-06, 4.808838e-05, 0.0001850727 | worsened / worsened_all_three |
| student_t_d2_df3.0_cor | B-C | angular_sw_mean | -0.1118473 | -0.02922134, -0.2804443, -0.02587633 | improved / improved_all_three |
| student_t_d2_df3.0_cor | B-C | angular_sw_bin0 | -0.2390405 | -0.2539452, -0.4507591, -0.01241702 | improved / improved_all_three |
| student_t_d2_df3.0_cor | B-C | angular_sw_bin1 | -0.09450662 | 0.04501373, -0.3215729, -0.00696069 | improved / mixed_or_tied_seeds |
| student_t_d2_df3.0_cor | B-C | angular_sw_bin2 | -0.06559497 | 0.04380208, -0.2137984, -0.02678859 | improved / mixed_or_tied_seeds |
| student_t_d2_df3.0_cor | B-C | angular_sw_bin3 | -0.04824728 | 0.04824406, -0.1356469, -0.05733901 | improved / mixed_or_tied_seeds |
| student_t_d32_df3.0_cor | B-C | angular_sw_mean | 4.184898e-06 | 6.473437e-05, 2.874108e-05, -8.092076e-05 | worsened / mixed_or_tied_seeds |
| student_t_d32_df3.0_cor | B-C | angular_sw_bin0 | 0.0001105852 | 0.0001005065, 0.0001483103, 8.293893e-05 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | B-C | angular_sw_bin1 | 3.008793e-05 | 0.0001473138, 0.0001194961, -0.0001765462 | worsened / mixed_or_tied_seeds |
| student_t_d32_df3.0_cor | B-C | angular_sw_bin2 | -0.000141625 | 4.309323e-05, -0.00020028, -0.0002676882 | improved / mixed_or_tied_seeds |
| student_t_d32_df3.0_cor | B-C | angular_sw_bin3 | 1.76914e-05 | -3.197603e-05, 4.743785e-05, 3.761239e-05 | worsened / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-C | angular_sw_mean | 0.0001492842 | 0.0001656681, 0.0001404407, 0.0001417439 | worsened / worsened_all_three |
| student_t_d64_df3.0_cor | B-C | angular_sw_bin0 | 0.0002077076 | 4.885346e-05, 0.0006125756, -3.830623e-05 | worsened / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-C | angular_sw_bin1 | 6.698631e-05 | 3.273785e-05, 0.000116365, 5.185604e-05 | worsened / worsened_all_three |
| student_t_d64_df3.0_cor | B-C | angular_sw_bin2 | 1.439142e-05 | 7.035024e-05, -6.244332e-05, 3.526732e-05 | worsened / mixed_or_tied_seeds |
| student_t_d64_df3.0_cor | B-C | angular_sw_bin3 | 0.0003080516 | 0.0005107308, -0.0001047347, 0.0005181585 | worsened / mixed_or_tied_seeds |
| student_t_d8_df3.0_cor | B-C | angular_sw_mean | -0.003392337 | -0.003156928, -0.003549161, -0.003470921 | improved / improved_all_three |
| student_t_d8_df3.0_cor | B-C | angular_sw_bin0 | -0.002704974 | -0.001965998, -0.00342482, -0.002724104 | improved / improved_all_three |
| student_t_d8_df3.0_cor | B-C | angular_sw_bin1 | -0.0006596331 | -0.001002053, -0.0009327782, -4.406832e-05 | improved / improved_all_three |
| student_t_d8_df3.0_cor | B-C | angular_sw_bin2 | -0.001898624 | -0.002550089, -0.00155119, -0.001594594 | improved / improved_all_three |
| student_t_d8_df3.0_cor | B-C | angular_sw_bin3 | -0.008306116 | -0.007109574, -0.008287854, -0.009520918 | improved / improved_all_three |
| weather_au_wind | B-C | angular_sw_mean | -0.003124223 | -0.003222366, -0.003576383, -0.002573919 | improved / improved_all_three |
| weather_au_wind | B-C | angular_sw_bin0 | -0.006960284 | -0.006739853, -0.006915547, -0.007225452 | improved / improved_all_three |
| weather_au_wind | B-C | angular_sw_bin1 | -0.0002622536 | -0.0003731502, -0.0005648294, 0.0001512188 | improved / mixed_or_tied_seeds |
| weather_au_wind | B-C | angular_sw_bin2 | -1.761876e-05 | -0.0001311079, 7.863529e-05, -3.837049e-07 | improved / mixed_or_tied_seeds |
| weather_au_wind | B-C | angular_sw_bin3 | -0.005256734 | -0.005645353, -0.00690379, -0.003221058 | improved / improved_all_three |
| aniso_k1 | tflow-A | angular_sw_mean | 0.07995228 | 0.08954468, 0.06973996, 0.08057221 | worsened / worsened_all_three |
| aniso_k1 | tflow-A | angular_sw_bin0 | 0.06971446 | 0.0780897, 0.05821736, 0.07283633 | worsened / worsened_all_three |
| aniso_k1 | tflow-A | angular_sw_bin1 | 0.07729515 | 0.09017824, 0.06701536, 0.07469184 | worsened / worsened_all_three |
| aniso_k1 | tflow-A | angular_sw_bin2 | 0.08204934 | 0.08960458, 0.07314431, 0.08339914 | worsened / worsened_all_three |
| aniso_k1 | tflow-A | angular_sw_bin3 | 0.09075018 | 0.1003062, 0.08058282, 0.09136151 | worsened / worsened_all_three |
| aniso_k100 | tflow-A | angular_sw_mean | 0.07870078 | 0.07508095, 0.07998671, 0.0810347 | worsened / worsened_all_three |
| aniso_k100 | tflow-A | angular_sw_bin0 | 0.05830104 | 0.05338236, 0.05796541, 0.06355534 | worsened / worsened_all_three |
| aniso_k100 | tflow-A | angular_sw_bin1 | 0.0748062 | 0.07131936, 0.07117923, 0.08192002 | worsened / worsened_all_three |
| aniso_k100 | tflow-A | angular_sw_bin2 | 0.08533952 | 0.08495268, 0.08781504, 0.08325083 | worsened / worsened_all_three |
| aniso_k100 | tflow-A | angular_sw_bin3 | 0.09635638 | 0.0906694, 0.1029871, 0.09541259 | worsened / worsened_all_three |
| aniso_k3 | tflow-A | angular_sw_mean | 0.03246695 | 0.02296382, 0.02689872, 0.04753832 | worsened / worsened_all_three |
| aniso_k3 | tflow-A | angular_sw_bin0 | 0.03029794 | 0.02433088, 0.02366775, 0.04289519 | worsened / worsened_all_three |
| aniso_k3 | tflow-A | angular_sw_bin1 | 0.02802718 | 0.01896975, 0.0230475, 0.04206428 | worsened / worsened_all_three |
| aniso_k3 | tflow-A | angular_sw_bin2 | 0.03107868 | 0.02002509, 0.02356982, 0.04964112 | worsened / worsened_all_three |
| aniso_k3 | tflow-A | angular_sw_bin3 | 0.04046403 | 0.02852958, 0.0373098, 0.0555527 | worsened / worsened_all_three |
| aniso_k30 | tflow-A | angular_sw_mean | 0.07829382 | 0.06077817, 0.09302221, 0.08108108 | worsened / worsened_all_three |
| aniso_k30 | tflow-A | angular_sw_bin0 | 0.05830383 | 0.04341719, 0.06394276, 0.06755153 | worsened / worsened_all_three |
| aniso_k30 | tflow-A | angular_sw_bin1 | 0.07962951 | 0.05095127, 0.1069158, 0.0810214 | worsened / worsened_all_three |
| aniso_k30 | tflow-A | angular_sw_bin2 | 0.09044599 | 0.07174534, 0.1065066, 0.09308599 | worsened / worsened_all_three |
| aniso_k30 | tflow-A | angular_sw_bin3 | 0.08479596 | 0.07699889, 0.09472358, 0.08266542 | worsened / worsened_all_three |
| aniso_k300 | tflow-A | angular_sw_mean | 0.05564078 | 0.05404714, 0.06214706, 0.05072814 | worsened / worsened_all_three |
| aniso_k300 | tflow-A | angular_sw_bin0 | 0.05363142 | 0.05717541, 0.05685739, 0.04686147 | worsened / worsened_all_three |
| aniso_k300 | tflow-A | angular_sw_bin1 | 0.05665648 | 0.05484992, 0.06522904, 0.04989048 | worsened / worsened_all_three |
| aniso_k300 | tflow-A | angular_sw_bin2 | 0.05542896 | 0.0504005, 0.06734069, 0.0485457 | worsened / worsened_all_three |
| aniso_k300 | tflow-A | angular_sw_bin3 | 0.05684627 | 0.05376275, 0.05916113, 0.05761492 | worsened / worsened_all_three |
| finance_ff49 | tflow-A | angular_sw_mean | 0.04231066 | 0.03629281, 0.04619469, 0.04444448 | worsened / worsened_all_three |
| finance_ff49 | tflow-A | angular_sw_bin0 | 0.05874345 | 0.05708147, 0.05320826, 0.06594062 | worsened / worsened_all_three |
| finance_ff49 | tflow-A | angular_sw_bin1 | 0.04225919 | 0.03373862, 0.04462658, 0.04841238 | worsened / worsened_all_three |
| finance_ff49 | tflow-A | angular_sw_bin2 | 0.03993004 | 0.02767553, 0.04802443, 0.04409017 | worsened / worsened_all_three |
| finance_ff49 | tflow-A | angular_sw_bin3 | 0.02830995 | 0.02667563, 0.03891947, 0.01933474 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | tflow-A | angular_sw_mean | 0.05958875 | 0.06887382, 0.04474385, 0.06514857 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | tflow-A | angular_sw_bin0 | 0.03128013 | 0.02750522, 0.02591886, 0.04041631 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | tflow-A | angular_sw_bin1 | 0.04716013 | 0.04830839, 0.04015357, 0.05301843 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | tflow-A | angular_sw_bin2 | 0.06691711 | 0.07864571, 0.04315833, 0.0789473 | worsened / worsened_all_three |
| gaussian_aniso_d16_cor | tflow-A | angular_sw_bin3 | 0.09299762 | 0.121036, 0.06974464, 0.08821226 | worsened / worsened_all_three |
| piv_d16 | tflow-A | angular_sw_mean | 0.08769434 | 0.09123105, 0.1107328, 0.06111917 | worsened / worsened_all_three |
| piv_d16 | tflow-A | angular_sw_bin0 | 0.08584257 | 0.1119139, 0.06995637, 0.07565748 | worsened / worsened_all_three |
| piv_d16 | tflow-A | angular_sw_bin1 | 0.1019984 | 0.08493216, 0.1417424, 0.07932075 | worsened / worsened_all_three |
| piv_d16 | tflow-A | angular_sw_bin2 | 0.07714502 | 0.1128315, 0.0926168, 0.02598675 | worsened / worsened_all_three |
| piv_d16 | tflow-A | angular_sw_bin3 | 0.08579136 | 0.05524666, 0.1386157, 0.0635117 | worsened / worsened_all_three |
| piv_d32 | tflow-A | angular_sw_mean | 0.03730549 | 0.03730428, 0.02028644, 0.05432576 | worsened / worsened_all_three |
| piv_d32 | tflow-A | angular_sw_bin0 | 0.04479412 | 0.03672782, 0.03131662, 0.06633791 | worsened / worsened_all_three |
| piv_d32 | tflow-A | angular_sw_bin1 | 0.04703371 | 0.04317468, 0.0286995, 0.06922695 | worsened / worsened_all_three |
| piv_d32 | tflow-A | angular_sw_bin2 | 0.03377862 | 0.04544419, 0.0117539, 0.04413776 | worsened / worsened_all_three |
| piv_d32 | tflow-A | angular_sw_bin3 | 0.02361553 | 0.02387042, 0.009375729, 0.03760043 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | tflow-A | angular_sw_mean | 0.09231598 | 0.06921374, 0.1078153, 0.09991885 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | tflow-A | angular_sw_bin0 | 0.1056481 | 0.07134021, 0.1259386, 0.1196655 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | tflow-A | angular_sw_bin1 | 0.1004838 | 0.06849017, 0.1192321, 0.1137292 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | tflow-A | angular_sw_bin2 | 0.1040818 | 0.08459243, 0.1187777, 0.1088753 | worsened / worsened_all_three |
| student_t_d16_df1.5_cor | tflow-A | angular_sw_bin3 | 0.05905017 | 0.05243216, 0.067313, 0.05740536 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | tflow-A | angular_sw_mean | 0.09849613 | 0.1407917, 0.1166265, 0.03807017 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | tflow-A | angular_sw_bin0 | 0.08883141 | 0.1104946, 0.1104075, 0.04559207 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | tflow-A | angular_sw_bin1 | 0.09602967 | 0.1320198, 0.1170464, 0.03902283 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | tflow-A | angular_sw_bin2 | 0.09997114 | 0.1498787, 0.1192681, 0.03076659 | worsened / worsened_all_three |
| student_t_d16_df10.0_cor | tflow-A | angular_sw_bin3 | 0.1091523 | 0.1707738, 0.119784, 0.03689919 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | tflow-A | angular_sw_mean | 0.07060665 | 0.03840046, 0.08101287, 0.09240662 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | tflow-A | angular_sw_bin0 | 0.08146407 | 0.04272055, 0.09857723, 0.1030944 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | tflow-A | angular_sw_bin1 | 0.07733041 | 0.04986543, 0.09049124, 0.09163458 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | tflow-A | angular_sw_bin2 | 0.07620883 | 0.03841196, 0.08513704, 0.1050775 | worsened / worsened_all_three |
| student_t_d16_df2.0_cor | tflow-A | angular_sw_bin3 | 0.04742329 | 0.02260392, 0.04984597, 0.06981998 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | tflow-A | angular_sw_mean | 0.1087198 | 0.150606, 0.1386648, 0.0368886 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | tflow-A | angular_sw_bin0 | 0.1017873 | 0.1353849, 0.1304868, 0.03949011 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | tflow-A | angular_sw_bin1 | 0.1102715 | 0.1558909, 0.1328287, 0.04209474 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | tflow-A | angular_sw_bin2 | 0.1164312 | 0.165741, 0.1468781, 0.03667455 | worsened / worsened_all_three |
| student_t_d16_df3.0_cor | tflow-A | angular_sw_bin3 | 0.1063892 | 0.1454071, 0.1444654, 0.02929501 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | tflow-A | angular_sw_mean | 0.1056841 | 0.1644437, 0.1118619, 0.04074663 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | tflow-A | angular_sw_bin0 | 0.0920538 | 0.1197888, 0.1158859, 0.04048668 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | tflow-A | angular_sw_bin1 | 0.1042282 | 0.1662323, 0.1071094, 0.03934286 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | tflow-A | angular_sw_bin2 | 0.1181353 | 0.1837508, 0.1275685, 0.04308673 | worsened / worsened_all_three |
| student_t_d16_df5.0_cor | tflow-A | angular_sw_bin3 | 0.108319 | 0.1880028, 0.09688393, 0.04007025 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | tflow-A | angular_sw_mean | 0.09050068 | 0.136561, 0.0898227, 0.0451183 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | tflow-A | angular_sw_bin0 | 0.07698427 | 0.09851177, 0.08330579, 0.04913524 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | tflow-A | angular_sw_bin1 | 0.08345837 | 0.1160344, 0.08478555, 0.04955522 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | tflow-A | angular_sw_bin2 | 0.09029199 | 0.1448028, 0.08893491, 0.03713828 | worsened / worsened_all_three |
| student_t_d16_df50.0_cor | tflow-A | angular_sw_bin3 | 0.1112681 | 0.1868952, 0.1022646, 0.04464446 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | tflow-A | angular_sw_mean | 0.1147438 | 0.1284996, 0.1199361, 0.09579563 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | tflow-A | angular_sw_bin0 | 0.1110879 | 0.1178531, 0.1129163, 0.1024944 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | tflow-A | angular_sw_bin1 | 0.115455 | 0.1222195, 0.1246864, 0.09945911 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | tflow-A | angular_sw_bin2 | 0.1194727 | 0.1388442, 0.1249734, 0.09460049 | worsened / worsened_all_three |
| student_t_d32_df3.0_cor | tflow-A | angular_sw_bin3 | 0.1129595 | 0.1350818, 0.1171683, 0.08662855 | worsened / worsened_all_three |
| student_t_d8_df3.0_cor | tflow-A | angular_sw_mean | 0.1553705 | 0.168265, 0.1597587, 0.1380876 | worsened / worsened_all_three |
| student_t_d8_df3.0_cor | tflow-A | angular_sw_bin0 | 0.0944578 | 0.08334392, 0.08438534, 0.1156442 | worsened / worsened_all_three |
| student_t_d8_df3.0_cor | tflow-A | angular_sw_bin1 | 0.1418107 | 0.1528802, 0.144396, 0.128156 | worsened / worsened_all_three |
| student_t_d8_df3.0_cor | tflow-A | angular_sw_bin2 | 0.1713439 | 0.1871036, 0.1812721, 0.1456561 | worsened / worsened_all_three |
| student_t_d8_df3.0_cor | tflow-A | angular_sw_bin3 | 0.2138694 | 0.2497323, 0.2289814, 0.1628943 | worsened / worsened_all_three |

Angular sliced W1 is lower-is-better. Empty/nonfinite bins and incomplete seed groups do not acquire averages.

## Recorded cost

| Condition | Ratio | Training: mean seed ratio | Sampling: mean seed ratio |
|---|---|---|---|
| aniso_k1 | B/A | 0.9771411 | 1.514212 |
| aniso_k10 | B/A | 1.034765 | 1.469214 |
| aniso_k100 | B/A | 0.9517821 | 1.517783 |
| aniso_k3 | B/A | 1.037528 | 1.511159 |
| aniso_k30 | B/A | 1.051035 | 1.563972 |
| aniso_k300 | B/A | 1.017695 | 1.422357 |
| audiomnist_stft | B/A | 1.001966 | 1.003662 |
| finance_ff49 | B/A | 1.014874 | 1.537914 |
| gaussian_aniso_d16_cor | B/A | 1.0302 | 1.445777 |
| imagenette_dcae | B/A | 1.016998 | 0.9986976 |
| piv_d16 | B/A | 1.024624 | 1.434989 |
| piv_d256 | B/A | 1.084314 | 1.368758 |
| piv_d32 | B/A | 1.001396 | 1.535782 |
| piv_d64 | B/A | 0.945496 | 1.498254 |
| student_t_d128_df3.0_cor | B/A | 1.062121 | 1.591048 |
| student_t_d16_df1.5_cor | B/A | 1.048409 | 1.476674 |
| student_t_d16_df10.0_cor | B/A | 1.020373 | 1.394375 |
| student_t_d16_df2.0_cor | B/A | 1.092551 | 1.435165 |
| student_t_d16_df3.0_cor | B/A | 1.054313 | 1.524515 |
| student_t_d16_df5.0_cor | B/A | 1.008031 | 1.499576 |
| student_t_d16_df50.0_cor | B/A | 1.061444 | 1.476843 |
| student_t_d256_df3.0_cor | B/A | 1.084681 | 1.373227 |
| student_t_d32_df3.0_cor | B/A | 0.9335396 | 1.476478 |
| student_t_d64_df3.0_cor | B/A | 1.016366 | 1.569298 |
| student_t_d8_df3.0_cor | B/A | 1.022744 | 1.449338 |
| weather_au_wind | B/A | 1.019022 | 1.489006 |
| aniso_k1 | B/C | 0.9798256 | 1.1349 |
| aniso_k10 | B/C | 1.0214 | 1.116045 |
| aniso_k100 | B/C | 1.014897 | 1.138891 |
| aniso_k3 | B/C | 1.01325 | 1.140112 |
| aniso_k30 | B/C | 0.995352 | 1.147737 |
| aniso_k300 | B/C | 0.9575948 | 1.124766 |
| audiomnist_stft | B/C | 1.001588 | 1.007894 |
| finance_ff49 | B/C | 0.9777842 | 1.185797 |
| gaussian_aniso_d16_cor | B/C | 0.977868 | 1.101767 |
| imagenette_dcae | B/C | 1.011332 | 0.9993051 |
| piv_d16 | B/C | 0.996719 | 1.11367 |
| piv_d256 | B/C | 1.057918 | 1.058723 |
| piv_d32 | B/C | 1.023988 | 1.098732 |
| piv_d64 | B/C | 0.9608142 | 1.091905 |
| student_t_d128_df3.0_cor | B/C | 1.002673 | 1.125532 |
| student_t_d16_df1.5_cor | B/C | 0.9620146 | 1.10039 |
| student_t_d16_df10.0_cor | B/C | 0.9537725 | 1.121786 |
| student_t_d16_df2.0_cor | B/C | 1.050807 | 1.119489 |
| student_t_d16_df3.0_cor | B/C | 1.014978 | 1.165419 |
| student_t_d16_df5.0_cor | B/C | 0.9814909 | 1.130985 |
| student_t_d16_df50.0_cor | B/C | 0.9680009 | 1.137975 |
| student_t_d256_df3.0_cor | B/C | 1.069478 | 1.087665 |
| student_t_d2_df3.0_cor | B/C | 1.025687 | 1.131207 |
| student_t_d32_df3.0_cor | B/C | 0.9228765 | 1.112085 |
| student_t_d64_df3.0_cor | B/C | 1.009299 | 1.135745 |
| student_t_d8_df3.0_cor | B/C | 0.9498459 | 1.142686 |
| weather_au_wind | B/C | 0.9729371 | 1.170624 |

These are raw ratios of measured times on matching recorded GPU/software and precision, not significance claims. Ratios of means and each seed ratio are also saved in findings.json.

| Condition | Method | Total parameters | Added conditioning parameters | Added fraction |
|---|---|---|---|---|
| aniso_k1 | A | 41504 | 0 | 0 |
| aniso_k1 | B | 41632 | 128 | 0.00308404 |
| aniso_k1 | C | 41632 | 128 | 0.00308404 |
| aniso_k1 | tflow | 41504 | unavailable | unavailable |
| aniso_k10 | A | 41504 | 0 | 0 |
| aniso_k10 | B | 41632 | 128 | 0.00308404 |
| aniso_k10 | C | 41632 | 128 | 0.00308404 |
| aniso_k100 | A | 41504 | 0 | 0 |
| aniso_k100 | B | 41632 | 128 | 0.00308404 |
| aniso_k100 | C | 41632 | 128 | 0.00308404 |
| aniso_k100 | tflow | 41504 | unavailable | unavailable |
| aniso_k3 | A | 41504 | 0 | 0 |
| aniso_k3 | B | 41632 | 128 | 0.00308404 |
| aniso_k3 | C | 41632 | 128 | 0.00308404 |
| aniso_k3 | tflow | 41504 | unavailable | unavailable |
| aniso_k30 | A | 41504 | 0 | 0 |
| aniso_k30 | B | 41632 | 128 | 0.00308404 |
| aniso_k30 | C | 41632 | 128 | 0.00308404 |
| aniso_k30 | tflow | 41504 | unavailable | unavailable |
| aniso_k300 | A | 41504 | 0 | 0 |
| aniso_k300 | B | 41632 | 128 | 0.00308404 |
| aniso_k300 | C | 41632 | 128 | 0.00308404 |
| aniso_k300 | tflow | 41504 | unavailable | unavailable |
| audiomnist_stft | A | 2.78968e+07 | 0 | 0 |
| audiomnist_stft | B | 2.790119e+07 | 4384 | 0.0001571506 |
| audiomnist_stft | C | 2.790119e+07 | 4384 | 0.0001571506 |
| audiomnist_stft | tflow | 2.78968e+07 | unavailable | unavailable |
| finance_ff49 | A | 45873 | 0 | 0 |
| finance_ff49 | B | 46001 | 128 | 0.002790312 |
| finance_ff49 | C | 46001 | 128 | 0.002790312 |
| finance_ff49 | tflow | 45873 | unavailable | unavailable |
| gaussian_aniso_d16_cor | A | 37392 | 0 | 0 |
| gaussian_aniso_d16_cor | B | 37520 | 128 | 0.003423192 |
| gaussian_aniso_d16_cor | C | 37520 | 128 | 0.003423192 |
| gaussian_aniso_d16_cor | tflow | 37392 | unavailable | unavailable |
| imagenette_dcae | A | 3.251562e+07 | 0 | 0 |
| imagenette_dcae | B | 3.252218e+07 | 6560 | 0.0002017492 |
| imagenette_dcae | C | 3.252218e+07 | 6560 | 0.0002017492 |
| imagenette_dcae | tflow | 3.251562e+07 | unavailable | unavailable |
| piv_d16 | A | 37392 | 0 | 0 |
| piv_d16 | B | 37520 | 128 | 0.003423192 |
| piv_d16 | C | 37520 | 128 | 0.003423192 |
| piv_d16 | tflow | 37392 | unavailable | unavailable |
| piv_d256 | A | 99072 | 0 | 0 |
| piv_d256 | B | 99200 | 128 | 0.00129199 |
| piv_d256 | C | 99200 | 128 | 0.00129199 |
| piv_d32 | A | 41504 | 0 | 0 |
| piv_d32 | B | 41632 | 128 | 0.00308404 |
| piv_d32 | C | 41632 | 128 | 0.00308404 |
| piv_d32 | tflow | 41504 | unavailable | unavailable |
| piv_d64 | A | 49728 | 0 | 0 |
| piv_d64 | B | 49856 | 128 | 0.002574003 |
| piv_d64 | C | 49856 | 128 | 0.002574003 |
| student_t_d128_df3.0_cor | A | 66176 | 0 | 0 |
| student_t_d128_df3.0_cor | B | 66304 | 128 | 0.001934236 |
| student_t_d128_df3.0_cor | C | 66304 | 128 | 0.001934236 |
| student_t_d16_df1.5_cor | A | 37392 | 0 | 0 |
| student_t_d16_df1.5_cor | B | 37520 | 128 | 0.003423192 |
| student_t_d16_df1.5_cor | C | 37520 | 128 | 0.003423192 |
| student_t_d16_df1.5_cor | tflow | 37392 | unavailable | unavailable |
| student_t_d16_df10.0_cor | A | 37392 | 0 | 0 |
| student_t_d16_df10.0_cor | B | 37520 | 128 | 0.003423192 |
| student_t_d16_df10.0_cor | C | 37520 | 128 | 0.003423192 |
| student_t_d16_df10.0_cor | tflow | 37392 | unavailable | unavailable |
| student_t_d16_df2.0_cor | A | 37392 | 0 | 0 |
| student_t_d16_df2.0_cor | B | 37520 | 128 | 0.003423192 |
| student_t_d16_df2.0_cor | C | 37520 | 128 | 0.003423192 |
| student_t_d16_df2.0_cor | tflow | 37392 | unavailable | unavailable |
| student_t_d16_df3.0_cor | A | 37392 | 0 | 0 |
| student_t_d16_df3.0_cor | B | 37520 | 128 | 0.003423192 |
| student_t_d16_df3.0_cor | C | 37520 | 128 | 0.003423192 |
| student_t_d16_df3.0_cor | tflow | 37392 | unavailable | unavailable |
| student_t_d16_df5.0_cor | A | 37392 | 0 | 0 |
| student_t_d16_df5.0_cor | B | 37520 | 128 | 0.003423192 |
| student_t_d16_df5.0_cor | C | 37520 | 128 | 0.003423192 |
| student_t_d16_df5.0_cor | tflow | 37392 | unavailable | unavailable |
| student_t_d16_df50.0_cor | A | 37392 | 0 | 0 |
| student_t_d16_df50.0_cor | B | 37520 | 128 | 0.003423192 |
| student_t_d16_df50.0_cor | C | 37520 | 128 | 0.003423192 |
| student_t_d16_df50.0_cor | tflow | 37392 | unavailable | unavailable |
| student_t_d256_df3.0_cor | A | 99072 | 0 | 0 |
| student_t_d256_df3.0_cor | B | 99200 | 128 | 0.00129199 |
| student_t_d256_df3.0_cor | C | 99200 | 128 | 0.00129199 |
| student_t_d2_df3.0_cor | B | 33922 | 128 | 0.003787655 |
| student_t_d2_df3.0_cor | C | 33922 | 128 | 0.003787655 |
| student_t_d2_df3.0_cor | tflow | 33794 | unavailable | unavailable |
| student_t_d32_df3.0_cor | A | 41504 | 0 | 0 |
| student_t_d32_df3.0_cor | B | 41632 | 128 | 0.00308404 |
| student_t_d32_df3.0_cor | C | 41632 | 128 | 0.00308404 |
| student_t_d32_df3.0_cor | tflow | 41504 | unavailable | unavailable |
| student_t_d64_df3.0_cor | A | 49728 | 0 | 0 |
| student_t_d64_df3.0_cor | B | 49856 | 128 | 0.002574003 |
| student_t_d64_df3.0_cor | C | 49856 | 128 | 0.002574003 |
| student_t_d8_df3.0_cor | A | 35336 | 0 | 0 |
| student_t_d8_df3.0_cor | B | 35464 | 128 | 0.003622368 |
| student_t_d8_df3.0_cor | C | 35464 | 128 | 0.003622368 |
| student_t_d8_df3.0_cor | tflow | 35336 | unavailable | unavailable |
| toy_radial_angular | tflow | 33794 | unavailable | unavailable |
| weather_au_wind | A | 57952 | 0 | 0 |
| weather_au_wind | B | 58080 | 128 | 0.002208724 |
| weather_au_wind | C | 58080 | 128 | 0.002208724 |

## t-Flow

Against newly run A, t-Flow has 19 complete comparisons: {"improved": 0, "tied": 0, "worsened": 19} on each condition’s primary metric. Ranking is descriptive and limited to the complete methods listed below.

| Condition | Primary metric | Methods ordered by mean | Missing methods |
|---|---|---|---|
| aniso_k1 | sliced_w1 | A, C, B, tflow | none |
| aniso_k100 | sliced_w1 | B, C, A, tflow | none |
| aniso_k3 | sliced_w1 | A, C, B, tflow | none |
| aniso_k30 | sliced_w1 | B, A, C, tflow | none |
| aniso_k300 | sliced_w1 | B, C, A, tflow | none |
| audiomnist_stft | digit_acc | A, C, B, tflow | none |
| finance_ff49 | sliced_w1 | A, B, C, tflow | none |
| gaussian_aniso_d16_cor | sliced_w1 | A, B, C, tflow | none |
| imagenette_dcae | fid | A, tflow, C, B | none |
| piv_d16 | sliced_w1 | A, C, B, tflow | none |
| piv_d32 | sliced_w1 | A, C, B, tflow | none |
| student_t_d16_df1.5_cor | sliced_w1 | C, B, A, tflow | none |
| student_t_d16_df10.0_cor | sliced_w1 | A, C, B, tflow | none |
| student_t_d16_df2.0_cor | sliced_w1 | C, B, A, tflow | none |
| student_t_d16_df3.0_cor | sliced_w1 | A, C, B, tflow | none |
| student_t_d16_df5.0_cor | sliced_w1 | A, B, C, tflow | none |
| student_t_d16_df50.0_cor | sliced_w1 | A, B, C, tflow | none |
| student_t_d2_df3.0_cor | sliced_w1 | B, C, tflow | A |
| student_t_d32_df3.0_cor | sliced_w1 | C, B, A, tflow | none |
| student_t_d8_df3.0_cor | sliced_w1 | B, A, C, tflow | none |

| Condition | Validation training-loop seconds | Validation sampling seconds | Training-update equivalents |
|---|---|---|---|
| aniso_k1 | 20.49362 | 4.163436 | 0.55 |
| aniso_k10 | 24.3679 | 4.199747 | 0.55 |
| aniso_k100 | 20.17464 | 4.204593 | 0.55 |
| aniso_k3 | 20.69776 | 4.205527 | 0.55 |
| aniso_k30 | 22.33547 | 4.108426 | 0.55 |
| aniso_k300 | 20.6037 | 4.200517 | 0.55 |
| audiomnist_stft | 4919.846 | 4835.394 | 0.55 |
| finance_ff49 | 19.71517 | 4.354534 | 0.55 |
| gaussian_aniso_d16_cor | 22.53189 | 4.293257 | 0.55 |
| imagenette_dcae | 1105.997 | 183.0851 | 0.55 |
| piv_d16 | 20.83483 | 4.162566 | 0.55 |
| piv_d256 | 22.0748 | 4.203993 | 0.55 |
| piv_d32 | 28.76755 | 4.148067 | 0.55 |
| piv_d64 | 19.79262 | 4.305727 | 0.55 |
| student_t_d128_df3.0_cor | 19.07718 | 4.042663 | 0.55 |
| student_t_d16_df1.5_cor | 19.25178 | 4.148318 | 0.55 |
| student_t_d16_df10.0_cor | 18.95523 | 4.076806 | 0.55 |
| student_t_d16_df2.0_cor | 21.7214 | 4.472294 | 0.55 |
| student_t_d16_df3.0_cor | 19.56787 | 4.104465 | 0.55 |
| student_t_d16_df5.0_cor | 18.70861 | 4.064509 | 0.55 |
| student_t_d16_df50.0_cor | 17.81318 | 4.143133 | 0.55 |
| student_t_d256_df3.0_cor | 20.46317 | 4.21281 | 0.55 |
| student_t_d2_df3.0_cor | 18.86274 | 3.898608 | 0.55 |
| student_t_d32_df3.0_cor | 19.23536 | 4.167972 | 0.55 |
| student_t_d64_df3.0_cor | 23.87276 | 4.22157 | 0.55 |
| student_t_d8_df3.0_cor | 23.21805 | 4.25049 | 0.55 |
| toy_radial_angular | 18.80803 | 3.870854 | 0.55 |
| weather_au_wind | 20.70263 | 4.269771 | 0.55 |

Tuning training time sums initial cumulative times plus each finalist’s cumulative time minus its own initial time. It does not double-count continuation checkpoints. Validation metrics, data loading and scheduling overhead are unmeasured here; incomplete/inconsistent trials are marked unavailable.

## Audio checkpoint reference

Best measured digit-accuracy mean among the complete compatible listed methods: **fixed_spherical_empirical_gain_reference**.

| Method | Accuracy | Energy KS | Coverage > q95 | Coverage > q99 |
|---|---|---|---|---|
| A | 0.7883333 ± 0.01133088 | 0.02183333 ± 0 | 0.05 ± 0 | 0.0145 ± 0 |
| B | 0.7765 ± 0.01349691 | 0.02183333 ± 0 | 0.05 ± 0 | 0.0145 ± 0 |
| C | 0.7868333 ± 0.008209074 | 0.02183333 ± 0 | 0.05 ± 0 | 0.0145 ± 0 |
| tflow | 0.1448333 ± 0.006908127 | 0.8322778 ± 0.009593915 | 0.8755 ± 0.01098484 | 0.764 ± 0.01784657 |
| fixed_spherical_empirical_gain_reference | 0.8066667 ± 0.007352248 | 0.02183333 ± 0 | 0.05 ± 0 | 0.0145 ± 0 |

These are newly measured metrics for the same saved Y/X under the current study backend. Archived metrics and raw audit status remain unchanged. Digit predictions, accuracy, energy KS and coverage rates are identical; PIT, radial W1, logits and near-fixed-radius energy-bin assignments can differ. Only these current-backend values enter new comparisons; no historical training-time comparison.

The raw backend audit remains compatibility_discrepancy; numerical differences are retained in findings.json and the report reference record.

Measured fixed-spherical accuracy is about 0.8067, not an exact reproduction of the reported 0.810 ± 0.013. The original baseline mismatch is preserved.

The checkpoint-based fixed-spherical+gain row is not new A and is not used for matched training-time claims. Constructed gains are independent of content; a B/C benefit is not presumed. The RAFM-Ang versus RAFM-Vel interpretation remains separate.

## Missing, failed and blocked work

- Current final seed aniso_k10 / tflow / 77395: failed; []
- Current final seed piv_d256 / tflow / 8925: failed; []
- Current final seed piv_d256 / tflow / 77395: failed; []
- Current final seed piv_d256 / tflow / 65457: failed; []
- Current final seed piv_d64 / tflow / 8925: failed; []
- Current final seed piv_d64 / tflow / 77395: failed; []
- Current final seed piv_d64 / tflow / 65457: failed; []
- Current final seed student_t_d128_df3.0_cor / tflow / 8925: failed; []
- Current final seed student_t_d128_df3.0_cor / tflow / 77395: failed; []
- Current final seed student_t_d128_df3.0_cor / tflow / 65457: failed; []
- Current final seed student_t_d256_df3.0_cor / tflow / 8925: failed; []
- Current final seed student_t_d256_df3.0_cor / tflow / 77395: failed; []
- Current final seed student_t_d256_df3.0_cor / tflow / 65457: failed; []
- Current final seed student_t_d2_df3.0_cor / A / 65457: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed student_t_d64_df3.0_cor / tflow / 8925: failed; []
- Current final seed student_t_d64_df3.0_cor / tflow / 77395: failed; []
- Current final seed student_t_d64_df3.0_cor / tflow / 65457: failed; []
- Current final seed toy_radial_angular / A / 77395: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed toy_radial_angular / A / 65457: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed toy_radial_angular / B / 8925: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed toy_radial_angular / B / 65457: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed toy_radial_angular / C / 8925: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed toy_radial_angular / C / 77395: failed; {"message": "Nonfinite samples in batch starting at 0", "traceback": "Traceback (most recent call last):\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 394, in main\n    data=load_data(cfg); train(cfg,data,out,args.arm,args.seed); evaluate(cfg,data,out,args.arm,args.seed)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 321, in evaluate\n    generated=sample(cfg,model,data,cfg['evaluation']['n_samples'],cfg['evaluation']['sample_seed'])\n  File \"/mnt/vast01/users/fouad.oubari/msgm/msgm-sparse-control/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py\", line 116, in decorate_context\n    return func(*args, **kwargs)\n  File \"/mnt/vast01/users/fouad.oubari/msgm/rafm-additions/experiments/rafm_inputs/run.py\", line 252, in sample\n    raise FloatingPointError(f'Nonfinite samples in batch starting at {start}')\nFloatingPointError: Nonfinite samples in batch starting at 0\n", "type": "FloatingPointError"}
- Current final seed weather_au_wind / tflow / 8925: failed; []
- Current final seed weather_au_wind / tflow / 77395: failed; []
- Current final seed weather_au_wind / tflow / 65457: failed; []
- Unavailable B-A comparisons: student_t_d2_df3.0_cor, toy_radial_angular
- Unavailable B-C comparisons: toy_radial_angular
- Unavailable tflow-A comparisons: aniso_k10, piv_d256, piv_d64, student_t_d128_df3.0_cor, student_t_d256_df3.0_cor, student_t_d2_df3.0_cor, student_t_d64_df3.0_cor, toy_radial_angular, weather_au_wind

Separately preserved: 2 implementation-check failure archives and 1 earlier t-Flow startup-attempt archives. They are not current scientific seed failures.

## Interpretation limits

- t-Flow denotes an independent direct-noise reproduction with documented backbone and endpoint adaptations. For vector dimensions above 128, the width-128 affine output bottleneck leaves noise components amplified by 1/t_min=1000. These adaptation failures do not establish intrinsic inferiority of published t-Flow; see docs/tflow_matched_backbone_failure_analysis.md.
- New synthetic realizations are shared A/B/C/t-Flow comparisons, never historical tensors or replacement historical results.
- Three-seed mean differences and sign consistency are descriptive, not tests of statistical significance.
- No pooling of metric units across datasets, no averaging over missing/failed seeds, and no data-dependent tolerance for a tie.
- B versus A changes input parameterization and includes radius conditioning; B versus C isolates access to the scalar radius in identical modules.
- Condition sweeps share related generators and are not independent statistical replicates.
- This input reparameterization preserves the population angular transport objective and creates no new theoretical guarantee.
