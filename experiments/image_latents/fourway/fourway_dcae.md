# Phase 5 four-way on DC-AE latent (2048-d, CoV 12%), 3 seeds

Radial-W1 floor (empirical source, irreducible): train/val 0.19, train/test 0.5121. Latent-distribution metrics (not image FID).

| method | radial_w1 | sliced_w1 | dir_sliced_w1 | cr_sliced_w1 | ks_stat | nan_rate |
|---|---|---|---|---|---|---|
| gaussian_euclidean | 23.9493±1.4386 | 0.4630±0.0262 | 0.0023±0.0002 | 0.2380±0.0225 | 0.8037±0.0320 | 0.0000 |
| matched_euclidean | 19.2300±0.5548 | 0.3680±0.0046 | 0.0015±0.0000 | 0.1451±0.0027 | 0.5449±0.0107 | 0.0000 |
| fixed_spherical | 9.1047±0.0000 | 0.1280±0.0189 | 0.0012±0.0002 | 0.1280±0.0144 | 0.5351 | 0.0000 |
| rafm_empirical | 0.6718±0.1141 | 0.1164±0.0060 | 0.0012±0.0001 | 0.1243±0.0134 | 0.0367±0.0081 | 0.0000 |
