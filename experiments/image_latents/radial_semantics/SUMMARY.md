# Radial-semantics (CPU, model space = scaled x0.41407, train-centered)

- n=13394 (train 8036), D=2048; radius mean 41.67 std 4.89 CoV 0.117
- train radial quantiles: q05=34.89 q50=41.06 q95=51.13 q99=56.57 (min 27.46 max 61.96)
- quantile group sizes: bottom05=651, bottom10=1336, median45_55=1319, top10=1340, top05=681, top01=149
- class effect on radius: eta^2=0.162 (classes low->high radius: n01440764, n02102040, n03445777, n03888257, n03417042, n03028079, n03000684, n03394916, n03425413, n02979186)
- PCA(sub=4000): top-10 EVR sum 0.224, 50-comp 0.357; max |corr(radius,PC)|=0.307 @PC3

Saved: radial_semantics.json, radial_quantile_indices.npz, pca_subsample.npz