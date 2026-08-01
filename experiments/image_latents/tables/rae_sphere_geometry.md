# Phase 2b — RAE/DINOv2-B sphere geometry (N=2000)

| quantity | value | meaning |
|---|---|---|
| global norm CoV | 0.0162 | spread of ||z_flat|| (196608-d) across images |
| pooled per-token CoV | 0.0385 | all tokens×images |
| within-image cross-token CoV | 0.0345 | do tokens in ONE image share a radius? |
| per-position CoV (mean / max) | 0.0320 / 0.0354 | does a fixed token position keep constant radius across images? |
| between-position mean-radius rel-std | 0.0213 | do the 256 positions sit on the SAME sphere? |

**Verdict:** product of ~identical near-fixed-radius token spheres (≈ single global fixed-radius sphere)

The global RAFM implementation treats the flattened vector as a single global sphere; the DINO/RAE representation is more precisely a product of per-token spheres (per-token LayerNorm). Both are near-fixed-radius here.
