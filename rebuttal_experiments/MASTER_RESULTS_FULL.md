# Angular RAFM — full experimental suite (all previously-run settings)

Mean ± std over 3 seeds. All metrics **lower = better** (radial_w1, ks, sliced_w1, angular_sw). Blank = method not run in that setting. Angular RAFM added to every setting where standard RAFM / baselines already existed; all other numbers are the retained baselines, unchanged. See `ANGULAR_AUDIT.md` for the radial-metric anomaly analysis.

## Cross-domain headline — DC-AE image + AudioMNIST

### DC-AE ImageNette latents (SiT, 40k)

| Method | fid | radial_w1 | ks | sliced_w1 | precision | recall | coverage |
|---|---|---|---|---|---|---|---|
| Gaussian FM | 160.956±0.610 | 4.680±0.096 | 0.417±0.008 | 0.093±0.002 | 0.135±0.003 | 0.693±0.010 | 0.043±0.001 |
| Matched-Eucl. | 171.741±1.205 | 2.626±0.153 | 0.243±0.017 | 0.063±0.002 | 0.124±0.004 | 0.685±0.006 | 0.035±0.001 |
| Fixed-spherical | 166.485±0.393 | 3.770±0.000 | 0.535±0.000 | 0.045±0.000 | 0.137±0.004 | 0.671±0.007 | 0.046±0.002 |
| RAFM (std) | 174.507±1.165 | 0.266±0.000 | 0.040±0.000 | 0.042±0.000 | 0.130±0.002 | 0.667±0.035 | 0.041±0.004 |
| Angular RAFM | 147.345±0.923 | 0.320±0.000 | 0.044±0.000 | 0.040±0.000 | 0.160±0.003 | 0.654±0.011 | 0.082±0.003 |
| MSGM |  |  |  |  |  |  |  |

### AudioMNIST (reversible STFT, UNet, 24k)

| Method | digit_acc | energy_KS | cov>q95 | cov>q99 | PIT |
|---|---|---|---|---|---|
| Gaussian FM | 0.734±0.004 | 0.117±0.008 | 0.059±0.004 | 0.019±0.003 | 0.487±0.004 |
| Matched-Eucl. | 0.750±0.018 | 0.095±0.010 | 0.040±0.001 | 0.012±0.000 | 0.450±0.005 |
| Fixed-spherical | 0.810±0.013 | 0.592±0.000 | 0.000±0.000 | 0.000±0.000 | 0.592±0.000 |
| RAFM (std) | 0.711±0.014 | 0.022±0.000 | 0.050±0.000 | 0.015±0.000 | 0.492±0.000 |
| Angular RAFM | 0.764±0.025 | 0.022±0.000 | 0.050±0.000 | 0.015±0.000 | 0.492±0.000 |
| MSGM |  |  |  |  |  |

## Dimension sweep — Student-t df3 (E6)

### Overview: std-RAFM vs Angular across dimension

| dim | radial_w1 std | radial_w1 ang | sliced_w1 std | sliced_w1 ang | angular_sw std | angular_sw ang |
|---|---|---|---|---|---|---|
| 2 | 0.1340 | 0.0202 | 0.4667 | 0.2203 | 0.8086 | 0.5289 |
| 8 | 0.1504 | 0.3427 | 0.1715 | 0.1673 | 0.0163 | 0.0141 |
| 16 | 0.1904 | 0.2161 | 0.2454 | 0.2058 | 0.0129 | 0.0113 |
| 32 | 0.4353 | 0.6961 | 0.4297 | 0.3768 | 0.0108 | 0.0096 |
| 64 | 0.9821 | 0.6293 | 0.8401 | 0.6056 | 0.0099 | 0.0073 |
| 128 | 0.8503 | 1.4734 | 1.0071 | 0.9492 | 0.0060 | 0.0055 |
| 256 | 2.7032 | 2.3969 | 1.2412 | 1.2508 | 0.0037 | 0.0037 |

### dim = 2

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.0132±0.0024 | 0.0118±0.0031 | 0.0172±0.0028 | 0.0346±0.0055 | 3 |
| Source-only (emp.) | 0.0186±0.0018 | 0.0243±0.0054 | 0.0231±0.0069 | 0.0558±0.0171 | 3 |
| RAFM (std) | 0.1340±0.1696 | 0.2681±0.3052 | 0.4667±0.1025 | 0.8086±0.0225 | 3 |
| RAFM (oracle) | 0.1321±0.1692 | 0.2692±0.3042 | 0.4684±0.0908 | 0.8080±0.0191 | 3 |
| Angular RAFM | 0.0202±0.0122 | 0.3413±0.4657 | 0.2203±0.0636 | 0.5289±0.2205 | 2 |

**Angular − std-RAFM:** radial_w1 -0.1138 (better); ks_stat +0.0732 (worse); sliced_w1 -0.2464 (better); angular_sw_mean -0.2796 (better)

### dim = 8

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.4109±0.1179 | 0.0285±0.0093 | 0.2054±0.0234 | 0.0186±0.0039 | 3 |
| Source-only (emp.) | 0.1839±0.0128 | 0.0171±0.0037 | 0.1765±0.0174 | 0.0159±0.0015 | 3 |
| RAFM (std) | 0.1504±0.0068 | 0.0118±0.0028 | 0.1715±0.0136 | 0.0163±0.0010 | 3 |
| RAFM (oracle) | 0.1790±0.0276 | 0.0155±0.0007 | 0.1680±0.0112 | 0.0173±0.0009 | 3 |
| Angular RAFM | 0.3427±0.0156 | 0.0228±0.0025 | 0.1673±0.0288 | 0.0141±0.0019 | 3 |

**Angular − std-RAFM:** radial_w1 +0.1923 (worse); ks_stat +0.0110 (worse); sliced_w1 -0.0042 (better); angular_sw_mean -0.0022 (better)

### dim = 16

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 1.7486±0.7439 | 0.0719±0.0480 | 0.6902±0.0918 | 0.0251±0.0055 | 3 |
| Source-only (emp.) | 0.4935±0.0699 | 0.0235±0.0023 | 0.2779±0.0185 | 0.0144±0.0016 | 3 |
| RAFM (std) | 0.1904±0.0184 | 0.0114±0.0024 | 0.2454±0.0117 | 0.0129±0.0002 | 3 |
| RAFM (oracle) | 0.2535±0.0787 | 0.0118±0.0002 | 0.2424±0.0108 | 0.0140±0.0007 | 3 |
| Angular RAFM | 0.2161±0.0409 | 0.0108±0.0008 | 0.2058±0.0191 | 0.0113±0.0003 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0257 (worse); ks_stat -0.0007 (better); sliced_w1 -0.0397 (better); angular_sw_mean -0.0016 (better)

### dim = 32

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 9.8341±0.0929 | 0.3537±0.0048 | 1.4402±0.1313 | 0.0219±0.0018 | 3 |
| Source-only (emp.) | 1.2821±0.3077 | 0.0425±0.0048 | 0.5613±0.0813 | 0.0132±0.0012 | 3 |
| RAFM (std) | 0.4353±0.0649 | 0.0117±0.0023 | 0.4297±0.0515 | 0.0108±0.0003 | 3 |
| RAFM (oracle) | 0.3829±0.0307 | 0.0133±0.0036 | 0.4149±0.0437 | 0.0105±0.0005 | 3 |
| Angular RAFM | 0.6961±0.0471 | 0.0192±0.0011 | 0.3768±0.0232 | 0.0096±0.0006 | 3 |

**Angular − std-RAFM:** radial_w1 +0.2609 (worse); ks_stat +0.0075 (worse); sliced_w1 -0.0530 (better); angular_sw_mean -0.0012 (better)

### dim = 64

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 10.6461±0.6390 | 0.1644±0.0235 | 1.5405±0.1890 | 0.0148±0.0027 | 3 |
| Source-only (emp.) | 9.2176±0.1782 | 0.1665±0.0062 | 1.4280±0.1923 | 0.0119±0.0027 | 3 |
| RAFM (std) | 0.9821±0.1214 | 0.0133±0.0018 | 0.8401±0.1744 | 0.0099±0.0016 | 3 |
| RAFM (oracle) | 0.9288±0.0948 | 0.0141±0.0022 | 0.8442±0.1724 | 0.0097±0.0015 | 3 |
| Angular RAFM | 0.6293±0.1106 | 0.0094±0.0015 | 0.6056±0.0522 | 0.0073±0.0004 | 3 |

**Angular − std-RAFM:** radial_w1 -0.3529 (better); ks_stat -0.0040 (better); sliced_w1 -0.2344 (better); angular_sw_mean -0.0026 (better)

### dim = 128

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 124.2599±1.4248 | 0.9987±0.0006 | 9.0170±0.1472 | 0.0331±0.0045 | 3 |
| Source-only (emp.) | 18.6834±0.1840 | 0.1883±0.0052 | 1.6819±0.0233 | 0.0063±0.0002 | 3 |
| RAFM (std) | 0.8503±0.0399 | 0.0088±0.0004 | 1.0071±0.0296 | 0.0060±0.0002 | 3 |
| RAFM (oracle) | 1.0977±0.2629 | 0.0125±0.0020 | 1.0157±0.0356 | 0.0059±0.0002 | 3 |
| Angular RAFM | 1.4734±0.0325 | 0.0149±0.0022 | 0.9492±0.0489 | 0.0055±0.0003 | 3 |

**Angular − std-RAFM:** radial_w1 +0.6231 (worse); ks_stat +0.0061 (worse); sliced_w1 -0.0579 (better); angular_sw_mean -0.0006 (better)

### dim = 256

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 347.6869±0.6234 | 1.0000±0.0000 | 17.8086±0.0808 |  | 3 |
| Source-only (emp.) | 51.0532±2.4129 | 0.3148±0.0114 | 2.8793±0.1365 | 0.0044±0.0001 | 3 |
| RAFM (std) | 2.7032±0.1384 | 0.0169±0.0009 | 1.2412±0.0567 | 0.0037±0.0001 | 3 |
| RAFM (oracle) | 2.6763±0.1771 | 0.0198±0.0040 | 1.2212±0.0368 | 0.0037±0.0001 | 3 |
| Angular RAFM | 2.3969±0.1994 | 0.0119±0.0012 | 1.2508±0.0861 | 0.0037±0.0002 | 3 |

**Angular − std-RAFM:** radial_w1 -0.3063 (better); ks_stat -0.0050 (better); sliced_w1 +0.0097 (worse); angular_sw_mean -0.0000 (better)

## Tail / df sweep — Student-t d16 (E8)

### Overview: std-RAFM vs Angular across df (tail heaviness)

| df | radial_w1 std | radial_w1 ang | sliced_w1 std | sliced_w1 ang | angular_sw std | angular_sw ang |
|---|---|---|---|---|---|---|
| 1.5 | 4.0207 | 5.8336 | 2.8583 | 2.0726 | 0.0360 | 0.0139 |
| 2.0 | 1.0694 | 0.7052 | 0.8843 | 0.4217 | 0.0228 | 0.0119 |
| 3.0 | 0.2378 | 0.1939 | 0.2209 | 0.2033 | 0.0127 | 0.0123 |
| 5.0 | 0.0999 | 0.0947 | 0.1306 | 0.1388 | 0.0110 | 0.0120 |
| 10.0 | 0.1563 | 0.0790 | 0.1203 | 0.1137 | 0.0118 | 0.0109 |
| 50.0 | 0.0761 | 0.0485 | 0.1036 | 0.1062 | 0.0112 | 0.0120 |

### df = 1.5

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 44.2654±2.1672 | 0.6142±0.0234 | 11.1741±1.0479 | 0.1090±0.0100 | 3 |
| Source-only (emp.) | 8.6509±1.7466 | 0.0685±0.0184 | 3.3856±0.4723 | 0.0327±0.0036 | 3 |
| RAFM (std) | 4.0207±1.0604 | 0.0153±0.0012 | 2.8583±0.0858 | 0.0360±0.0014 | 3 |
| RAFM (oracle) | 4.1910±0.2877 | 0.0151±0.0006 | 2.9073±0.2912 | 0.0366±0.0016 | 3 |
| Angular RAFM | 5.8336±0.6460 | 0.0104±0.0024 | 2.0726±0.1255 | 0.0139±0.0010 | 3 |

**Angular − std-RAFM:** radial_w1 +1.8129 (worse); ks_stat -0.0049 (better); sliced_w1 -0.7856 (better); angular_sw_mean -0.0220 (better)

### df = 2.0

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 19.1238±0.5469 | 0.5116±0.0125 | 3.9535±0.4655 | 0.0440±0.0100 | 3 |
| Source-only (emp.) | 3.3450±0.2388 | 0.0676±0.0135 | 1.2266±0.0422 | 0.0271±0.0012 | 3 |
| RAFM (std) | 1.0694±0.2037 | 0.0129±0.0008 | 0.8843±0.0671 | 0.0228±0.0012 | 3 |
| RAFM (oracle) | 1.2717±0.1627 | 0.0128±0.0026 | 0.8878±0.0650 | 0.0219±0.0006 | 3 |
| Angular RAFM | 0.7052±0.1300 | 0.0087±0.0029 | 0.4217±0.0489 | 0.0119±0.0009 | 3 |

**Angular − std-RAFM:** radial_w1 -0.3642 (better); ks_stat -0.0042 (better); sliced_w1 -0.4626 (better); angular_sw_mean -0.0109 (better)

### df = 3.0

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 1.8207±1.1005 | 0.0752±0.0699 | 0.4454±0.1228 | 0.0156±0.0009 | 3 |
| Source-only (emp.) | 0.3540±0.1193 | 0.0223±0.0044 | 0.3942±0.0687 | 0.0179±0.0021 | 3 |
| RAFM (std) | 0.2378±0.0314 | 0.0089±0.0017 | 0.2209±0.0081 | 0.0127±0.0002 | 3 |
| RAFM (oracle) | 0.2604±0.0643 | 0.0136±0.0020 | 0.2287±0.0160 | 0.0131±0.0008 | 3 |
| Angular RAFM | 0.1939±0.0144 | 0.0099±0.0008 | 0.2033±0.0186 | 0.0123±0.0005 | 3 |

**Angular − std-RAFM:** radial_w1 -0.0439 (better); ks_stat +0.0010 (worse); sliced_w1 -0.0176 (better); angular_sw_mean -0.0004 (better)

### df = 5.0

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.4701±0.0394 | 0.0381±0.0060 | 0.1790±0.0243 | 0.0126±0.0014 | 3 |
| Source-only (emp.) | 0.2030±0.0123 | 0.0197±0.0032 | 0.2014±0.0430 | 0.0135±0.0017 | 3 |
| RAFM (std) | 0.0999±0.0051 | 0.0097±0.0013 | 0.1306±0.0211 | 0.0110±0.0009 | 3 |
| RAFM (oracle) | 0.1173±0.0208 | 0.0113±0.0017 | 0.1275±0.0187 | 0.0114±0.0005 | 3 |
| Angular RAFM | 0.0947±0.0060 | 0.0135±0.0029 | 0.1388±0.0205 | 0.0120±0.0008 | 3 |

**Angular − std-RAFM:** radial_w1 -0.0053 (better); ks_stat +0.0038 (worse); sliced_w1 +0.0082 (worse); angular_sw_mean +0.0010 (worse)

### df = 10.0

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.1912±0.0644 | 0.0201±0.0060 | 0.1742±0.0166 | 0.0134±0.0011 | 3 |
| Source-only (emp.) | 0.2119±0.0333 | 0.0242±0.0008 | 0.1413±0.0152 | 0.0127±0.0010 | 3 |
| RAFM (std) | 0.1563±0.0083 | 0.0165±0.0005 | 0.1203±0.0133 | 0.0118±0.0003 | 3 |
| RAFM (oracle) | 0.0805±0.0195 | 0.0093±0.0022 | 0.1155±0.0095 | 0.0118±0.0002 | 3 |
| Angular RAFM | 0.0790±0.0067 | 0.0093±0.0015 | 0.1137±0.0074 | 0.0109±0.0001 | 3 |

**Angular − std-RAFM:** radial_w1 -0.0773 (better); ks_stat -0.0073 (better); sliced_w1 -0.0066 (better); angular_sw_mean -0.0009 (better)

### df = 50.0

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.0816±0.0328 | 0.0116±0.0034 | 0.1481±0.0379 | 0.0137±0.0011 | 3 |
| Source-only (emp.) | 0.1396±0.0067 | 0.0174±0.0015 | 0.1261±0.0272 | 0.0124±0.0017 | 3 |
| RAFM (std) | 0.0761±0.0084 | 0.0144±0.0031 | 0.1036±0.0145 | 0.0112±0.0006 | 3 |
| RAFM (oracle) | 0.0735±0.0372 | 0.0111±0.0047 | 0.1040±0.0167 | 0.0124±0.0009 | 3 |
| Angular RAFM | 0.0485±0.0050 | 0.0103±0.0013 | 0.1062±0.0046 | 0.0120±0.0005 | 3 |

**Angular − std-RAFM:** radial_w1 -0.0276 (better); ks_stat -0.0041 (better); sliced_w1 +0.0026 (worse); angular_sw_mean +0.0008 (worse)

## Anisotropy sweep — Gaussian base d32 (E9)

### Overview: std-RAFM vs Angular across condition number

| kappa | radial_w1 std | radial_w1 ang | sliced_w1 std | sliced_w1 ang | angular_sw std | angular_sw ang |
|---|---|---|---|---|---|---|
| 1 | 0.0102 | 0.0102 | 0.0215 | 0.0222 | 0.0083 | 0.0080 |
| 3 | 0.0373 | 0.0373 | 0.0371 | 0.0493 | 0.0084 | 0.0086 |
| 10 | 0.0743 | 0.0743 | 0.1336 | 0.1248 | 0.0083 | 0.0082 |
| 30 | 0.2287 | 0.2287 | 0.4491 | 0.4377 | 0.0097 | 0.0094 |
| 100 | 0.9802 | 0.9802 | 1.0955 | 1.6211 | 0.0097 | 0.0110 |
| 300 | 3.2472 | 3.2472 | 4.2034 | 6.0534 | 0.0120 | 0.0134 |

### kappa = 1

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.0182±0.0041 | 0.0149±0.0044 | 0.0305±0.0008 | 0.0088±0.0000 | 3 |
| Source-only (emp.) | 0.0152±0.0034 | 0.0148±0.0039 | 0.0303±0.0016 | 0.0090±0.0003 | 3 |
| RAFM (std) | 0.0102±0.0016 | 0.0097±0.0013 | 0.0215±0.0008 | 0.0083±0.0001 | 3 |
| Angular RAFM | 0.0102±0.0016 | 0.0097±0.0013 | 0.0222±0.0020 | 0.0080±0.0004 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 +0.0007 (worse); angular_sw_mean -0.0003 (better)

### kappa = 3

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.5845±0.0926 | 0.1430±0.0246 | 0.1040±0.0118 | 0.0095±0.0005 | 3 |
| Source-only (emp.) | 0.0879±0.0063 | 0.0283±0.0033 | 0.0633±0.0061 | 0.0095±0.0006 | 3 |
| RAFM (std) | 0.0373±0.0032 | 0.0157±0.0025 | 0.0371±0.0007 | 0.0084±0.0001 | 3 |
| Angular RAFM | 0.0373±0.0032 | 0.0157±0.0025 | 0.0493±0.0006 | 0.0086±0.0001 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 +0.0122 (worse); angular_sw_mean +0.0002 (worse)

### kappa = 10

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.3821±0.0232 | 0.0397±0.0049 | 0.2409±0.0148 | 0.0112±0.0009 | 3 |
| Source-only (emp.) | 0.4381±0.0208 | 0.0368±0.0041 | 0.2216±0.0527 | 0.0106±0.0011 | 3 |
| RAFM (std) | 0.0743±0.0048 | 0.0111±0.0019 | 0.1336±0.0232 | 0.0083±0.0003 | 3 |
| Angular RAFM | 0.0743±0.0048 | 0.0111±0.0019 | 0.1248±0.0081 | 0.0082±0.0002 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 -0.0088 (better); angular_sw_mean -0.0001 (better)

### kappa = 30

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 1.9594±0.2431 | 0.0681±0.0078 | 1.0052±0.2193 | 0.0149±0.0021 | 3 |
| Source-only (emp.) | 1.4277±0.1058 | 0.0399±0.0042 | 0.5493±0.1213 | 0.0110±0.0015 | 3 |
| RAFM (std) | 0.2287±0.0323 | 0.0134±0.0015 | 0.4491±0.0786 | 0.0097±0.0008 | 3 |
| Angular RAFM | 0.2287±0.0323 | 0.0134±0.0015 | 0.4377±0.0753 | 0.0094±0.0006 | 3 |

**Angular − std-RAFM:** radial_w1 -0.0000 (better); ks_stat +0.0000 (worse); sliced_w1 -0.0114 (better); angular_sw_mean -0.0003 (better)

### kappa = 100

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 8.4445±1.0445 | 0.0712±0.0109 | 2.7814±0.5505 | 0.0161±0.0026 | 3 |
| Source-only (emp.) | 3.8283±0.2900 | 0.0355±0.0041 | 1.7163±0.2991 | 0.0115±0.0010 | 3 |
| RAFM (std) | 0.9802±0.0765 | 0.0113±0.0012 | 1.0955±0.1028 | 0.0097±0.0001 | 3 |
| Angular RAFM | 0.9802±0.0765 | 0.0113±0.0012 | 1.6211±0.1977 | 0.0110±0.0007 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 +0.5256 (worse); angular_sw_mean +0.0013 (worse)

### kappa = 300

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 37.3715±2.5017 | 0.1017±0.0121 | 13.7835±3.3810 | 0.0298±0.0052 | 3 |
| Source-only (emp.) | 24.0427±4.0488 | 0.0533±0.0042 | 5.7398±1.1616 | 0.0104±0.0017 | 3 |
| RAFM (std) | 3.2472±0.4465 | 0.0148±0.0003 | 4.2034±0.2865 | 0.0120±0.0002 | 3 |
| Angular RAFM | 3.2472±0.4465 | 0.0148±0.0003 | 6.0534±1.1800 | 0.0134±0.0016 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 +1.8500 (worse); angular_sw_mean +0.0014 (worse)

## Ablation singletons

### Student-t d16 df3 (E1 main)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 1.7312±0.3975 | 0.0732±0.0245 | 0.4203±0.0205 | 0.0147±0.0013 | 3 |
| Source-only (emp.) | 0.3981±0.0524 | 0.0181±0.0036 | 0.3090±0.0350 | 0.0151±0.0005 | 3 |
| Source-only (oracle) | 0.4427±0.1361 | 0.0170±0.0044 | 0.3378±0.0431 | 0.0152±0.0013 | 3 |
| RAFM (std) | 0.2610±0.0080 | 0.0127±0.0006 | 0.2685±0.0385 | 0.0134±0.0007 | 3 |
| RAFM (oracle) | 0.2872±0.0711 | 0.0128±0.0019 | 0.2760±0.0388 | 0.0137±0.0014 | 3 |
| Angular RAFM | 0.4669±0.0590 | 0.0130±0.0013 | 0.2499±0.0207 | 0.0113±0.0005 | 3 |

**Angular − std-RAFM:** radial_w1 +0.2059 (worse); ks_stat +0.0003 (worse); sliced_w1 -0.0186 (better); angular_sw_mean -0.0021 (better)

### Student-t d32 df3 (E1)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 9.4155±0.4178 | 0.3373±0.0154 | 1.4366±0.0220 | 0.0217±0.0011 | 3 |
| Source-only (emp.) | 0.9543±0.0438 | 0.0357±0.0021 | 0.7696±0.0903 | 0.0160±0.0016 | 3 |
| Source-only (oracle) | 1.0775±0.2036 | 0.0349±0.0018 | 0.7912±0.1225 | 0.0156±0.0012 | 3 |
| RAFM (std) | 0.3523±0.0544 | 0.0123±0.0019 | 0.4401±0.0943 | 0.0105±0.0013 | 3 |
| RAFM (oracle) | 0.3705±0.1119 | 0.0109±0.0022 | 0.4290±0.0833 | 0.0108±0.0012 | 3 |
| Angular RAFM | 0.4013±0.0426 | 0.0107±0.0020 | 0.3469±0.0328 | 0.0094±0.0003 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0490 (worse); ks_stat -0.0016 (better); sliced_w1 -0.0932 (better); angular_sw_mean -0.0012 (better)

### Gaussian-aniso d16 (E1 control)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.0826±0.0338 | 0.0130±0.0034 | 0.1473±0.0343 | 0.0138±0.0017 | 3 |
| Source-only (emp.) | 0.1134±0.0186 | 0.0178±0.0025 | 0.1495±0.0175 | 0.0144±0.0015 | 3 |
| Source-only (oracle) | 0.1299±0.0019 | 0.0167±0.0011 | 0.1467±0.0213 | 0.0130±0.0005 | 3 |
| RAFM (std) | 0.0549±0.0089 | 0.0124±0.0019 | 0.1040±0.0075 | 0.0120±0.0005 | 3 |
| RAFM (oracle) | 0.0720±0.0050 | 0.0122±0.0024 | 0.1044±0.0085 | 0.0112±0.0007 | 3 |
| Angular RAFM | 0.1024±0.0112 | 0.0200±0.0034 | 0.1265±0.0171 | 0.0127±0.0008 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0475 (worse); ks_stat +0.0076 (worse); sliced_w1 +0.0226 (worse); angular_sw_mean +0.0007 (worse)

### Toy-2D radial-angular (E5)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.0489±0.0197 | 0.0297±0.0081 | 0.0437±0.0062 | 0.0346±0.0070 | 3 |
| Source-only (emp.) | 0.0286±0.0045 | 0.0133±0.0025 | 0.0439±0.0048 | 0.0584±0.0118 | 3 |
| RAFM (std) | 0.3087±0.3796 | 0.2529±0.2249 | 0.8845±0.3148 | 0.7415±0.0722 | 3 |
| RAFM (oracle) | 0.3093±0.3797 | 0.2528±0.2250 | 0.8822±0.3146 | 0.7394±0.0664 | 3 |
| Angular RAFM | 0.1537±0.1192 | 0.0674±0.0703 | 0.7580±0.2229 | 0.5678±0.2176 | 2 |

**Angular − std-RAFM:** radial_w1 -0.1550 (better); ks_stat -0.1856 (better); sliced_w1 -0.1265 (better); angular_sw_mean -0.1737 (better)

### PIV d64 (E10)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.1916±0.0135 | 0.1631±0.0045 | 0.0353±0.0014 | 0.0251±0.0011 | 3 |
| Source-only (emp.) | 0.1177±0.0119 | 0.1158±0.0119 | 0.0252±0.0028 | 0.0233±0.0017 | 3 |
| RAFM (std) | 0.0482±0.0019 | 0.0469±0.0026 | 0.0251±0.0006 | 0.0227±0.0002 | 3 |
| Angular RAFM | 0.0946±0.0278 | 0.0746±0.0081 | 0.0338±0.0030 | 0.0350±0.0017 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0464 (worse); ks_stat +0.0277 (worse); sliced_w1 +0.0087 (worse); angular_sw_mean +0.0124 (worse)

### Finance ff49 — chrono split (main)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 1.4179±0.1132 | 0.2184±0.0090 | 0.1941±0.0180 | 0.0236±0.0025 | 3 |
| Source-only (emp.) | 1.7068±0.0398 | 0.2460±0.0009 | 0.2076±0.0045 | 0.0205±0.0006 | 3 |
| RAFM (std) | 1.4229±0.0401 | 0.2003±0.0028 | 0.1741±0.0026 | 0.0173±0.0003 | 3 |
| Angular RAFM | 1.4229±0.0401 | 0.2003±0.0028 | 0.1721±0.0037 | 0.0172±0.0006 | 3 |
| MSGM | 1.4168±0.0335 | 0.1975±0.0059 | 0.1820±0.0029 | 0.0195±0.0004 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 -0.0019 (better); angular_sw_mean -0.0001 (better)

### Finance ff49 — random split

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.2516±0.0381 | 0.0336±0.0018 | 0.0690±0.0045 | 0.0139±0.0018 | 3 |
| Source-only (emp.) | 0.3512±0.0238 | 0.0658±0.0077 | 0.0688±0.0058 | 0.0134±0.0010 | 3 |
| RAFM (std) | 0.1748±0.0142 | 0.0202±0.0010 | 0.0570±0.0057 | 0.0130±0.0001 | 3 |
| Angular RAFM | 0.1748±0.0142 | 0.0202±0.0010 | 0.0551±0.0031 | 0.0122±0.0004 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 -0.0020 (better); angular_sw_mean -0.0008 (better)

### Weather AU wind — MLP (main)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.1786±0.0311 | 0.0442±0.0102 | 0.0854±0.0075 | 0.0150±0.0001 | 3 |
| Source-only (emp.) | 0.3797±0.0371 | 0.0838±0.0178 | 0.0944±0.0070 | 0.0150±0.0006 | 3 |
| RAFM (std) | 0.0873±0.0048 | 0.0330±0.0018 | 0.0746±0.0045 | 0.0134±0.0005 | 3 |
| Angular RAFM | 0.0873±0.0048 | 0.0330±0.0018 | 0.0795±0.0114 | 0.0141±0.0012 | 3 |
| MSGM | 0.0987±0.0282 | 0.0330±0.0093 | 0.1199±0.0029 | 0.0187±0.0003 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 +0.0049 (worse); angular_sw_mean +0.0007 (worse)

### Weather AU wind — ResMLP 256x4 (big)

| Method | radial_w1 | ks_stat | sliced_w1 | angular_sw_mean | n |
|---|---|---|---|---|---|
| Gaussian FM | 0.3107±0.2108 | 0.0915±0.0441 | 0.1049±0.0081 | 0.0160±0.0007 | 3 |
| Source-only (emp.) | 0.1847±0.1126 | 0.0490±0.0240 | 0.0924±0.0097 | 0.0146±0.0003 | 3 |
| RAFM (std) | 0.0873±0.0048 | 0.0330±0.0018 | 0.0811±0.0088 | 0.0137±0.0009 | 3 |
| Angular RAFM | 0.0873±0.0048 | 0.0330±0.0018 | 0.0764±0.0080 | 0.0137±0.0010 | 3 |

**Angular − std-RAFM:** radial_w1 +0.0000 (worse); ks_stat +0.0000 (worse); sliced_w1 -0.0047 (better); angular_sw_mean -0.0000 (better)

## MSGM — runtime assessment (lowest priority, left missing)

MSGM is present only where it was already run (Finance, Weather main splits). Measured cost on those: **~18,000–20,000 s per seed (~5.3 h)** vs RAFM's ~35 s (~525× slower). Filling the remaining gaps (PIV, Student-t dim sweep ×7, df sweep ×6, d32, gaussian-aniso, toy2d, anisotropy ×6, finance random-split, weather big-model ≈ 26 settings × 3 seeds) would take **≈ 410 GPU-hours (~17 days continuous)**; a single setting is ~16 h. Per the brief (record the estimate, do not block Angular completion, leave missing if prohibitive), MSGM is **left missing**. `run_msgm_real.py` is resumable (checkpoint every 1000 steps) if any specific MSGM cell is later requested.

## Sample-size sensitivity

*No sample-size sweep exists in the retained result folders (E1/E5/E6/E8/E9 vary dim, df, anisotropy — not N). Not run, per the no-new-experiments constraint.*

## E4 — radius drift vs NFE (RK4, tangent projection ON), std-RAFM vs Angular

Drift = mean |‖x_1‖ − ‖x_0‖| over the trajectory; lower = better radial conservation. Sampler-only sweep on a fixed checkpoint (no retrain).

| Dataset | method | drift@nfe4 | drift@nfe20 | drift@nfe48 | drift@nfe100 |
|---|---|---|---|---|---|
| Student-t d16 | std-RAFM | 1.76e+00 | 7.02e-03 | 3.86e-04 | 1.74e-05 |
| Student-t d16 | Angular | 2.51e+00 | 7.74e-03 | 4.96e-04 | 2.28e-05 |
| Toy-2D | std-RAFM | 4.09e+02 | 3.25e+00 | 7.76e-01 | 1.46e-01 |
| Toy-2D | Angular | inf | nan | nan | nan |
| PIV d64 | std-RAFM | 2.14e-01 | 6.95e-04 | 2.11e-05 | 1.53e-06 |
| PIV d64 | Angular | 3.93e-01 | 9.60e-04 | 2.85e-05 | 2.21e-06 |
| Weather | std-RAFM | 6.99e-01 | 4.84e-04 | 1.62e-05 | 9.48e-07 |
| Weather | Angular | 9.68e-01 | 6.43e-04 | 1.87e-05 | 1.01e-06 |

## Numerical stability scan (Angular RAFM)

Per-seed check for NaN in `radial_w1`/`sliced_w1`. Aggregates elsewhere are computed over the finite seeds only; any NaN seed is disclosed here rather than hidden.

**NaN seeds found:**
- E6_dim_scaling/student_t_d2_df3.0_cor/seed_8925 (nan_rate=0.8461999893188477)
- E5_toy2d/toy_radial_angular/seed_77395 (nan_rate=0.000699999975040555)

Interpretation: **confined to d=2** (both `student_t_d2` and the `toy_radial_angular` d=2 toy). At d≥8 every Angular seed is finite. Mechanism: the reconstruction `v=‖x‖·A` amplifies at large radius, and in d=2 the sphere's tangent space is only 1-D, so a rare heavy-tail trajectory can overshoot and diverge under RK4. It is a **numerical limitation of the parameterization in the low-dimensional limit**, not an implementation error — the identical code path is stable at every d≥8, every df, every anisotropy, and on all real datasets. The affected d=2 aggregates are computed over the finite seeds only (n disclosed per row); a NaN-free d=2 result would need higher NFE or radius clipping, which we did **not** apply so the protocol stays identical to the baselines.

## Gap plots

- gap vs dimension: `figs/gap_vs_dim.png`
- gap vs tail heaviness (df): `figs/gap_vs_df.png`
- gap vs anisotropy: `figs/gap_vs_aniso.png`
