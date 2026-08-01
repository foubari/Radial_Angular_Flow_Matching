# Dataset citations / attributions (for OpenReview & camera-ready)

## 1. Finance — Kenneth French 49 Industry Portfolios (daily, value-weighted)
- Source (data): Kenneth R. French Data Library, Tuck School of Business, Dartmouth.
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  Direct file: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip
- Terms: free for academic/research use; French requests that the source be acknowledged. No formal license file.
- Methodology / industry definitions (cite the paper): Fama & French (1997).

```bibtex
@misc{frenchdatalib,
  author       = {French, Kenneth R.},
  title        = {Data Library --- 49 Industry Portfolios (Daily)},
  howpublished = {\url{https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html}},
  note         = {Tuck School of Business, Dartmouth College. Accessed 2026-07.}
}
@article{fama1997industry,
  author  = {Fama, Eugene F. and French, Kenneth R.},
  title   = {Industry costs of equity},
  journal = {Journal of Financial Economics},
  volume  = {43}, number = {2}, pages = {153--193}, year = {1997},
  doi     = {10.1016/S0304-405X(96)00896-3}
}
```
Suggested sentence: "Daily value-weighted returns of the 49 industry portfolios from the Kenneth R. French Data Library (French, Dartmouth; industry definitions of Fama and French, 1997)."

## 2. Weather — WeatherBench 2 (ERA5), 10 m wind over Australia
Two things to cite: the benchmark (WeatherBench 2) AND the underlying reanalysis (ERA5).

- Benchmark / data host (public GCS `gs://weatherbench2`): Rasp et al. (2024).
- Underlying data: ERA5 reanalysis, Hersbach et al. (2020), from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).

```bibtex
@article{rasp2024weatherbench2,
  author  = {Rasp, Stephan and Hoyer, Stephan and Merose, Alexander and others},
  title   = {{WeatherBench 2}: A benchmark for the next generation of data-driven global weather models},
  journal = {Journal of Advances in Modeling Earth Systems (JAMES)},
  volume  = {16}, number = {6}, pages = {e2023MS004019}, year = {2024},
  doi     = {10.1029/2023MS004019}, note = {arXiv:2308.15560}
}
@article{hersbach2020era5,
  author  = {Hersbach, Hans and Bell, Bill and Berrisford, Paul and others},
  title   = {The {ERA5} global reanalysis},
  journal = {Quarterly Journal of the Royal Meteorological Society},
  volume  = {146}, number = {730}, pages = {1999--2049}, year = {2020},
  doi     = {10.1002/qj.3803}
}
```
**Mandatory Copernicus/ERA5 attribution** (put in the paper): 
"Contains modified Copernicus Climate Change Service information 2024. Neither the European Commission nor ECMWF is responsible for any use of the Copernicus information." ERA5 data were accessed via WeatherBench 2 (Rasp et al., 2024).
Links: WeatherBench2 https://weatherbench2.readthedocs.io ; ERA5 (C3S CDS) https://cds.climate.copernicus.eu ; ERA5 doc https://doi.org/10.24381/cds.adbb2d47

## 3. PIV (already in the paper) — flow over a circular cylinder, Re=3900
- DOI: 10.57745/DHJXM6 (Recherche Data Gouv / Dataverse). Already cited in the submission; reuse that entry.

```bibtex
@dataset{piv_re3900,
  title     = {Non-time-resolved {PIV} dataset of flow over a circular cylinder at {Re}=3900},
  publisher = {Recherche Data Gouv},
  doi       = {10.57745/DHJXM6}, year = {2023}
}
```
