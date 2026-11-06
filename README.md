# Classify-Then-Regress (CTR) — code and results

Code and released results for the manuscript "The Classify-Then-Regress Architecture: Addressing Over-Forecasting in Sparse Retail Time Series via Shared LSTM Representations".

## Contents
- `son2.py` — full pipeline for the primary catalogue: feature construction, chronological 70/15/15 split, CTR and all benchmark models (FNN, RNN, LSTM, CTR_Regression, SES, Croston, SBA), per-SKU threshold search, metrics and summary tables.
- `sb_classify_definitive.py` — Syntetos–Boylan (ADI, CV²) classification of the analytical sample.
- `M5/prep/m5_prepare.py` — selection of the external catalogue: final 1,095 days of the M5 horizon (2013-05-24 to 2016-05-22), ADI > 1.32 and at least 30 non-zero observations (27,853 of 30,490 series).
- `M5/run/ctr_experiment.py`, `merge_results.py`, `run_mac.sh` — the same protocol run on the external catalogue (seed 42), with resumable parallel execution and result merging.
- `M5/prep/skus.csv.gz` — the 27,853 selected M5 series (M5 identifiers, ADI, zero rate).
- `M5/sonuc_m5/combined_results_part1.csv.gz` and `_part2.csv.gz` — per-series metrics of all eight models on the external catalogue (split by model to respect file-size limits; concatenate the two to obtain the full table); `tablo*.csv` — summary tables.
- `figures/` — scripts generating the manuscript figures.
- `analysis/allzero_analysis.py` — always-zero reference analysis.

## Data
- External catalogue: M5 Forecasting – Accuracy, https://www.kaggle.com/competitions/m5-forecasting-accuracy. Place `sales_train_evaluation.csv` and `calendar.csv` in `M5/`, then run `M5/prep/m5_prepare.py` and `M5/run/run_mac.sh`.
- Primary catalogue: proprietary transaction data, not included (`Veri/` is empty). It may be made available by the corresponding author on reasonable request, subject to the written permission of the data provider. `son2.py` expects `Veri/filtered_data.csv` with columns `URUNID, Date, Satis Adet` (product id, date, daily units sold).
