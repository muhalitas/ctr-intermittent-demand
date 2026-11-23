# Classify-Then-Regress (CTR) — code and results

Code and released results for the manuscript "The Classify-Then-Regress Architecture: Addressing Over-Forecasting in Sparse Retail Time Series via an Occurrence-Gated LSTM".

CTR forecasts intermittent demand with a single LSTM encoder shared by two heads: a classification head that estimates whether demand occurs on a given day, and a regression head that estimates its size. Both heads are trained jointly (loss = 0.5 · binary cross-entropy + 0.5 · a weighted absolute-error loss that gives non-zero days three times the weight). The final forecast is the regression output on days where the occurrence probability exceeds a per-SKU threshold τ\*, selected on the validation split, and zero otherwise.

## Contents
- `son2.py` — full pipeline for the primary catalogue: feature construction, chronological 70/15/15 split, CTR and all benchmark models (FNN, RNN, LSTM, SES, Croston, SBA; the scripts also train a legacy single-output variant, `CTR_Regression`, which is not part of the comparison reported in the paper), per-SKU threshold search, metrics and summary tables.
- `sb_classify_definitive.py` — Syntetos–Boylan (ADI, CV²) classification of the analytical sample.
- `M5/prep/m5_prepare.py` — selection of the external catalogue: final 1,095 days of the M5 horizon (2013-05-24 to 2016-05-22), ADI > 1.32 and at least 30 non-zero observations (27,853 of 30,490 series).
- `M5/run/ctr_experiment.py`, `merge_results.py`, `run.sh` — the same protocol run on the external catalogue (seed 42), with resumable parallel execution and result merging.
- `M5/prep/skus.csv.gz` — the 27,853 selected M5 series (M5 identifiers, ADI, zero rate).
- `M5/results_m5/combined_results_part1.csv.gz` to `_part4.csv.gz` — per-series metrics of CTR and the six benchmarks on the external catalogue (split by model to respect file-size limits; concatenate the four parts to obtain the full table); `clipped_metrics_neural.csv.gz` and `clipped_metrics_classical.csv.gz` — per-series sMAPE, MASE and RMSSE computed after the non-negativity clip of Section 3.4, i.e. the values summarized in Table 7 (`MAE_clipped` is in the four parts); `tableA`–`tableC` — summary tables.
- `M5/run/rivals.py`, `merge_rivals.py`, `run_rivals.sh` — the seven additional benchmarks of Section 3.2 (TSB, ADIDA, IMAPA, LightGBM, XGBoost, the two-stage pipeline and GRU; seed 42) under the same preparation, features, split, windows and non-negativity clip as the main experiment (`bash M5/run/run_rivals.sh m5`, after steps 2–3 of "Reproducing the M5 experiment"). The script also writes variants that are not reported in the paper (`_trva`, `_roll`, `LightGBM_tweedie`, `TwoStage_F1`).
- `results_primary/per_sku_metrics.csv.gz` and `M5/results_rivals/rivals_per_series_part1.csv.gz` to `_part4.csv.gz` — per-SKU and per-series results of the seven additional benchmarks (the M5 file is split by model to respect file-size limits); `M5/results_rivals/reproduce_m5_tables.py` recomputes Tables 7 and 8, the mean ranks and the Friedman test for all fourteen models on the M5 catalogue (run from the repository root).
- `M5/run/ctr_ablation.py`, `ablation_helper.py`, `run_ablation.sh` — ablation of Section 4.8 (CTR_full, CTR_nogate, CTR_reghead; seed 42) on the primary catalogue (all 4,256 SKUs; results released, data proprietary) and on 5,000 M5 series drawn with seed 42 (runnable with `bash M5/run/run_ablation.sh` after steps 2–3 of "Reproducing the M5 experiment").
- `M5/results_ablation/primary/` and `M5/results_ablation/m5/` — per-SKU/per-series ablation results (`ablation_per_sku.csv.gz`, `ablation_per_series.csv.gz`), summaries (`ablation_summary.csv`: medians of Table 10; its `zeroFP` and `recall` columns are per-item means, whereas Table 10 reports the rates pooled over all item-days, which can be recomputed from the per-item files) and paired tests (`ablation_paired.csv`, Table 11); primary SKU codes use the same random re-coding as `results_primary/`; `series_ids.csv` lists the 5,000 M5 series.

## Data
- External catalogue: M5 Forecasting – Accuracy, https://www.kaggle.com/competitions/m5-forecasting-accuracy.
- Primary catalogue: proprietary transaction data, not included (`Veri/` is empty). It may be made available by the corresponding author on reasonable request, subject to the written permission of the data provider. `son2.py` expects `Veri/filtered_data.csv` with columns `URUNID, Date, Satis Adet` (product id, date, daily units sold).

## Reproducing the M5 experiment
1. Install the requirements (Python 3.9 or later): `pip install -r requirements.txt`
2. Download `sales_train_evaluation.csv` and `calendar.csv` from the M5 competition page and place them in `M5/`.
3. Prepare the data: `python3 M5/prep/m5_prepare.py`
4. Run the experiment: `bash M5/run/run.sh` (set `WORKERS` to change the number of parallel workers; an interrupted run resumes when the same command is repeated). For a quick check on a few series: `python3 M5/run/ctr_experiment.py --out M5/results_test --limit 5 --workers 1`
5. Merge the results and print the summary tables: `bash M5/run/run.sh merge`

`son2.py` and `merge_results.py` report sMAPE, MASE and RMSSE before the non-negativity clip of Section 3.4 (`MAE_clipped` is the clipped MAE) and include the legacy `CTR_Regression` variant. The values reported in the paper are the clipped ones released in `results_primary/` and in `M5/results_m5/clipped_metrics_*.csv.gz`; the released `tableA`–`tableC` cover the seven models of the paper.

With 8 parallel workers on an Apple M4 machine the experiment took about 3.5 seconds per series, i.e. about 27 hours for all series.

## Additional benchmarks
- Library versions of the reported run (both catalogues, Apple Mac mini M4): Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, statsforecast 2.0.3, statsmodels 0.14.6, LightGBM 4.6.0, XGBoost 2.1.4, TensorFlow 2.20.0. Computation per series was about 9 s, almost all of it for the GRU.
- Fall-backs in `rivals.py`: when an item has no positive demand in its training partition (131 primary SKUs and 217 M5 series), LightGBM, XGBoost and the two-stage pipeline forecast zero; when it has a single positive training day (146 SKUs and 18 series), the magnitude stage of the two-stage pipeline uses the value of that day.
- Training-partition eligibility check of Section 4.7: after the first seven records of a series are dropped, the training partition is the first `m − 2·int(0.15·m)` of the remaining `m` records; a series is eligible when the number of training records divided by the number of non-zero training records exceeds 1.32 (4,143 of the 4,256 primary SKUs; for M5, where at least 30 non-zero training records are also required, 26,551 of the 30,490 series).

## License
MIT
