# Primary catalogue: per-SKU results and SKU characteristics

The primary catalogue consists of proprietary daily sales data of a wholesale retailer and cannot be shared. This folder releases, for its 4,256 SKUs, everything except the sales quantities themselves, so that the accuracy tables reported for the primary catalogue can be recomputed.

SKU identifiers are randomly re-coded (`SKU_0001` … `SKU_4256`); they do not correspond to the retailer's product codes.

## Files
- `per_sku_metrics.csv.gz` — one row per SKU and model: `MAE`, `sMAPE`, `MASE`, `RMSSE` on the test partition after non-negativity clipping (Section 3.4), and `MAE_unclipped` (Table 6). Models: CTR, FNN, RNN, LSTM, GRU, SES, Croston, SBA, TSB, ADIDA, IMAPA, LightGBM, XGBoost, TwoStage (the two-stage pipeline), and the all-zero reference forecast of Table 2. MASE and RMSSE are scaled by the in-sample one-step naive error of the training partition.
- `sku_characteristics.csv.gz` — one row per SKU: series length, number of non-zero days, zero rate over the full series and over the test partition, ADI, CV² and Syntetos–Boylan class (full series), and the sparsity band used in Table 3.
- `reproduce_primary_tables.py` — recomputes Tables 2, 3 and 6, the mean ranks and Friedman test of Table 2, the paired counts and Holm-corrected Wilcoxon p-values of Table 4, and the winner count of Section 4.3 from the two files above.
