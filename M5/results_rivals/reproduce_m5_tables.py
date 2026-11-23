"""Recompute the M5 accuracy tables (Tables 7 and 8, mean ranks, Friedman test, paired counts) for all
fourteen models from the released files: M5/results_m5 (CTR and the six original benchmarks) and
M5/results_rivals (the seven additional benchmarks of Section 3.2).
Run from the repository root: python3 M5/results_rivals/reproduce_m5_tables.py"""
import glob
import pandas as pd
from scipy.stats import friedmanchisquare

MODELS = ['SES', 'Croston', 'SBA', 'TSB', 'ADIDA', 'IMAPA', 'LightGBM', 'XGBoost', 'TwoStage',
          'FNN', 'RNN', 'LSTM', 'GRU', 'CTR']
TOL = 1e-6
pd.set_option('display.width', 200)

o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('M5/results_m5/combined_results_part*.csv.gz'))])
band = o.drop_duplicates('Product_ID').set_index('Product_ID').Sparsity_Level
cl = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('M5/results_m5/clipped_metrics_*.csv.gz'))])
o = o[['Product_ID', 'Model', 'MAE_clipped']].merge(cl, on=['Product_ID', 'Model'])
o.columns = ['Product_ID', 'Model', 'MAE', 'sMAPE', 'MASE', 'RMSSE']
r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('M5/results_rivals/rivals_per_series_part*.csv.gz'))])[['Product_ID', 'Model', 'MAE', 'sMAPE', 'MASE', 'RMSSE']]
d = pd.concat([o, r])
d = d[d.Model.isin(MODELS)]

A = d.pivot(index='Product_ID', columns='Model', values='MAE')[MODELS]
print(f'{len(A):,} series')
t7 = d.groupby('Model')[['MAE', 'sMAPE', 'MASE', 'RMSSE']].median().reindex(MODELS)
t7['mean_rank'] = A.round(6).rank(axis=1).mean()
print('Table 7 - median metrics (clipped):')
print(t7.round({'MAE': 3, 'sMAPE': 2, 'MASE': 3, 'RMSSE': 3, 'mean_rank': 2}).to_string())
fr = friedmanchisquare(*[A[k].values for k in MODELS])
print(f'Friedman test on the MAE ranks: chi2 = {fr.statistic:,.0f}, d.f. = {len(MODELS) - 1}, p = {fr.pvalue:.2g}\n')

print('Table 8 - median MAE by sparsity band:')
print(A.groupby(band.reindex(A.index)).median().T.round(3).to_string(), '\n')

print('Paired per-series MAE: CTR better / worse / tie')
for k in MODELS[:-1]:
    x = A[k] - A['CTR']
    print(f'  CTR vs {k:9s} {int((x > TOL).sum()):6d} / {int((x < -TOL).sum()):6d} / {int((x.abs() <= TOL).sum()):6d}')
