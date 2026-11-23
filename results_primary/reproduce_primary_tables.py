"""Recompute the primary-catalogue accuracy tables from the released per-SKU files.

Tables 2, 3, 4 (paired counts and Holm-corrected one-sided Wilcoxon p-values; bootstrap intervals omitted)
and 6, the mean ranks and Friedman test of Table 2, and the item-by-item winner count of Section 4.3.
Two forecasts are counted as tied when their MAE differs by less than 1e-6."""
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

MODELS = ['SES', 'Croston', 'SBA', 'TSB', 'ADIDA', 'IMAPA', 'LightGBM', 'XGBoost', 'TwoStage',
          'FNN', 'RNN', 'LSTM', 'GRU', 'CTR']
RIVALS = MODELS[:-1]
TOL = 1e-6
pd.set_option('display.width', 200)

m = pd.read_csv('per_sku_metrics.csv.gz')
c = pd.read_csv('sku_characteristics.csv.gz')
order = MODELS + ['All-zero (reference)']

print('Table 2 - median metrics (clipped):')
t2 = m.groupby('model')[['MAE', 'sMAPE', 'MASE', 'RMSSE']].median().reindex(order)
w = m.pivot(index='sku', columns='model', values='MAE')
A = w[MODELS]
t2['mean_rank'] = A.round(6).rank(axis=1).mean()          # MAE rounded to six decimals, ties averaged
print(t2.round({'MAE': 3, 'sMAPE': 2, 'MASE': 3, 'RMSSE': 3, 'mean_rank': 2}).to_string())
fr = friedmanchisquare(*[A[k].values for k in MODELS])
print(f'Friedman test on the MAE ranks: chi2 = {fr.statistic:,.0f}, d.f. = {len(MODELS) - 1}, p = {fr.pvalue:.2g}\n')

print('Table 6 - median MAE before and after clipping:')
t6 = pd.DataFrame({'raw': m.groupby('model')['MAE_unclipped'].median(), 'clipped': m.groupby('model')['MAE'].median()}).reindex(MODELS)
t6['reduction_%'] = (100 * (1 - t6.clipped / t6.raw)).where(lambda v: v.abs() >= 0.05, 0.0)
print(t6.round({'raw': 3, 'clipped': 3, 'reduction_%': 1}).to_string(), '\n')

x = m.merge(c[['sku', 'sparsity_band_table3']], on='sku')
print('Table 3 - median MAE by sparsity band:')
t3 = x.pivot_table(index='model', columns='sparsity_band_table3', values='MAE', aggfunc='median').reindex(MODELS)
print(t3.round(3).to_string(), '\n')

print('Table 4 - paired per-SKU MAE: CTR better / worse / tie, median difference, Holm-corrected p (one-sided):')
p = {k: wilcoxon(A[k], A['CTR'], alternative='greater').pvalue for k in RIVALS}
holm, run = {}, 0.0
for i, (k, v) in enumerate(sorted(p.items(), key=lambda z: z[1])):
    run = max(run, min(1.0, (len(p) - i) * v)); holm[k] = run
for k in RIVALS:
    d = A[k] - A['CTR']
    print(f'  CTR vs {k:9s} {int((d > TOL).sum()):5d} / {int((d < -TOL).sum()):5d} / {int((d.abs() <= TOL).sum()):5d}'
          f'   median {d.median():.4f}   p = {holm[k]:.2g}')

best = A.min(axis=1)
joint = A['CTR'] <= best + TOL
print(f'\nCTR has the lowest or joint-lowest MAE of the {len(MODELS)} models on {int(joint.sum()):,} of {len(A):,} SKUs ({100 * joint.mean():.1f}%)')
