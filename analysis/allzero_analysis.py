"""R2 Madde 1: always-zero baseline under the identical evaluation protocol.
Read-only on existing results. Inputs: per-SKU published metrics (combined_results.csv) and
point predictions. All-zero forecast: yhat_t = 0 for every test point.
MAE_az = mean(y_test); MASE_az = MAE_az / s_i, where s_i = mean|y_t - y_{t-1}| over the SKU's
training window, recovered exactly as s_i = MAE_m / MASE_m from the published row of any model m
(same scale for all models). RMSSE analogously with RMSE/RMSSE.
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats

ROOT = sys.argv[1]; OUT = sys.argv[2]
import os
CLIP = os.environ.get('CLIP') == '1'   # manuscript §3: all predictions clipped max(0,yhat) before metrics
rng = np.random.default_rng(20261105)
TOL = 1e-9
CFG = {
 'BTM': dict(comb=f'{ROOT}/Son2.py_sonuclar/combined_results.csv',
             pred=f'{ROOT}/Son2.py_sonuclar/ctr_predictions.csv', pcol='CTR_Predictions', ctr='CTR'),
 'M5':  dict(comb=f'{ROOT}/M5/sonuc_m5/combined_results.csv',
             pred=f'{ROOT}/M5/sonuc_m5/predictions.csv', pcol='CTR_pred', ctr='CTR'),
}

def boot_ci(d, fn, B=2000):
    n = len(d); vals = np.empty(B)
    for b in range(B): vals[b] = fn(d[rng.integers(0, n, n)])
    return np.percentile(vals, [2.5, 97.5])

def paired(a_ctr, b_az, label):
    d = a_ctr - b_az                       # >0 : CTR worse than all-zero
    d = np.where(np.abs(d) < TOL, 0.0, d)  # float round-off (e.g. CTR forecast identically 0) -> exact tie
    nz = d[d != 0]
    w = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='approx')
    wp = stats.wilcoxon(d, zero_method='pratt', alternative='two-sided', method='approx')
    # matched-pairs rank-biserial correlation on non-zero differences (sign: + => CTR better)
    r = stats.rankdata(np.abs(nz)); rb = (r[nz < 0].sum() - r[nz > 0].sum()) / r.sum()
    q = np.percentile(d, [5, 10, 25, 50, 75, 90, 95])
    return {'metric': label, 'n_sku': len(d), 'ctr_better': int((d < 0).sum()), 'az_better': int((d > 0).sum()),
            'ties': int((d == 0).sum()), 'median_ctr': float(np.median(a_ctr)), 'median_az': float(np.median(b_az)),
            'mean_d': float(d.mean()), 'mean_d_ci': boot_ci(d, np.mean).tolist(),
            'median_d': float(np.median(d)), 'median_d_ci': boot_ci(d, np.median).tolist(),
            'median_d_nonzero': float(np.median(nz)) if len(nz) else np.nan,
            'q05_10_25_50_75_90_95': q.tolist(),
            'wilcoxon_p_wilcox': float(w.pvalue), 'wilcoxon_p_pratt': float(wp.pvalue),
            'rank_biserial_ctr_better_pos': float(rb)}

report = {}
for ds, c in CFG.items():
    comb = pd.read_csv(c['comb'])
    pr = pd.read_csv(c['pred'], usecols=['Product_ID', 'y_test', c['pcol']])
    pr = pr.rename(columns={c['pcol']: 'p'})
    if CLIP: pr['p'] = pr.p.clip(lower=0)
    pr['ae'] = (pr.y_test - pr.p).abs()
    allzero_ctr_sku = int(pr.groupby('Product_ID').p.apply(lambda v: (v == 0).all()).sum())
    g = pr.groupby('Product_ID', sort=False)
    per = pd.DataFrame({'n': g.size(), 'mae_ctr_re': g.ae.mean(), 'mae_az': g.y_test.mean(),
                        'rmse_az': g.y_test.apply(lambda s: np.sqrt((s**2).mean()))})
    # scales from published rows (any model with finite, non-zero MASE)
    cc = comb.copy()
    cc['s_mase'] = cc.MAE / cc.MASE; cc['s_rmsse'] = cc.RMSE / cc.RMSSE
    ok1 = np.isfinite(cc.s_mase) & (cc.MASE > 0); ok2 = np.isfinite(cc.s_rmsse) & (cc.RMSSE > 0)
    s1 = cc[ok1].groupby('Product_ID').s_mase.median(); s2 = cc[ok2].groupby('Product_ID').s_rmsse.median()
    s1_spread = cc[ok1].groupby('Product_ID').s_mase.agg(lambda v: (v.max()-v.min())/v.median()).max()
    ctr = comb[comb.Model == c['ctr']].set_index('Product_ID')
    per = per.join(ctr[['MAE', 'MASE', 'RMSE', 'RMSSE', 'Sparsity_Level']], how='inner')
    per = per.join(s1.rename('s1')).join(s2.rename('s2'))
    if CLIP:   # replace published (raw) CTR metrics by clipped ones, same scales
        gc = pr.groupby('Product_ID', sort=False)
        per['MAE'] = per.mae_ctr_re
        per['RMSE'] = gc.apply(lambda x: np.sqrt(((x.y_test - x.p)**2).mean()))
        per['MASE'] = per.MAE / per.s1; per['RMSSE'] = per.RMSE / per.s2
    per['mase_az'] = np.where(per.mae_az == 0, 0.0, per.mae_az / per.s1)
    per['rmsse_az'] = np.where(per.rmse_az == 0, 0.0, per.rmse_az / per.s2)
    chk = float((per.mae_ctr_re - per.MAE).abs().max())
    # ---- table-1 style medians: all models + all-zero
    med = comb.groupby('Model')[['MAE', 'MASE', 'RMSSE']].median()
    if CLIP: med.loc['CTR (clipped)'] = [per.MAE.median(), np.nanmedian(per.MASE), np.nanmedian(per.RMSSE)]
    med.loc['All-zero'] = [per.mae_az.median(), np.nanmedian(per.mase_az), np.nanmedian(per.rmsse_az)]
    med = med.sort_values('MAE')
    # ---- paired
    res = [paired(per.MAE.values, per.mae_az.values, 'MAE')]
    m = np.isfinite(per.mase_az) & np.isfinite(per.MASE)
    res.append(paired(per.MASE.values[m], per.mase_az.values[m], 'MASE'))
    # ---- pooled (all test points)
    pooled = {'pooled_MAE_ctr': float(pr.ae.mean()), 'pooled_MAE_az': float(pr.y_test.mean()), 'n_points': len(pr)}
    # ---- conditional: y>0 and y=0 points
    pos = pr[pr.y_test > 0]; zer = pr[pr.y_test == 0]
    gp = pos.groupby('Product_ID')
    cpos = pd.DataFrame({'ctr': gp.ae.mean(), 'az': gp.y_test.mean()})
    cond = {'n_pos_points': len(pos), 'n_zero_points': len(zer), 'n_sku_with_pos': len(cpos),
            'pooled_MAE_pos_ctr': float(pos.ae.mean()), 'pooled_MAE_pos_az': float(pos.y_test.mean()),
            'median_sku_MAE_pos_ctr': float(cpos.ctr.median()), 'median_sku_MAE_pos_az': float(cpos.az.median()),
            'pooled_MAE_zero_ctr': float(zer.ae.mean()), 'pooled_MAE_zero_az': 0.0,
            'abs_error_total_ctr': float(pr.ae.sum()), 'abs_error_on_pos_ctr': float(pos.ae.sum()),
            'abs_error_on_zero_ctr': float(zer.ae.sum()), 'abs_error_total_az': float(pos.y_test.sum()),
            'reduction_on_pos_by_ctr': float(pos.y_test.sum() - pos.ae.sum()),
            'added_on_zero_by_ctr': float(zer.ae.sum()),
            'pos_points_ctr_closer': int((pos.ae < pos.y_test).sum()), 'pos_points_ctr_farther': int((pos.ae > pos.y_test).sum()),
            'pos_points_equal': int((pos.ae == pos.y_test).sum())}
    cond['paired_pos_MAE'] = paired(cpos.ctr.values, cpos.az.values, 'MAE|y>0')
    # event (>=0.5 rule)
    ev = pr.p >= 0.5; act = pr.y_test > 0
    tp = int((ev & act).sum()); fp = int((ev & ~act).sum()); fn = int((~ev & act).sum())
    cond['event_ctr'] = {'TP': tp, 'FP': fp, 'FN': fn, 'recall': tp/(tp+fn), 'precision': tp/(tp+fp) if tp+fp else np.nan}
    # by sparsity bin
    per['d'] = per.MAE - per.mae_az
    per.loc[per.d.abs() < TOL, 'd'] = 0.0
    byb = per.groupby('Sparsity_Level').agg(n=('d', 'size'), ctr_better=('d', lambda v: int((v < 0).sum())),
            az_better=('d', lambda v: int((v > 0).sum())), ties=('d', lambda v: int((v == 0).sum())),
            median_mae_ctr=('MAE', 'median'), median_mae_az=('mae_az', 'median'), mean_d=('d', 'mean'))
    per.to_csv(f'{OUT}/{ds}_per_sku_allzero.csv')
    med.to_csv(f'{OUT}/{ds}_median_table_with_allzero.csv'); byb.to_csv(f'{OUT}/{ds}_by_sparsity.csv')
    report[ds] = {'n_sku': len(per), 'check_max_abs_diff_CTR_MAE_recomputed_vs_published': chk,
                  'scale_max_rel_spread_across_models': float(s1_spread),
                  'n_sku_ctr_forecast_identically_zero': allzero_ctr_sku,
                  'n_sku_scale_1e-10': int((per.s1 < 1e-9).sum()), 'n_sku_no_scale': int(per.s1.isna().sum()), 'medians': med.round(6).to_dict('index'),
                  'paired': res, 'pooled': pooled, 'conditional': cond, 'by_sparsity': byb.round(6).reset_index().to_dict('records')}
json.dump(report, open(f'{OUT}/madde1_report.json', 'w'), indent=1, default=float)
print(json.dumps(report, indent=1, default=float))
