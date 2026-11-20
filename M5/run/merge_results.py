#!/usr/bin/env python3
"""Merge the result parts and produce all summary tables.
Usage: python3 merge_results.py RESULTS_DIR

Outputs:
  Table A  overall median performance (raw + clipped)
  Table B  median MAE by sparsity band
  Table C  paired Wilcoxon (MAE) + Holm-Bonferroni
  Table D  mechanism diagnostics: zero-FP / recall / precision / F1
  Table E  raw vs clipped MAE
  Table F  paired Wilcoxon (F1) + Spearman(dF1, dMAE)
  Table G  threshold sensitivity: per-SKU vs fixed threshold
"""
import sys, os, glob, argparse
import numpy as np, pandas as pd

# CTR_Regression: NOTE: legacy baseline kept for completeness of the original run; it is not part of the comparison reported in the paper. The controlled ablation of Section 4.8 is in M5/run/ctr_ablation.py.
MODELS = ['SES','Croston','SBA','CTR_Regression','FNN','RNN','LSTM','CTR']
NEURAL = ['CTR_Regression','FNN','RNN','LSTM','CTR']
BINS   = [0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0001]
LABELS = ['[0.0, 0.2)','[0.2, 0.3)','[0.3, 0.4)','[0.4, 0.5)',
          '[0.5, 0.6)','[0.6, 0.7)','[0.7, 0.8)','[0.8, 1.0)']

ap = argparse.ArgumentParser()
ap.add_argument('outdir'); ap.add_argument('--dataset', choices=['m5'], default='m5')
a = ap.parse_args()
OUT = os.path.abspath(a.outdir)
HERE = os.path.dirname(os.path.abspath(__file__))

met = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/parts/met_*.csv'))], ignore_index=True)
prd = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/parts/prd_*.csv'))], ignore_index=True)
met = met.drop_duplicates(['Product_ID','Model'])
prd = prd.drop_duplicates(['Product_ID'] + [c for c in prd.columns if c != 'Product_ID'])
print(f'{met.Product_ID.nunique()} SKUs  |  {len(prd)} prediction rows\n')

# --- sparsity: as in the original protocol, from the FULL series (not the test split) ---
sp_full = None
if a.dataset == 'm5':
    sk = os.path.join(HERE, '..', 'prep', 'skus.csv')
    if os.path.exists(sk):
        s = pd.read_csv(sk)[['ProductID','zero_rate','ADI','cat_id','store_id','state_id']]
        sp_full = s.set_index('ProductID')
        print('sparsity source: skus.csv (full series) - same as original protocol')
if sp_full is None:
    print('WARNING: skus.csv not found, sparsity computed from the test split (differs from original)')
    sp_full = prd.groupby('Product_ID').y_test.apply(lambda v: (v == 0).mean()).rename('zero_rate').to_frame()

met = met.merge(sp_full[['zero_rate']], left_on='Product_ID', right_index=True, how='left')
met['Sparsity_Level'] = pd.cut(met.zero_rate, bins=BINS, labels=LABELS, include_lowest=True, right=False)

# --- clipped MAE + per-SKU classification diagnostics ---
rows = []
for pid, g in prd.groupby('Product_ID'):
    yt = g.y_test.values; act = yt > 0
    for m in MODELS:
        yp = g[f'{m}_pred'].values
        ypc = np.clip(yp, 0, None)
        r = dict(Product_ID=pid, Model=m, MAE_clipped=np.mean(np.abs(yt - ypc)))
        if m in NEURAL:
            pos = ypc >= 0.5
            tp = int((pos & act).sum()); fp = int((pos & ~act).sum())
            fn = int((~pos & act).sum()); tn = int((~pos & ~act).sum())
            prec = tp/(tp+fp) if tp+fp else np.nan
            rec  = tp/(tp+fn) if tp+fn else np.nan
            r.update(zeroFP = fp/(fp+tn) if fp+tn else np.nan, recall=rec, precision=prec,
                     F1 = 2*prec*rec/(prec+rec) if prec and rec and (prec+rec) else 0.0)
        rows.append(r)
met = met.merge(pd.DataFrame(rows), on=['Product_ID','Model'], how='left')
met.to_csv(f'{OUT}/combined_results.csv', index=False)
prd.to_csv(f'{OUT}/predictions.csv', index=False)

print('\n=== TABLE A · overall median performance ===')
t1 = met.pivot_table(index='Model', values=['MAE','MAE_clipped','sMAPE','MASE','RMSSE'],
                     aggfunc='median').reindex(MODELS)
print(t1[['MAE','MAE_clipped','sMAPE','MASE','RMSSE']].round(3).to_string())

print('\n=== TABLE B · median MAE by sparsity band (clipped) ===')
t2 = met.pivot_table(index='Sparsity_Level', columns='Model', values='MAE_clipped',
                     aggfunc='median', observed=False).reindex(LABELS)
t2.insert(0, 'N', met[met.Model=='CTR'].groupby('Sparsity_Level', observed=False).size().reindex(LABELS))
print(t2[['N']+MODELS].round(3).to_string())

print('\n=== TABLE D · mechanism diagnostics (pooled, at y>=0.5) ===')
d4 = []
for m in NEURAL:
    g = prd; yt = g.y_test.values; act = yt > 0
    yp = np.clip(g[f'{m}_pred'].values, 0, None); pos = yp >= 0.5
    tp=(pos&act).sum(); fp=(pos&~act).sum(); fn=(~pos&act).sum(); tn=(~pos&~act).sum()
    pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
    d4.append(dict(Model=m, Zero_FP=fp/(fp+tn), Recall=rc, Precision=pr,
                   F1=2*pr*rc/(pr+rc) if pr+rc else 0))
print(pd.DataFrame(d4).set_index('Model').round(3).to_string())

print('\n=== TABLE E · effect of clipping (median MAE) ===')
t5 = t1[['MAE','MAE_clipped']].copy()
t5['Reduction_%'] = (100*(t1.MAE - t1.MAE_clipped)/t1.MAE).round(1)
print(t5.round(3).to_string())

try:
    from scipy import stats
    def holm(p):
        k=len(p); o=np.argsort(p); adj=np.empty(k)
        run=0
        for i,j in enumerate(o):
            run=max(run, p[j]*(k-i)); adj[j]=min(run,1.0)
        return adj

    print('\n=== TABLE C · paired Wilcoxon, clipped MAE (Holm-Bonferroni) ===')
    pv = met.pivot(index='Product_ID', columns='Model', values='MAE_clipped').dropna()
    r=[]
    for m in [x for x in MODELS if x!='CTR']:
        d = pv[m].values - pv['CTR'].values
        _, p = stats.wilcoxon(d, alternative='greater')
        r.append(dict(Comparison=f'CTR vs {m}', Wins=int((d>0).sum()), Losses=int((d<0).sum()),
                      Ties=int((d==0).sum()), Median_improvement=float(np.median(d)), p=p))
    w=pd.DataFrame(r); w['p_holm']=holm(w.p.values)
    print(w.round(6).to_string(index=False)); w.to_csv(f'{OUT}/tableC_wilcoxon_mae.csv', index=False)

    print('\n=== TABLE F · paired Wilcoxon, F1 ===')
    pf = met[met.Model.isin(NEURAL)].pivot(index='Product_ID', columns='Model', values='F1').dropna()
    pm = met[met.Model.isin(NEURAL)].pivot(index='Product_ID', columns='Model', values='MAE_clipped').loc[pf.index]
    r=[]
    for m in [x for x in NEURAL if x!='CTR']:
        d = pf['CTR'].values - pf[m].values          # CTR - competitor  (negative = lower F1 for CTR)
        try: _, p2 = stats.wilcoxon(d, alternative='two-sided')
        except ValueError: p2 = np.nan
        r.append(dict(Comparison=f'CTR - {m}', CTR_higher=int((d>0).sum()), CTR_lower=int((d<0).sum()),
                      Median_dF1=float(np.median(d)), p=p2))
    w6=pd.DataFrame(r); w6['p_holm']=holm(np.nan_to_num(w6.p.values, nan=1.0))
    print(w6.round(6).to_string(index=False)); w6.to_csv(f'{OUT}/tableF_wilcoxon_f1.csv', index=False)

    print('\n--- relationship between dF1 and dMAE ---')
    for m in [x for x in NEURAL if x!='CTR']:
        dF1  = pf['CTR'].values - pf[m].values
        dMAE = pm[m].values - pm['CTR'].values        # positive = better MAE for CTR
        ok = np.isfinite(dF1) & np.isfinite(dMAE)
        rho, pr = stats.spearmanr(dF1[ok], dMAE[ok])
        print(f'  CTR vs {m:<16s} Spearman rho = {rho:+.3f}  (p = {pr:.2e})  '
              f'SKUs with lower F1 but better MAE: {int(((dF1<0)&(dMAE>0)).sum())}/{ok.sum()}')

    print('\n=== per-SKU threshold distribution ===')
    ta = met[met.Model=='CTR'][['Product_ID','Best_Alpha_CTR','zero_rate']].dropna()
    print(f'  mean {ta.Best_Alpha_CTR.mean():.3f}  median {ta.Best_Alpha_CTR.median():.3f}  '
          f'std {ta.Best_Alpha_CTR.std():.3f}  share in [0.4,0.6]: {100*ta.Best_Alpha_CTR.between(0.4,0.6).mean():.1f}%')
    sl, ic, rr, pp, _ = stats.linregress(ta.zero_rate, ta.Best_Alpha_CTR)
    print(f'  tau* = {sl:.3f} x sparsity + {ic:.3f}   R2 = {rr**2:.3f}   p = {pp:.2e}')

    if {'CTR_prob','CTR_raw'}.issubset(prd.columns):
        print('\n=== TABLE G · threshold ablation ===')
        base = met[met.Model=='CTR'].set_index('Product_ID').MAE_clipped
        res = {'per-SKU tau* (proposed)': base.median()}
        for tau in (0.3, 0.4, 0.5, float(ta.Best_Alpha_CTR.median())):
            v = prd.groupby('Product_ID').apply(
                lambda g: np.mean(np.abs(g.y_test.values -
                    np.clip(np.where(g.CTR_prob.values > tau, g.CTR_raw.values, 0.0), 0, None))),
                include_groups=False)
            res[f'fixed tau = {tau:.2f}'] = v.median()
        t7 = pd.Series(res, name='Median MAE').to_frame()
        t7['Difference vs per-SKU %'] = (100*(t7['Median MAE']-res['per-SKU tau* (proposed)'])
                                        / res['per-SKU tau* (proposed)']).round(2)
        print(t7.round(4).to_string()); t7.to_csv(f'{OUT}/tableG_threshold_ablation.csv')
except ImportError:
    print('\n(scipy not installed - statistical tests skipped)')

t1.to_csv(f'{OUT}/tableA_overall.csv'); t2.to_csv(f'{OUT}/tableB_sparsity.csv')
print(f'\nwritten: {OUT}/  ->  combined_results.csv, predictions.csv, tableA..tableG')
