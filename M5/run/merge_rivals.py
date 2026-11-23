"""Merge the per-worker files of one rivals.py run and write the summary tables.

Usage: python3 merge_rivals.py <out dir>

Writes to <out dir>: rivals_per_item.csv (one row per item and model), rivals_summary.csv (medians),
rivals_meta_<part>.csv (per-item window lengths, thresholds, fall-back flags, fitting times) and
run_meta.json (item counts, library versions, mean fitting time per item, completeness checks)."""
import sys, os, glob, json, time, platform
import numpy as np, pandas as pd

out = sys.argv[1].rstrip('/')
M = ['MAE', 'sMAPE', 'MASE', 'RMSSE']
MODELS = {'classical': [m + x for m in ('SES', 'Croston', 'SBA', 'TSB', 'ADIDA', 'IMAPA') for x in ('', '_trva', '_roll')],
          'ml': ['LightGBM', 'XGBoost', 'TwoStage', 'GRU', 'LightGBM_tweedie', 'TwoStage_F1']}


def cat(pat, keys):
    fs = sorted(glob.glob(f'{out}/{pat}'))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(keys, keep='last')   # restarts may repeat an item


d = pd.concat([x for x in (cat('classical_w*.csv', ['Product_ID', 'Model']), cat('ml_w*.csv', ['Product_ID', 'Model'])) if x is not None])
d.to_csv(out + '/rivals_per_item.csv', index=False)
s = d.groupby('Model')[M].median().round(4)
s.insert(0, 'n', d.groupby('Model').size())
s.to_csv(out + '/rivals_summary.csv')

problems = []
nper = d.groupby('Model').size()
if nper.nunique() != 1:
    problems.append('models have different item counts: ' + str(nper.to_dict()))
if d[M].isna().any().any():
    problems.append('NaN in metrics: ' + str(d[M].isna().sum().to_dict()))
info = dict(finished=time.strftime('%Y-%m-%d %H:%M:%S'), items=int(d.Product_ID.nunique()), rows=int(len(d)), seed=42,
            python=platform.python_version(), machine=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
            items_per_model=nper.to_dict())
for part in ('classical', 'ml'):
    mt = cat(f'meta_{part}_w*.csv', ['Product_ID'])
    if mt is None:
        continue
    mt.to_csv(f'{out}/rivals_meta_{part}.csv', index=False)
    info[f'seconds_per_item_{part}'] = mt[[c for c in mt.columns if c.startswith('sec_')]].mean().round(4).to_dict()
    fs = sorted(glob.glob(f'{out}/daily_{part}_w*.csv'))
    if fs:   # daily forecast files complete for every item?
        cnt = pd.concat([ch for f in fs for ch in pd.read_csv(f, usecols=['Product_ID', 'part', 't'], chunksize=2_000_000)]).drop_duplicates(keep='last')
        per = cnt.groupby('Product_ID').size()
        exp = mt.set_index('Product_ID')[['n_val', 'n_test']].sum(axis=1)
        met = set(d.Product_ID[d.Model.isin(MODELS[part])])
        bad = [p for p in exp.index if per.get(p, 0) != exp[p]]
        if bad or set(exp.index) != met or set(per.index) != met:
            problems.append(f'{part}: metric, meta and daily files do not cover the same items ({len(bad)} items with a wrong number of daily rows)')
    vf = f'{out}/versions_{part}.json'
    if os.path.exists(vf):
        info[f'versions_{part}'] = json.load(open(vf))
info['problems'] = problems
json.dump(info, open(out + '/run_meta.json', 'w'), indent=1)
pd.set_option('display.width', 200)
print(s.to_string())
if problems:
    print('PROBLEMS:')
    for p in problems:
        print('  -', p)
    sys.exit(2)
print('completeness check passed: metric, meta and daily files cover the same items')
