"""
M5 -> CTR deney formatina hazirlik.
Orijinal protokolun birebir tekrari icin:
  * seri uzunlugu BTM ile ayni olsun diye son 1095 gun alinir
  * SADECE Syntetos-Boylan aralikli talep kriteri (ADI > 1.32) uygulanir
    (seyreklik bandina gore ELEME YOK; bantlar sonradan raporlama etiketi)
  * >= 30 sifirdan farkli gozlem sarti pratik bir gerekliliktir (7 gunluk
    pencere + %70/15/15 bolunme icin en az bir kac olay gerekiyor)
Cikti: sales.npy (float32, n_sku x 1095), skus.csv, dates.csv
"""
import numpy as np, pandas as pd, os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'prep')
W    = 1095
ADI_CUT = 1.32
MIN_NZ  = 30

cal = pd.read_csv(os.path.join(BASE,'calendar.csv'), usecols=['date','d'] if 'd' in pd.read_csv(os.path.join(BASE,'calendar.csv'), nrows=1).columns else None)
d   = pd.read_csv(os.path.join(BASE,'sales_train_evaluation.csv'))
daycols = [c for c in d.columns if c.startswith('d_')]
sub     = daycols[-W:]

# tarihler
if 'd' in cal.columns:
    dmap = dict(zip(cal['d'], cal['date']))
else:
    cal2 = pd.read_csv(os.path.join(BASE,'calendar.csv'))
    dmap = {f'd_{i+1}': v for i, v in enumerate(cal2['date'].tolist())}
dates = [dmap[c] for c in sub]

Y  = d[sub].to_numpy(dtype=np.float32)
nz = (Y > 0)
n_d = nz.sum(1)
T   = Y.shape[1]
adi = np.where(n_d > 0, T / np.maximum(n_d, 1), np.inf)

keep = (adi > ADI_CUT) & (n_d >= MIN_NZ) & np.isfinite(adi)
print(f'toplam seri      : {len(d)}')
print(f'ADI > {ADI_CUT} & nz >= {MIN_NZ}: {int(keep.sum())}')

Yk = Y[keep]
ids = (d['item_id'] + '_' + d['store_id'])[keep].reset_index(drop=True)
meta = pd.DataFrame({
    'URUNID'   : np.arange(1, len(ids)+1, dtype=np.int64),   # son2.py int ID bekliyor
    'm5_id'    : ids,
    'item_id'  : d['item_id'][keep].values,
    'store_id' : d['store_id'][keep].values,
    'cat_id'   : d['cat_id'][keep].values,
    'state_id' : d['state_id'][keep].values,
    'n_nonzero': n_d[keep],
    'ADI'      : adi[keep],
    'zero_rate': 1 - n_d[keep]/T,
})
np.save(os.path.join(OUT,'sales.npy'), Yk)
meta.to_csv(os.path.join(OUT,'skus.csv'), index=False)
pd.Series(dates, name='Date').to_csv(os.path.join(OUT,'dates.csv'), index=False)

bins   = [0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0001]
labels = ['[0.0, 0.2)','[0.2, 0.3)','[0.3, 0.4)','[0.4, 0.5)','[0.5, 0.6)','[0.6, 0.7)','[0.7, 0.8)','[0.8, 1.0)']
meta['Sparsity_Level'] = pd.cut(meta.zero_rate, bins=bins, labels=labels, include_lowest=True, right=False)
print('\nseyreklik bandi dagilimi (raporlama etiketi, eleme degil):')
print(meta.Sparsity_Level.value_counts().reindex(labels).to_string())
print(f'\nmatris: {Yk.shape}  ({Yk.nbytes/1e6:.0f} MB)')
print(f'tarih araligi: {dates[0]} .. {dates[-1]}')
print(f'ADI  ort {meta.ADI.mean():.2f}  medyan {meta.ADI.median():.2f}')
print(f'sifir orani ort {meta.zero_rate.mean():.3f}  medyan {meta.zero_rate.median():.3f}')
json.dump({'window':W,'adi_cut':ADI_CUT,'min_nonzero':MIN_NZ,'n_sku':int(keep.sum()),
           'date_start':dates[0],'date_end':dates[-1]},
          open(os.path.join(OUT,'prep_meta.json'),'w'), indent=2)
