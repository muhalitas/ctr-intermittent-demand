#!/usr/bin/env python3
"""
CTR ablation study (component analysis).

Three variants are trained for every series under IDENTICAL data preparation,
architecture, loss and training settings as ctr_experiment.py / son2.py:

  CTR_full    : the full two-head CTR model with the per-SKU gate
  CTR_nogate  : the SAME trained two-head model, regression head used without the gate
  CTR_reghead : the same encoder and regression head trained alone
                (no classification head, no gate), same custom loss, same training

Every model is initialised from the same fixed seed (42) for each series, so the
run is reproducible.  Data preparation is imported unchanged from ctr_experiment.py.

Usage:
  python3 ctr_ablation.py --dataset btm --out ../ablation_btm [--workers 8] [--limit N]
  python3 ctr_ablation.py --dataset m5  --out ../ablation_m5  [--workers 8] [--sample 5000]
  python3 ctr_ablation.py --merge ../ablation_btm
"""
import os, sys, gc, json, time, argparse, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ctr_experiment as CE          # same preparation, metrics, constants

SEED = 42
VARIANTS = ['CTR_full', 'CTR_nogate', 'CTR_reghead']


def prepare(pid, dates, sales):
    """Exactly the preparation block of ctr_experiment.run_one / son2.py."""
    df = pd.DataFrame({'URUNID': pid, 'Date': pd.to_datetime(dates), 'Satis Adet': sales})
    df['Day'] = df.Date.dt.day; df['Month'] = df.Date.dt.month; df['Year'] = df.Date.dt.year
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['IsWeekend'] = (df.DayOfWeek >= 5).astype(int)
    df['Lag_1'] = df['Satis Adet'].shift(1)
    df['Lag_2'] = df['Satis Adet'].shift(2)
    df['Lag_3'] = df['Satis Adet'].shift(3)
    df['Rolling_Mean_3'] = df['Satis Adet'].shift(1).rolling(3).mean()
    df['Rolling_Mean_7'] = df['Satis Adet'].shift(1).rolling(7).mean()
    df = df.dropna()
    if len(df) < CE.TIME_STEPS:
        return None
    X = df[['URUNID', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend',
            'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Rolling_Mean_7']].values.astype(np.float64)
    y = df['Satis Adet'].values.astype(np.float64)
    test_size = int(len(X) * 0.15); val_size = int(len(X) * 0.15)
    if test_size == 0 or val_size == 0 or len(X) - test_size - val_size < 1:
        return None
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(); sc.fit(X[:len(X) - test_size - val_size]); X = sc.transform(X)
    Xs, ys = CE.create_sequences(X, y)
    n = len(Xs)
    Xtr, Xva, Xte = Xs[:n - test_size - val_size], Xs[n - test_size - val_size:n - test_size], Xs[-test_size:]
    ytr, yva, yte = ys[:n - test_size - val_size], ys[n - test_size - val_size:n - test_size], ys[-test_size:]
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        return None
    return Xtr, Xva, Xte, ytr, yva, yte


def metrics(yte, pred, ytr):
    p = np.maximum(np.asarray(pred, float), 0.0)          # non-negativity clipping (Section 3.4)
    pos, zer = yte > 0, yte == 0
    return dict(MAE=CE.mae_(yte, p), sMAPE=CE.smape(yte, p), MASE=CE.mase(yte, p, ytr),
                RMSSE=CE.rmsse(yte, p, ytr),
                MAE_pos=float(np.mean(np.abs(yte[pos] - p[pos]))) if pos.any() else np.nan,
                zeroFP=float(np.mean(p[zer] >= 0.5)) if zer.any() else np.nan,
                recall=float(np.mean(p[pos] >= 0.5)) if pos.any() else np.nan,
                abs_err_sum=float(np.sum(np.abs(yte - p))), abs_err_pos_sum=float(np.sum(np.abs(yte[pos] - p[pos]))),
                n_test=int(len(yte)), n_pos=int(pos.sum()))


def run_one(pid, dates, sales):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    prep = prepare(pid, dates, sales)
    if prep is None:
        return None
    Xtr, Xva, Xte, ytr, yva, yte = prep

    def ctr_loss(y_true, y_pred):
        m = tf.cast(y_true > 0, tf.float32)
        return tf.reduce_mean(tf.abs(y_true - y_pred) * (1 + 2 * m)) + 0.005 * tf.reduce_mean(tf.square(y_pred))

    def encoder(inp):
        h = LSTM(32, activation='relu', return_sequences=False)(inp)
        h = Dense(64, activation='relu')(h); h = Dropout(0.1)(h); h = Dense(32, activation='relu')(h)
        return h

    out = []
    # ---- full CTR (identical to ctr_experiment.py, including its seeding and compile call)
    tf.keras.utils.set_random_seed(SEED)
    inp = Input(shape=(CE.TIME_STEPS, Xtr.shape[-1])); h = encoder(inp)
    co = Dense(1, activation='sigmoid', name='classification_output')(h)
    ro = Dense(1, name='regression_output')(h)
    m = Model(inp, [co, ro])
    m.compile(optimizer=Adam(learning_rate=0.0001),
              loss={'classification_output': 'binary_crossentropy', 'regression_output': ctr_loss},
              loss_weights={'classification_output': 0.5, 'regression_output': 0.5},
              metrics={'classification_output': 'accuracy', 'regression_output': 'mae'},
              steps_per_execution=CE.SPE)
    es = EarlyStopping(monitor='val_regression_output_loss', patience=15, restore_best_weights=True, mode='min')
    m.fit(Xtr, {'classification_output': (ytr > 0).astype(int), 'regression_output': ytr},
          validation_data=(Xva, {'classification_output': (yva > 0).astype(int), 'regression_output': yva}),
          epochs=100, batch_size=32, callbacks=[es], verbose=0)
    cv, rv = (np.asarray(a).flatten() for a in m(Xva, training=False))
    maes = [np.mean(np.abs(yva - np.where(cv > a, rv, 0.0))) for a in CE.ALPHA_LEVELS]
    alpha = float(CE.ALPHA_LEVELS[int(np.argmin(maes))])
    ct, rt = (np.asarray(a).flatten() for a in m(Xte, training=False))
    full = np.where(ct > alpha, rt, 0.0)
    out.append(dict(Product_ID=pid, Variant='CTR_full', alpha=alpha, MAE_raw=CE.mae_(yte, full), **metrics(yte, full, ytr)))
    out.append(dict(Product_ID=pid, Variant='CTR_nogate', alpha=np.nan, MAE_raw=CE.mae_(yte, rt), **metrics(yte, rt, ytr)))
    del m

    # ---- regression head alone: same encoder, same loss, same optimiser and training
    tf.keras.utils.set_random_seed(SEED)
    inp = Input(shape=(CE.TIME_STEPS, Xtr.shape[-1])); h = encoder(inp)
    ro = Dense(1, name='regression_output')(h)
    m = Model(inp, ro)
    m.compile(optimizer=Adam(learning_rate=0.0001), loss=ctr_loss, steps_per_execution=CE.SPE)
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    rg_te = np.asarray(m(Xte, training=False)).flatten(); rg_va = np.asarray(m(Xva, training=False)).flatten()
    out.append(dict(Product_ID=pid, Variant='CTR_reghead', alpha=np.nan, MAE_raw=CE.mae_(yte, rg_te),
                    **metrics(yte, rg_te, ytr)))
    del m
    tf.keras.backend.clear_session(); gc.collect()
    # every daily output is stored, so that any later analysis needs no re-run
    pr = pd.concat([
        pd.DataFrame({'Product_ID': pid, 'part': 'val', 't': np.arange(len(yva)), 'y': yva,
                      'ctr_prob': cv, 'ctr_raw': rv, 'reghead_raw': rg_va, 'alpha': alpha}),
        pd.DataFrame({'Product_ID': pid, 'part': 'test', 't': np.arange(len(yte)), 'y': yte,
                      'ctr_prob': ct, 'ctr_raw': rt, 'reghead_raw': rg_te, 'alpha': alpha})])
    return out, pr


_CNT = _TOT = _T0 = None

def pool_init(cnt, tot, t0):
    global _CNT, _TOT, _T0
    _CNT, _TOT, _T0 = cnt, tot, t0

def _hms(sec):
    sec = int(max(sec, 0)); h, m = divmod(sec, 3600); m, s = divmod(m, 60)
    return f'{h:d}h {m:02d}m' if h else f'{m:d}m {s:02d}s'

def progress(wid, pid, sec, note=''):
    with _CNT.get_lock():
        _CNT.value += 1; done = _CNT.value
    el = time.time() - _T0; rate = el / done; eta = (_TOT - done) * rate
    pct = 100.0 * done / _TOT; bar = '#' * int(pct / 2.5) + '.' * (40 - int(pct / 2.5))
    print(f'[{bar}] {done:>5}/{_TOT} {pct:5.1f}%  elapsed {_hms(el)}  remaining ~{_hms(eta)}  '
          f'(avg {rate:4.1f}s/series, this one {sec:4.1f}s, w{wid}){note}', flush=True)

def worker(args):
    wid, pids, series, outdir = args
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try: tf.config.set_visible_devices([], 'GPU')
    except Exception: pass
    path = f'{outdir}/parts/abl_w{wid}.csv'
    ppath = f'{outdir}/preds/prd_w{wid}.csv'
    t0 = time.time()
    for j, pid in enumerate(pids):
        ts = time.time()
        try:
            r = run_one(int(pid), series[j][0], series[j][1])
        except Exception as e:
            progress(wid, pid, time.time() - ts, f'  << ERROR {type(e).__name__}: {e}'); continue
        if r:
            r[1].to_csv(ppath, mode='a', header=not os.path.exists(ppath), index=False)
            pd.DataFrame(r[0]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        progress(wid, pid, time.time() - ts, '' if r else '  << skipped (too short)')
    return wid


def done_ids(outdir):
    ids = set()
    pdir = f'{outdir}/parts'
    if os.path.isdir(pdir):
        for f in os.listdir(pdir):
            if f.endswith('.csv'):
                try:
                    d = pd.read_csv(os.path.join(pdir, f), usecols=['Product_ID', 'Variant'])
                    c = d.groupby('Product_ID').Variant.nunique()
                    ids |= set(c[c == 3].index)
                except Exception:
                    pass
    return ids


def merge(outdir):
    parts = [pd.read_csv(os.path.join(outdir, 'parts', f)) for f in os.listdir(os.path.join(outdir, 'parts')) if f.endswith('.csv')]
    d = pd.concat(parts).drop_duplicates(['Product_ID', 'Variant'], keep='last')
    full = d.groupby('Product_ID').Variant.nunique(); d = d[d.Product_ID.isin(full[full == 3].index)]
    d.to_csv(os.path.join(outdir, 'ablation_per_series.csv'), index=False)
    pdir = os.path.join(outdir, 'preds')
    if os.path.isdir(pdir):
        pr = pd.concat([pd.read_csv(os.path.join(pdir, f)) for f in os.listdir(pdir) if f.endswith('.csv')])
        pr = pr.drop_duplicates(['Product_ID', 'part', 't'], keep='last')
        pr = pr[pr.Product_ID.isin(d.Product_ID.unique())].sort_values(['Product_ID', 'part', 't'])
        pr.to_csv(os.path.join(outdir, 'ablation_daily_outputs.csv.gz'), index=False, compression='gzip')
    s = d.groupby('Variant').agg(n=('Product_ID', 'nunique'), MAE=('MAE', 'median'), sMAPE=('sMAPE', 'median'),
                                 MASE=('MASE', 'median'), RMSSE=('RMSSE', 'median'),
                                 pooled_MAE=('abs_err_sum', 'sum'), pooled_MAE_pos=('abs_err_pos_sum', 'sum'),
                                 n_test=('n_test', 'sum'), n_pos=('n_pos', 'sum'),
                                 zeroFP=('zeroFP', 'mean'), recall=('recall', 'mean'))
    s['pooled_MAE'] = s.pooled_MAE / s.n_test; s['pooled_MAE_pos'] = s.pooled_MAE_pos / s.n_pos
    s = s.drop(columns=['n_test', 'n_pos'])
    from scipy.stats import wilcoxon
    w = d.pivot(index='Product_ID', columns='Variant', values='MAE')
    rows = []
    for a, b in [('CTR_full', 'CTR_nogate'), ('CTR_full', 'CTR_reghead'), ('CTR_nogate', 'CTR_reghead')]:
        diff = w[b] - w[a]
        try: p = wilcoxon(diff[diff != 0]).pvalue
        except Exception: p = np.nan
        rows.append(dict(pair=f'{a} vs {b}', better=int((diff > 0).sum()), worse=int((diff < 0).sum()),
                         ties=int((diff == 0).sum()), median_diff=float(diff.median()), wilcoxon_p=p))
    s.to_csv(os.path.join(outdir, 'ablation_summary.csv'))
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'ablation_paired.csv'), index=False)
    print(s.round(4).to_string()); print(pd.DataFrame(rows).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['btm', 'm5'])
    ap.add_argument('--out'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int); ap.add_argument('--sample', type=int)
    ap.add_argument('--merge')
    a = ap.parse_args()
    if a.merge:
        return merge(a.merge)
    meta, series = CE.load_dataset(a.dataset, a.limit)[:2]
    pids = list(meta['URUNID'])
    if a.sample and a.sample < len(pids):
        rng = np.random.default_rng(SEED)
        keep = sorted(rng.choice(len(pids), a.sample, replace=False))
        pids = [pids[i] for i in keep]; series = [series[i] for i in keep]
    os.makedirs(f'{a.out}/parts', exist_ok=True); os.makedirs(f'{a.out}/preds', exist_ok=True)
    pd.DataFrame({'Product_ID': pids}).to_csv(f'{a.out}/series_ids.csv', index=False)
    skip = done_ids(a.out)
    todo = [(p, s) for p, s in zip(pids, series) if p not in skip]
    import tensorflow as _tf, sklearn as _sk, platform as _pf
    json.dump(dict(tf=_tf.__version__, sklearn=_sk.__version__, numpy=np.__version__, python=_pf.python_version(),
                   machine=_pf.platform(),
                   dataset=a.dataset, n_total=len(pids), n_todo=len(todo), workers=a.workers, seed=SEED,
                   sample=a.sample, started=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(f'{a.out}/run_meta.json', 'w'), indent=1)
    print(f'{a.dataset}: {len(pids)} series, {len(todo)} to do, {a.workers} workers', flush=True)
    chunks = [todo[i::a.workers] for i in range(a.workers)]
    jobs = [(i, [c[0] for c in ch], [c[1] for c in ch], a.out) for i, ch in enumerate(chunks) if ch]
    import multiprocessing as mp
    ctx = mp.get_context('spawn'); cnt = ctx.Value('i', 0)
    with ctx.Pool(len(jobs), initializer=pool_init, initargs=(cnt, len(todo), time.time())) as pool:
        for _ in pool.imap_unordered(worker, jobs):
            pass
    merge(a.out)


if __name__ == '__main__':
    main()
