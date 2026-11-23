"""Additional benchmark models under the protocol of the main experiment.

Part "classical" (no TensorFlow): SES, Croston, SBA, TSB, ADIDA, IMAPA, each in three information regimes
    <name>       fitted on the training targets, one constant forecast for the evaluation window
                 (this reproduces the SES and SBA benchmarks of the main experiment; its Croston benchmark
                 was fitted on the training and validation targets plus the six rows before the first window)
    <name>_trva  same, with the state carried to the end of the validation window (SES keeps the smoothing
                 constant and initial level estimated on the training targets)
    <name>_roll  one-step-ahead: the state is updated with every observed day, i.e. the method sees the
                 same past demand that the neural and tree models receive through their lag features
Part "ml": LightGBM, XGBoost (flattened seven-row window, library defaults), TwoStage (random-forest
    occurrence classifier + XGBoost magnitude regressor, threshold chosen as for CTR), GRU (the LSTM
    benchmark with the recurrent layer replaced); diagnostics LightGBM_tweedie and TwoStage_F1.

Unchanged from the main experiment: preparation, features, chronological 70/15/15 split, scaler fitted on
the training rows, seven-row windows (ctr_ablation.prepare); metrics after the non-negativity clip
(ctr_ablation.metrics); seed 42; one model per item; one CPU thread per model.
The two parts run in separate processes because TensorFlow and pyarrow (a dependency of statsforecast)
cannot be loaded in the same process on macOS.

Files written per worker k in --out:
    <part>_w<k>.csv         one row per item and model: test-window metrics
    daily_<part>_w<k>.csv   one row per item and day of the validation and test windows: date, demand,
                            unclipped forecast of every model (and TwoStage probability / magnitude)
    meta_<part>_w<k>.csv    one row per item: window lengths, scaling denominators, thresholds, fall-back
                            flags, fitting time of every model in seconds

Usage (from M5/run):
  python3 rivals.py --dataset btm|m5 --part classical|ml --out <dir> [--sample N] [--workers 8] [--worker-id 0]
"""
import os, sys, argparse, time, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
ALPHA = 0.4                      # smoothing constant of the Croston and SBA benchmarks
CLASSICAL = ['SES', 'Croston', 'SBA', 'TSB', 'ADIDA', 'IMAPA']
PARTS = {'classical': [m + s for m in CLASSICAL for s in ('', '_trva', '_roll')],
         'ml': ['LightGBM', 'XGBoost', 'TwoStage', 'GRU', 'LightGBM_tweedie', 'TwoStage_F1']}
EXTRA_DAILY = {'classical': [], 'ml': ['TwoStage_prob', 'TwoStage_mag']}


def tsb_states(y, alpha_d=ALPHA, alpha_p=ALPHA):
    """Teunter-Syntetos-Babai recursion (reference implementation, used only by the self-test).
    Same recursion as statsforecast: exponential smoothing of the 0/1 demand indicator (probability)
    and of the non-zero demand sizes (size)."""
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0, 0.0
    z = y[nz[0]]
    p = 1.0 if nz[0] == 0 else 0.0
    for t in range(1, len(y)):
        if y[t] > 0:
            if t > nz[0]:
                z = alpha_d * y[t] + (1 - alpha_d) * z
            p = alpha_p * 1.0 + (1 - alpha_p) * p
        else:
            p = (1 - alpha_p) * p
    return p, z


def one_pass(y, alpha_ses, l0):
    """f[name][t] = forecast for index t from observations 0..t-1 (t = 0..n), for SES, Croston, SBA, TSB.
    Croston and SBA follow ctr_experiment.croston_forecast / sba_forecast exactly; TSB follows tsb_states."""
    n = len(y); f = {k: np.zeros(n + 1) for k in ('SES', 'Croston', 'SBA', 'TSB')}
    lvl = l0
    cz = None; cp = 1.0; clast = None
    sz = None; sp = 1.0; slast = None; scount = 0
    tp = 0.0; tz = 0.0; tfirst = None
    A = ALPHA
    for t in range(n + 1):
        f['SES'][t] = lvl
        f['Croston'][t] = 0.0 if cz is None else cz / (cp if cp != 0 else 1)
        f['SBA'][t] = 0.0 if sz is None else (1 - A / 2) * sz / (sp if sp != 0 else 1)
        f['TSB'][t] = tp * tz
        if t == n:
            break
        v = y[t]
        lvl = alpha_ses * v + (1 - alpha_ses) * lvl
        if v > 0:
            if cz is None:
                cz, cp = v, 1.0
            else:
                cz = A * v + (1 - A) * cz; cp = A * (t - clast) + (1 - A) * cp
            clast = t
            if sz is None:
                sz, sp = v, 1.0
            else:
                sz = A * v + (1 - A) * sz
                sp = float(t - slast) if scount == 1 else A * (t - slast) + (1 - A) * sp
            slast = t; scount += 1
            if tfirst is None:
                tfirst = t; tz = v; tp = 1.0 if t == 0 else A
            else:
                tz = A * v + (1 - A) * tz; tp = A + (1 - A) * tp
        elif tfirst is not None:
            tp = (1 - A) * tp
    return f


def selftest(CE):
    """The one-pass recursions reproduce the reference functions on every prefix of random series."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(5, 100)); y = rng.poisson(0.3, n).astype(float) * rng.integers(1, 6, n)
        f = one_pass(y, 0.3, 0.0)
        for t in range(1, n + 1):
            c = CE.croston_forecast(y[:t], 1)[0]; s = CE.sba_forecast(y[:t], extra_periods=1)[0]
            p, z = tsb_states(y[:t])
            assert abs(f['Croston'][t] - c) < 1e-9 and abs(f['SBA'][t] - s) < 1e-9 and abs(f['TSB'][t] - p * z) < 1e-9


def target_dates(dates, n, AB):
    """Calendar dates of the n targets returned by ctr_ablation.prepare (rows lost to lags and to the window)."""
    d = pd.to_datetime(pd.Series(list(dates)))
    d = d.iloc[len(d) - n:]
    return d.dt.strftime('%Y-%m-%d').to_numpy()


def run_one(pid, dates, sales, AB, part):
    prep = AB.prepare(pid, dates, sales)
    if prep is None:
        return None
    Xtr, Xva, Xte, ytr, yva, yte = prep
    ytr, yva, yte = (np.asarray(a, float) for a in (ytr, yva, yte))
    ntr, nva, h = len(ytr), len(yva), len(yte)
    n0 = ntr + nva
    dts = target_dates(dates, ntr + nva + h, AB)
    Pte, Pva, T = {}, {}, {}
    ne = np.abs(ytr[1:] - ytr[:-1])
    meta = dict(Product_ID=pid, n_train=ntr, n_val=nva, n_test=h, pos_train=int((ytr > 0).sum()), pos_val=int((yva > 0).sum()),
                pos_test=int((yte > 0).sum()), scale_mae=float(ne.mean()) if len(ne) else np.nan,
                scale_mse=float((ne ** 2).mean()) if len(ne) else np.nan)

    if part == 'classical':
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        from statsforecast.models import ADIDA, IMAPA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        full = np.concatenate([ytr, yva, yte])
        t0 = time.perf_counter(); ses_tr = None
        try:
            m = SimpleExpSmoothing(ytr, initialization_method='estimated').fit(optimized=True)
            a_ses, l0 = float(m.params['smoothing_level']), float(m.params['initial_level'])
            ses_tr = float(np.asarray(m.forecast(1))[0])
        except Exception:
            a_ses, l0 = 0.1, float(ytr.mean())
        meta['alpha_ses'] = a_ses; meta['ses_fit_failed'] = int(ses_tr is None)
        f = one_pass(full, a_ses, l0)
        tt = time.perf_counter() - t0
        for k in f:
            c_tr = ses_tr if (k == 'SES' and ses_tr is not None) else f[k][ntr]
            Pte[k], Pva[k] = np.full(h, c_tr), np.full(nva, c_tr)
            Pte[k + '_trva'], Pva[k + '_trva'] = np.full(h, f[k][n0]), np.full(nva, f[k][n0])
            Pte[k + '_roll'], Pva[k + '_roll'] = f[k][n0:n0 + h], f[k][ntr:n0]
            T[k] = tt / 4
        for name, cls in (('ADIDA', ADIDA), ('IMAPA', IMAPA)):
            t0 = time.perf_counter()
            fc = lambda hist: float(cls().forecast(y=hist, h=1)['mean'][0]) if (hist > 0).any() else 0.0
            c_tr, c_tv = fc(ytr), fc(full[:n0])
            Pte[name], Pva[name] = np.full(h, c_tr), np.full(nva, c_tr)
            Pte[name + '_trva'], Pva[name + '_trva'] = np.full(h, c_tv), np.full(nva, c_tv)
            Pte[name + '_roll'] = np.array([fc(full[:n0 + t]) for t in range(h)])
            Pva[name + '_roll'] = np.array([fc(full[:ntr + t]) for t in range(nva)])
            T[name] = time.perf_counter() - t0
    else:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GRU, Dense, Input
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_recall_curve
        import lightgbm as lgb
        import xgboost as xgb
        Xtr2, Xva2, Xte2 = (a.reshape(len(a), -1) for a in (Xtr, Xva, Xte))

        # ---- tree ensembles on the flattened seven-row window, library default settings
        np.random.seed(SEED)
        fallback = int(ntr < 2 or ytr.min() == ytr.max())      # nothing to learn: constant forecast
        meta['tree_constant_fallback'] = fallback
        def fit_pred(name, make):
            t0 = time.perf_counter()
            if fallback:
                Pte[name], Pva[name] = np.full(h, float(ytr[0])), np.full(nva, float(ytr[0]))
            else:
                mdl = make().fit(Xtr2, ytr); Pte[name], Pva[name] = mdl.predict(Xte2), mdl.predict(Xva2)
            T[name] = time.perf_counter() - t0
        fit_pred('LightGBM', lambda: lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1))
        fit_pred('XGBoost', lambda: xgb.XGBRegressor(random_state=SEED, n_jobs=1))
        fit_pred('LightGBM_tweedie', lambda: lgb.LGBMRegressor(objective='tweedie', random_state=SEED, n_jobs=1, verbose=-1))

        # ---- two-stage pipeline: occurrence classifier, then magnitude regressor trained on demand days
        t0 = time.perf_counter()
        btr, bva = (ytr > 0).astype(int), (yva > 0).astype(int)
        single = int(btr.min() == btr.max()); meta['twostage_single_class'] = single
        thr = thr_f1 = np.nan
        if single:               # a single class in training: no classifier can be fitted
            c = 0.0 if btr.max() == 0 else float(np.median(ytr))
            for k in ('TwoStage', 'TwoStage_F1', 'TwoStage_mag'):
                Pte[k], Pva[k] = np.full(h, c), np.full(nva, c)
            Pte['TwoStage_prob'], Pva['TwoStage_prob'] = np.full(h, float(btr.max())), np.full(nva, float(btr.max()))
        else:
            clf = RandomForestClassifier(random_state=SEED, n_jobs=1).fit(Xtr2, btr)
            pva, pte = clf.predict_proba(Xva2)[:, 1], clf.predict_proba(Xte2)[:, 1]
            pos = ytr > 0
            if pos.sum() >= 2:
                reg = xgb.XGBRegressor(random_state=SEED, n_jobs=1).fit(Xtr2[pos], ytr[pos])
                magv, mag = reg.predict(Xva2), reg.predict(Xte2)
            else:
                magv, mag = np.full(nva, float(ytr[pos].mean())), np.full(h, float(ytr[pos].mean()))
            # release threshold chosen exactly as for CTR (Eq. 6): validation MAE over the same 50-point grid
            maes = [np.mean(np.abs(yva - np.where(pva > t, magv, 0.0))) for t in AB.CE.ALPHA_LEVELS]
            thr = float(AB.CE.ALPHA_LEVELS[int(np.argmin(maes))])
            Pte['TwoStage'], Pva['TwoStage'] = np.where(pte > thr, mag, 0.0), np.where(pva > thr, magv, 0.0)
            # diagnostic: F1-maximising threshold of the cited two-stage pipeline
            thr_f1 = 0.5
            if bva.max() == 1 and bva.min() == 0:
                prec, rec, ths = precision_recall_curve(bva, pva)
                f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
                thr_f1 = float(ths[int(np.argmax(f1))])
            Pte['TwoStage_F1'], Pva['TwoStage_F1'] = np.where(pte >= thr_f1, mag, 0.0), np.where(pva >= thr_f1, magv, 0.0)
            Pte['TwoStage_prob'], Pva['TwoStage_prob'] = pte, pva
            Pte['TwoStage_mag'], Pva['TwoStage_mag'] = mag, magv
        meta['twostage_threshold'] = thr; meta['twostage_threshold_f1'] = thr_f1
        T['TwoStage'] = time.perf_counter() - t0

        # ---- GRU: the LSTM benchmark with the recurrent layer replaced
        t0 = time.perf_counter()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        i = Input(shape=(Xtr.shape[1], Xtr.shape[-1]))
        g = Model(i, Dense(1)(GRU(50, activation='relu')(i)))
        g.compile(optimizer='adam', loss='mean_absolute_error')
        g.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=50, verbose=0)
        Pte['GRU'] = np.asarray(g(Xte, training=False)).flatten()
        Pva['GRU'] = np.asarray(g(Xva, training=False)).flatten()
        del g
        tf.keras.backend.clear_session()
        T['GRU'] = time.perf_counter() - t0

    rows = []
    for name in PARTS[part]:
        p = np.asarray(Pte[name], float)
        r = AB.metrics(yte, p, ytr)
        r.update(Product_ID=pid, Model=name, MAE_unclipped=float(np.mean(np.abs(yte - p))))
        rows.append(r)
    cols = PARTS[part] + EXTRA_DAILY[part]
    daily = pd.concat([
        pd.DataFrame({'Product_ID': pid, 'part': 'val', 't': np.arange(nva), 'date': dts[ntr:n0], 'y': yva,
                      **{k: np.asarray(Pva[k], float) for k in cols}}),
        pd.DataFrame({'Product_ID': pid, 'part': 'test', 't': np.arange(h), 'date': dts[n0:], 'y': yte,
                      **{k: np.asarray(Pte[k], float) for k in cols}})])
    meta['nonfinite'] = int(not all(np.isfinite(np.asarray(v, float)).all() for v in list(Pte.values()) + list(Pva.values())))
    meta.update({'sec_' + k: round(v, 4) for k, v in T.items()})
    return rows, daily, meta


def heal(path):
    """A worker can be killed while writing: cut a file back to its last complete line."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, 'rb+') as f:
        f.seek(0, os.SEEK_END); size = f.tell()
        f.seek(max(0, size - 1)); 
        if f.read(1) == b'\n':
            return
        back = min(size, 1 << 20); f.seek(size - back); tail = f.read(back)
        cut = tail.rfind(b'\n')
        f.truncate(size - back + cut + 1 if cut >= 0 else 0)


def append(df, path, **kw):
    df.to_csv(path, mode='a', header=not os.path.exists(path) or os.path.getsize(path) == 0, index=False, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['btm', 'm5'], required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--part', choices=['classical', 'ml'], required=True)
    ap.add_argument('--max-items', type=int, default=0, help='stop after this many items; the launcher restarts the worker (keeps per-item time constant)')
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--worker-id', type=int, default=0)
    ap.add_argument('--ids', default='', help='optional csv with a Product_ID column restricting the items')
    a = ap.parse_args()
    assert 0 <= a.worker_id < a.workers
    sys.path.insert(0, HERE)
    if a.part == 'ml':
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    import ctr_ablation as AB
    if a.part == 'classical':
        selftest(AB.CE)
    meta, series = AB.CE.load_dataset(a.dataset, None)[:2]            # same loader and item order as the main experiment
    col = 'URUNID' if 'URUNID' in meta.columns else 'ProductID'
    pids = pd.Series(list(meta[col]))
    keep = pd.Series(True, index=pids.index)
    if a.ids:
        keep &= pids.isin(set(pd.read_csv(a.ids).Product_ID))
    idx = pids.index[keep]
    if a.sample:
        idx = np.sort(np.random.default_rng(SEED).choice(idx, size=a.sample, replace=False))
    mine = list(idx)[a.worker_id::a.workers]
    os.makedirs(a.out, exist_ok=True)
    path = f'{a.out}/{a.part}_w{a.worker_id}.csv'
    dpath = f'{a.out}/daily_{a.part}_w{a.worker_id}.csv'
    mpath = f'{a.out}/meta_{a.part}_w{a.worker_id}.csv'
    NM = len(PARTS[a.part])
    for f_ in (path, dpath, mpath):
        heal(f_)
    done = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:      # keep only items with a complete set of rows
        try:
            prev = pd.read_csv(path, float_precision='round_trip')
        except pd.errors.EmptyDataError:
            prev = None
        if prev is not None and len(prev):
            c = prev.Product_ID.value_counts(); done = set(c[c == NM].index)
            if (c != NM).any():                                 # rewrite only if an incomplete item is present, atomically
                prev[prev.Product_ID.isin(done)].to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
    vers = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__}
    for mod in ('sklearn', 'lightgbm', 'xgboost', 'tensorflow', 'statsforecast', 'statsmodels'):
        if mod in sys.modules or (a.part == 'classical' and mod in ('statsforecast', 'statsmodels', 'sklearn')) or (a.part == 'ml' and mod in ('sklearn', 'lightgbm', 'xgboost')):
            try: vers[mod] = __import__(mod).__version__
            except Exception: pass
    if a.worker_id == 0:
        import json
        json.dump(vers, open(f'{a.out}/versions_{a.part}.json', 'w'), indent=1)
    t0 = time.time(); k = 0; nfail = 0
    for i in mine:
        pid = pids[i]
        if pid in done:
            continue
        try:
            out = run_one(pid, series[i][0], np.asarray(series[i][1], dtype=np.float64), AB, a.part)
        except Exception as e:
            nfail += 1; print(f'worker {a.worker_id}: item {pid} failed: {e!r}', flush=True); continue
        if out:
            rows, daily, meta_row = out
            append(daily, dpath, float_format='%.9g')          # daily forecasts first: an item counts as done only once its metrics are written
            append(pd.DataFrame([meta_row]), mpath)
            append(pd.DataFrame(rows), path)
        k += 1
        if k % 25 == 0:
            print(f'worker {a.worker_id}: {k}/{len(mine)} items, {time.time()-t0:.0f}s', flush=True)
        if a.max_items and k >= a.max_items:
            print('restart', a.worker_id, flush=True); return
    if nfail:
        print(f'incomplete {a.worker_id}: {nfail} items failed', flush=True); sys.exit(3)
    print('done', a.worker_id, flush=True)


if __name__ == '__main__':
    main()
