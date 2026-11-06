#!/usr/bin/env python3
"""
CTR deneyi - son2.py'nin birebir portu, paralel + devam ettirilebilir.

Modelde, hiperparametrede, bolunmede, esik aramasinda VEYA metrik
tanimlarinda hicbir degisiklik yoktur. Degisen sadece sunlar:
  1) SKU'lar surec havuzunda paralel islenir (her isci tek is parcacigi)
  2) veri her SKU icin tam tarama yerine onceden gruplanir
  3) sonuclar parcalar halinde yazilir -> is yarida kesilirse kaldigi yerden devam

Kullanim:
  python3 ctr_experiment.py --dataset m5  --out SONUC_DIZINI [--limit N] [--workers K]
  python3 ctr_experiment.py --dataset btm --out SONUC_DIZINI [--limit N] [--workers K]
"""
import os, sys, gc, json, time, argparse, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M5_PREP = os.path.abspath(os.path.join(HERE, '..', 'prep'))
BTM_CSV = os.path.abspath(os.path.join(HERE, '..', '..', 'Veri', 'filtered_data.csv'))

TIME_STEPS = 7
ALPHA_LEVELS = np.linspace(0.01, 0.99, 50)
SEED = 42
SPE = 32                        # Keras adim toplulastirma (gradyan sirasini degistirmez)
CHUNK = 25                      # kac SKU'da bir diske yaz

# ----------------------------------------------------------------- metrikler
def mase(y_true, y_pred, y_train):
    ne = np.abs(y_train[1:] - y_train[:-1])
    scale = np.mean(ne) if len(ne) > 0 and np.mean(ne) != 0 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / scale

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    abs_diff = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    mask = den == 0
    if np.all(mask):
        return 0.0 if np.all(abs_diff == 0) else 100.0
    den = np.where(mask, 1, den)
    return float(np.mean(np.where(mask, 0, 2 * abs_diff / den)) * 100)

def rmsse(y_true, y_pred, y_train):
    mse = np.mean((y_true - y_pred) ** 2)
    ne = np.mean((y_train[1:] - y_train[:-1]) ** 2)
    return float(np.sqrt(mse / (ne if ne != 0 else 1e-10)))

def mae_(a, b):  return float(np.mean(np.abs(np.asarray(a,float) - np.asarray(b,float))))
def rmse_(a, b): return float(np.sqrt(np.mean((np.asarray(a,float) - np.asarray(b,float)) ** 2)))
def mape_(a, b):
    a = np.asarray(a,float); b = np.asarray(b,float)
    d = np.where(a == 0, 1e-10, a)
    return float(np.mean(np.abs((a - b) / d)))

def sba_forecast(y, extra_periods=0, alpha=0.4, beta=0.4):
    """son2.py'deki sba() ile ayni."""
    demand, intervals, last = [], [], None
    for t, v in enumerate(y):
        if v > 0:
            demand.append(v)
            if last is not None:
                intervals.append(t - last)
            last = t
    if not demand:
        return np.zeros(extra_periods)
    z = demand[0]
    p = intervals[0] if intervals else 1
    for i in range(1, len(demand)):
        z = alpha * demand[i] + (1 - alpha) * z
        if i - 1 < len(intervals):
            p = beta * intervals[i-1] + (1 - beta) * p
    p = p if p != 0 else 1
    return np.array([(1 - beta / 2) * z / p] * extra_periods)

def croston_forecast(y, extra_periods, smoothing=0.4):
    """sktime Croston(smoothing) ile ayni: boyut ve araligi ayni katsayiyla duzlestir."""
    demand, intervals, last = [], [], None
    for t, v in enumerate(y):
        if v > 0:
            demand.append(v)
            intervals.append(t - last if last is not None else 1)
            last = t
    if not demand:
        return np.zeros(extra_periods)
    z, p = demand[0], intervals[0] if intervals else 1
    for i in range(1, len(demand)):
        z = smoothing * demand[i] + (1 - smoothing) * z
        p = smoothing * intervals[i] + (1 - smoothing) * p
    p = p if p != 0 else 1
    return np.array([z / p] * extra_periods)

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs = np.stack([X[i:i+time_steps] for i in range(len(X) - time_steps + 1)])
    ys = y[time_steps-1:]
    return Xs, ys

MODELS = ['CTR', 'CTR_Regression', 'FNN', 'RNN', 'LSTM', 'SES', 'Croston', 'SBA']

# ----------------------------------------------------------------- tek SKU
def run_one(pid, dates, sales):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    np.random.seed(SEED); tf.random.set_seed(SEED)

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
    if len(df) < TIME_STEPS:
        return None

    X = df[['URUNID','Day','Month','Year','DayOfWeek','IsWeekend',
            'Lag_1','Lag_2','Lag_3','Rolling_Mean_3','Rolling_Mean_7']].values.astype(np.float64)
    y = df['Satis Adet'].values.astype(np.float64)

    test_size = int(len(X) * 0.15); val_size = int(len(X) * 0.15)
    if test_size == 0 or val_size == 0 or len(X) - test_size - val_size < 1:
        return None

    sc = StandardScaler(); sc.fit(X[:len(X)-test_size-val_size]); X = sc.transform(X)
    Xs, ys = create_sequences(X, y)
    n = len(Xs)
    Xtr, Xva, Xte = Xs[:n-test_size-val_size], Xs[n-test_size-val_size:n-test_size], Xs[-test_size:]
    ytr, yva, yte = ys[:n-test_size-val_size], ys[n-test_size-val_size:n-test_size], ys[-test_size:]
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        return None
    Xtr2, Xva2, Xte2 = (a.reshape(len(a), -1) for a in (Xtr, Xva, Xte))

    def ctr_loss(y_true, y_pred):
        m = tf.cast(y_true > 0, tf.float32)
        return tf.reduce_mean(tf.abs(y_true - y_pred) * (1 + 2*m)) + 0.005*tf.reduce_mean(tf.square(y_pred))

    P, M = {}, {}
    def score(name, pred):
        M[name] = dict(MAE=mae_(yte,pred), MAPE=mape_(yte,pred), RMSE=rmse_(yte,pred),
                       MASE=mase(yte,pred,ytr), sMAPE=smape(yte,pred), RMSSE=rmsse(yte,pred,ytr))
        P[name] = np.asarray(pred, float)

    # ---- CTR
    inp = Input(shape=(TIME_STEPS, Xtr.shape[-1]))
    h = LSTM(32, activation='relu', return_sequences=False)(inp)
    h = Dense(64, activation='relu')(h); h = Dropout(0.1)(h); h = Dense(32, activation='relu')(h)
    co = Dense(1, activation='sigmoid', name='classification_output')(h)
    ro = Dense(1, name='regression_output')(h)
    m_ctr = Model(inp, [co, ro])
    m_ctr.compile(optimizer=Adam(learning_rate=0.0001),
                  loss={'classification_output':'binary_crossentropy','regression_output':ctr_loss},
                  loss_weights={'classification_output':0.5,'regression_output':0.5},
                  metrics={'classification_output':'accuracy','regression_output':'mae'},
                  steps_per_execution=SPE)
    es = EarlyStopping(monitor='val_regression_output_loss', patience=15,
                       restore_best_weights=True, mode='min')
    m_ctr.fit(Xtr, {'classification_output':(ytr>0).astype(int),'regression_output':ytr},
              validation_data=(Xva, {'classification_output':(yva>0).astype(int),'regression_output':yva}),
              epochs=100, batch_size=32, callbacks=[es], verbose=0)
    cv, rv = (np.asarray(a).flatten() for a in m_ctr(Xva, training=False))
    maes = [np.mean(np.abs(yva - np.where(cv > a, rv, 0.0))) for a in ALPHA_LEVELS]
    best_alpha = float(ALPHA_LEVELS[int(np.argmin(maes))])
    ct, rt = (np.asarray(a).flatten() for a in m_ctr(Xte, training=False))
    score('CTR', np.where(ct > best_alpha, rt, 0.0))
    P['_ctr_prob'] = ct; P['_ctr_raw'] = rt

    # ---- CTR_Regression (ablasyon)
    i2 = Input(shape=(Xtr2.shape[1],))
    g = Dense(64, activation='relu')(i2); g = Dense(64, activation='relu')(g); o2 = Dense(1)(g)
    m2 = Model(i2, o2); m2.compile(optimizer='adam', loss='mean_absolute_error', steps_per_execution=SPE)
    m2.fit(Xtr2, ytr, validation_data=(Xva2, yva), epochs=50, verbose=0)
    score('CTR_Regression', np.asarray(m2(Xte2, training=False)).flatten())

    # ---- FNN
    i3 = Input(shape=(Xtr2.shape[1],))
    g = Dense(10, activation='relu')(i3); g = Dense(3, activation='relu')(g); o3 = Dense(1)(g)
    m3 = Model(i3, o3); m3.compile(optimizer='adam', loss='mean_absolute_error', steps_per_execution=SPE)
    m3.fit(Xtr2, ytr, epochs=50, verbose=0)
    score('FNN', np.asarray(m3(Xte2, training=False)).flatten())

    # ---- RNN (esikli)
    i4 = Input(shape=(TIME_STEPS, Xtr.shape[-1]))
    o4 = Dense(1)(LSTM(50, activation='relu')(i4))
    m4 = Model(i4, o4); m4.compile(optimizer='adam', loss='mean_absolute_error', steps_per_execution=SPE)
    m4.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=50, verbose=0)
    rvp = np.asarray(m4(Xva, training=False)).flatten()
    rvc = (rvp > np.mean(rvp)).astype(int)
    maes = [np.mean(np.abs(yva - np.where(rvc > t, rvp, 0.0))) for t in ALPHA_LEVELS]
    best_thr_rnn = float(ALPHA_LEVELS[int(np.argmin(maes))])
    rtp = np.asarray(m4(Xte, training=False)).flatten()
    rtc = (rtp > np.mean(rtp)).astype(int)
    score('RNN', np.where(rtc > best_thr_rnn, rtp, 0.0))

    # ---- LSTM
    i5 = Input(shape=(TIME_STEPS, Xtr.shape[-1]))
    o5 = Dense(1)(LSTM(50, activation='relu')(i5))
    m5m = Model(i5, o5); m5m.compile(optimizer='adam', loss='mean_absolute_error', steps_per_execution=SPE)
    m5m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=50, verbose=0)
    score('LSTM', np.asarray(m5m(Xte, training=False)).flatten())

    # ---- istatistiksel
    score('SES', SimpleExpSmoothing(ytr, initialization_method='estimated').fit(optimized=True).forecast(len(yte)))
    score('Croston', croston_forecast(y[:len(y)-test_size], len(yte)))
    score('SBA', sba_forecast(ytr, extra_periods=len(yte)))

    del m_ctr, m2, m3, m4, m5m
    tf.keras.backend.clear_session()
    gc.collect()

    met = [dict(Product_ID=pid, Model=k, **M[k],
                Best_Alpha_CTR=best_alpha, Best_Threshold_RNN=best_thr_rnn) for k in MODELS]
    pr = pd.DataFrame({'Product_ID': pid, 'y_test': yte,
                       **{f'{k}_pred': P[k] for k in MODELS},
                       'CTR_prob': P['_ctr_prob'], 'CTR_raw': P['_ctr_raw']})
    return met, pr

# ----------------------------------------------------------------- ilerleme
_CNT = _TOT = _T0 = _OUT = _RECENT = None

def pool_init(cnt, tot, t0, outdir, recent=None):
    global _CNT, _TOT, _T0, _OUT, _RECENT
    _CNT, _TOT, _T0, _OUT, _RECENT = cnt, tot, t0, outdir, recent

def _hms(sec):
    sec = int(max(sec, 0)); h, m = divmod(sec, 3600); m, s = divmod(m, 60)
    return f'{h:d}sa {m:02d}dk' if h else f'{m:d}dk {s:02d}sn'

def tick(wid, pid, sku_sec, note=''):
    """Her SKU sonrasi ortak sayaci ilerletip tek satir ilerleme basar."""
    if _CNT is None:
        return
    with _CNT.get_lock():
        _CNT.value += 1
        done = _CNT.value
    el   = time.time() - _T0
    rate = el / done                      # kosu basindan beri ortalama
    now  = time.time()
    rrate = rate
    if _RECENT is not None:               # SON 200 SKU'nun gercek hizi
        try:
            _RECENT.append(now)
            while len(_RECENT) > 200:
                _RECENT.pop(0)
            if len(_RECENT) >= 20:
                span = _RECENT[-1] - _RECENT[0]
                if span > 0:
                    rrate = span / (len(_RECENT) - 1)
        except Exception:
            pass
    eta  = (_TOT - done) * rrate
    pct  = 100.0 * done / _TOT
    bar  = '#' * int(pct / 2.5) + '.' * (40 - int(pct / 2.5))
    print(f'[{bar}] {done:>6}/{_TOT} {pct:5.1f}%  '
          f'w{wid} SKU {pid}  {sku_sec:5.1f}sn  '
          f'guncel {rrate:4.1f}sn (ort {rate:4.1f})  gecen {_hms(el)}  kalan ~{_hms(eta)}{note}', flush=True)
    if done % 25 == 0 or done == _TOT:
        try:
            with open(os.path.join(_OUT, 'progress.txt'), 'w') as f:
                f.write(json.dumps(dict(done=done, total=_TOT, pct=round(pct, 2),
                                        elapsed_s=round(el), eta_s=round(eta),
                                        sec_per_sku=round(rate, 2), sec_per_sku_recent=round(rrate, 2),
                                        updated=time.strftime('%Y-%m-%d %H:%M:%S'))))
        except Exception:
            pass

# ----------------------------------------------------------------- isci
def worker(args):
    wid, pids, series, outdir = args
    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        try: tf.config.set_visible_devices([], 'GPU')
        except Exception: pass
    except Exception as e:
        print('TF yuklenemedi:', e, flush=True); raise
    buf_m, buf_p, k, t0, nok = [], [], 0, time.time(), 0
    for j, pid in enumerate(pids):
        ts = time.time()
        try:
            r = run_one(int(pid), series[j][0], series[j][1])
        except Exception as e:
            tick(wid, pid, time.time()-ts, f'  << HATA {type(e).__name__}: {e}')
            continue
        if r is None:
            tick(wid, pid, time.time()-ts, '  << atlandi (yetersiz veri)')
            continue
        buf_m.extend(r[0]); buf_p.append(r[1]); nok += 1
        tick(wid, pid, time.time()-ts)
    if buf_p:
        pd.DataFrame(buf_m).to_csv(f'{outdir}/parts/met_t{wid}.csv', index=False)
        pd.concat(buf_p).to_csv(f'{outdir}/parts/prd_t{wid}.csv', index=False)
    return wid, nok, time.time()-t0

# ----------------------------------------------------------------- ana
def load_dataset(name, limit=None):
    """meta (URUNID) ve SKU basina (tarihler, satislar) listesi dondurur.

    ONEMLI: BTM tarafinda seriler DUZENSIZ uzunluktadir. son2.py her SKU'nun
    satirlarini CSV'deki haliyle isler; takvim bosluklarini doldurmaz ve
    Lag_1 bir onceki SATIR'dir, bir onceki takvim gunu degil. Ayni davranis
    burada da korunur. M5 matrisi zaten her gun dolu oldugu icin etkilenmez.
    """
    if name == 'm5':
        meta  = pd.read_csv(f'{M5_PREP}/skus.csv')
        dates = pd.read_csv(f'{M5_PREP}/dates.csv')['Date'].tolist()
        mat   = np.load(f'{M5_PREP}/sales.npy')
        if limit:
            meta, mat = meta.iloc[:limit].copy(), mat[:limit]
        series = [(dates, mat[i].astype(np.float64)) for i in range(len(meta))]
    else:
        d = pd.read_csv(BTM_CSV)
        pids, series = [], []
        for pid, grp in d.groupby('URUNID', sort=False):      # dosya sirasi korunur
            pids.append(pid)
            series.append((grp['Date'].tolist(), grp['Satis Adet'].to_numpy(dtype=np.float64)))
        meta = pd.DataFrame({'URUNID': pids})
        if limit:
            meta, series = meta.iloc[:limit].copy(), series[:limit]
    return meta, series


def main():
    global CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['m5','btm'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--chunk', type=int, default=CHUNK,
                    help='bir gorevde islenecek SKU sayisi; her gorevden sonra isci sureci yenilenir')
    ap.add_argument('--shard', default=None,
                    help='i/N -> SKU listesinin sadece i. dilimini isle')
    a = ap.parse_args()

    CHUNK = max(1, a.chunk)
    outdir = os.path.abspath(a.out)
    os.makedirs(f'{outdir}/parts', exist_ok=True)
    print('veri yukleniyor...', flush=True)
    meta, series = load_dataset(a.dataset, a.limit)

    import glob
    done = set()
    for f in glob.glob(f'{outdir}/parts/met_*.csv'):
        try: done |= set(pd.read_csv(f, usecols=['Product_ID']).Product_ID.unique().tolist())
        except Exception: pass
    todo = [i for i, p in enumerate(meta.URUNID.tolist()) if p not in done]
    if a.shard:
        si, sn = (int(x) for x in a.shard.split('/'))
        todo = todo[si::sn]

    W = max(1, a.workers)
    print('=' * 96)
    print(f'  veri seti     : {a.dataset}')
    print(f'  toplam SKU    : {len(meta)}')
    print(f'  bitmis        : {len(done)}')
    print(f'  bu kosuda     : {len(todo)}' + (f'   (dilim {a.shard})' if a.shard else ''))
    print(f'  isci sayisi   : {W}   (makinede {os.cpu_count()} cekirdek)')
    print(f'  cikti         : {outdir}')
    print('=' * 96, flush=True)
    if not todo:
        print('yapilacak is yok - hepsi bitmis.'); return

    # --- GOREVLERI KUCUK PARCALARA BOL ---------------------------------------
    # Her gorev CHUNK kadar SKU isler ve maxtasksperchild=1 sayesinde gorev
    # bitince ISCI SURECI OLUR, yenisi dogar. TensorFlow'un iade etmedigi bellek
    # boylece her parcada isletim sistemine geri verilir; kosu yavaslamaz.
    # Maliyet: parca basina ~5 sn TF import suresi (SKU basina ~0.2 sn).
    tasks = []
    for k, i in enumerate(range(0, len(todo), CHUNK)):
        idxs = todo[i:i + CHUNK]
        tasks.append((k, meta.URUNID.values[idxs], [series[j] for j in idxs], outdir))
    print(f'  gorev sayisi  : {len(tasks)}  ({CHUNK} SKU/gorev, her gorevden sonra isci yenilenir)')
    print('=' * 96, flush=True)

    json.dump(dict(dataset=a.dataset, n_sku=len(meta), todo=len(todo), workers=W,
                   chunk=CHUNK, tasks=len(tasks), limit=a.limit, shard=a.shard,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(f'{outdir}/run_meta.json','w'), indent=2)

    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    cnt = ctx.Value('i', 0)
    mgr = ctx.Manager()
    recent = mgr.list()
    if W == 1:
        pool_init(cnt, len(todo), t0, outdir, recent)
        for j in tasks: worker(j)
    else:
        with ctx.Pool(W, initializer=pool_init,
                      initargs=(cnt, len(todo), t0, outdir, recent),
                      maxtasksperchild=1) as pool:
            for tid, n, el in pool.imap_unordered(worker, tasks):
                pass
    el = time.time() - t0
    print('=' * 96)
    print(f'BITTI  |  {len(todo)} SKU  |  toplam {_hms(el)}  |  {el/max(len(todo),1):.1f} sn/SKU')
    print(f'sonuclar: {outdir}/parts/   (birlestirmek icin: python3 merge_results.py {outdir})')
    print('=' * 96)

if __name__ == '__main__':
    main()
