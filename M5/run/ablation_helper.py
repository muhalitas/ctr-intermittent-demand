"""Helper for a running M5 ablation (ctr_ablation.py, 8 workers).

The running workers process their lists (series_ids[w::8]) from the front.
This helper runs extra processes that take series from the BACK of the longest
remaining lists, using the unchanged run_one() of ctr_ablation.py (same seed,
one CPU thread, no GPU). Results go to separate files abl_w{8+k}.csv /
prd_w{8+k}.csv; duplicates are removed by ctr_ablation.merge().

  cd M5/run
  caffeinate -dims ~/.m5venv/bin/python ablation_helper.py --helpers 4

When it prints ALL DONE, stop the main run (Ctrl+C in its window).
"""
import os, sys, time, argparse, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.abspath(os.path.join(HERE, '..', 'ablation_m5'))
MAIN_WORKERS = 8


def lists_and_done():
    import pandas as pd
    import ctr_ablation as AB
    ids = pd.read_csv(f'{OUT}/series_ids.csv').Product_ID.tolist()
    return [ids[w::MAIN_WORKERS] for w in range(MAIN_WORKERS)], AB.done_ids(OUT), ids


def helper(hid, lock, claimed):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try: tf.config.set_visible_devices([], 'GPU')
    except Exception: pass
    import numpy as np, pandas as pd
    import ctr_ablation as AB, ctr_experiment as CE
    meta, series = CE.load_dataset('m5', None)[:2]
    pos = {int(p): i for i, p in enumerate(meta['URUNID'])}
    wid = MAIN_WORKERS + hid
    path, ppath = f'{OUT}/parts/abl_w{wid}.csv', f'{OUT}/preds/prd_w{wid}.csv'
    while True:
        with lock:
            lists, done, _ = lists_and_done()
            best = None
            for L in lists:
                rem = [p for p in L if p not in done and p not in claimed]
                # rem[0] is (or is about to be) in progress in the main worker: never take it
                if len(rem) >= 2 and (best is None or len(rem) > best[0]):
                    best = (len(rem), rem[-1])
            if best is None:
                break
            pid = best[1]; claimed[pid] = hid
        ts = time.time()
        i = pos[int(pid)]
        r = AB.run_one(int(pid), series[i][0], series[i][1])
        if r:
            r[1].to_csv(ppath, mode='a', header=not os.path.exists(ppath), index=False)
            pd.DataFrame(r[0]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        print(f'helper {hid}: series {pid} done in {time.time() - ts:5.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--helpers', type=int, default=4)
    a = ap.parse_args()
    lists, done, ids = lists_and_done()
    print(f'{len(ids) - len(done)} series left; starting {a.helpers} helpers', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Manager() as man:
        lock, claimed = man.Lock(), man.dict()
        ps = [ctx.Process(target=helper, args=(k, lock, claimed)) for k in range(a.helpers)]
        for p in ps: p.start()
        for p in ps: p.join()
    while True:
        lists, done, ids = lists_and_done()
        left = len(ids) - len(done)
        if left == 0:
            print('ALL DONE: every series is finished. Stop the main run with Ctrl+C in its window.', flush=True)
            break
        print(f'helpers finished; waiting for the main workers ({left} series in progress)', flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
