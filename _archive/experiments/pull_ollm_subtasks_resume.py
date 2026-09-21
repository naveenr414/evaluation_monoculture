"""Resumable, rate-limited continuation of the Open LLM subtask pull.

HF allows 1000 /api/ requests per 5 minutes. The file-resolve endpoint appears not to
count against that, so the metadata call is the binding constraint: throttle to ~3/s.
Appends to data/ollm_subtasks.parquet and checkpoints as it goes.
"""
import os, time, threading
import requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TOK = os.environ.get('HF_TOKEN2') or os.environ.get('HF_TOKEN')
OUT = 'data/ollm_subtasks.parquet'
LIST = '/tmp/det_list.txt'
API_RPS = 3.0

METRICS = ['acc_norm,none', 'exact_match,none', 'acc,none',
           'prompt_level_strict_acc,none', 'inst_level_strict_acc,none']
DROP = {'leaderboard', 'leaderboard_bbh', 'leaderboard_gpqa',
        'leaderboard_math_hard', 'leaderboard_musr'}

sess = requests.Session()
sess.headers.update({'Authorization': f'Bearer {TOK}'})

_lock = threading.Lock()
_next = [0.0]


def throttle():
    with _lock:
        now = time.time()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + 1.0 / API_RPS
    if wait:
        time.sleep(wait)


def api_get(url, tries=6):
    for a in range(tries):
        throttle()
        try:
            r = sess.get(url, timeout=60)
        except Exception:
            time.sleep(2 * (a + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            time.sleep(20 * (a + 1)); continue
        return None
    return None


def fetch(ds):
    r = api_get(f'https://huggingface.co/api/datasets/{ds}')
    if r is None:
        return None
    try:
        files = [s['rfilename'] for s in r.json().get('siblings', [])
                 if '/results_' in s['rfilename'] and s['rfilename'].endswith('.json')]
        if not files:
            return None
        f = sorted(files)[-1]
        for a in range(4):
            d = sess.get(f'https://huggingface.co/datasets/{ds}/resolve/main/{f}', timeout=90)
            if d.status_code == 200:
                break
            if d.status_code == 429:
                time.sleep(15 * (a + 1)); continue
            return None
        else:
            return None
        js = d.json()
        res = js.get('results', {})
        cfg = js.get('config_general', {}) or js.get('config', {})
        row = {'dataset': ds,
               'model': ds[len('open-llm-leaderboard/'):-len('-details')].replace('__', '/'),
               'run_file': f}
        for k in ('model_name', 'model_dtype', 'model_size', 'start_time'):
            if k in cfg:
                row[f'cfg_{k}'] = cfg[k]
        for task, vals in res.items():
            if task in DROP or not isinstance(vals, dict):
                continue
            for m in METRICS:
                if m in vals and vals[m] is not None:
                    row[task] = float(vals[m]); break
        return row
    except Exception:
        return None


def main():
    have = set()
    old = pd.DataFrame()
    if os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        have = set(old.dataset)
    ids = [l.strip() for l in open(LIST) if l.strip() and l.strip() not in have]
    print(f'have {len(have)}, fetching {len(ids)}', flush=True)

    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, d): d for d in ids}
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            if r:
                rows.append(r)
            if done % 200 == 0:
                el = time.time() - t0
                print(f'  {done}/{len(ids)} ok={len(rows)} {el:.0f}s '
                      f'eta {(len(ids)-done)/max(done/el,1e-9)/60:.1f}min', flush=True)
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_parquet(OUT, index=False)
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(OUT, index=False)
    tc = [c for c in df.columns if c.startswith('leaderboard_')]
    print(f'\nDONE: {df.shape[0]} models, {len(tc)} leaf subtasks', flush=True)


if __name__ == '__main__':
    main()
