"""Pull per-subtask Open LLM Leaderboard v2 results for every evaluated model.

Each open-llm-leaderboard/<org>__<model>-details repo carries a results_*.json with
~39 LEAF subtask scores (24 BBH tasks, 7 MATH subjects, 3 GPQA, 3 MuSR, IFEval,
MMLU-Pro, ARC) instead of the 6 aggregates in data/benchmarks.parquet.

Writes data/ollm_subtasks.parquet.

Token is read from HF_TOKEN2 (or HF_TOKEN) in the environment; it is never written
to disk by this script.
"""
import os, sys, json, time
import requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TOK = os.environ.get('HF_TOKEN2') or os.environ.get('HF_TOKEN')
H = {'Authorization': f'Bearer {TOK}'}
OUT = 'data/ollm_subtasks.parquet'
LIST = '/tmp/det_list.txt'

# metric preference order: first one present wins
METRICS = ['acc_norm,none', 'exact_match,none', 'acc,none',
           'prompt_level_strict_acc,none', 'inst_level_strict_acc,none']
# aggregates to drop, keeping only leaves
DROP = {'leaderboard', 'leaderboard_bbh', 'leaderboard_gpqa',
        'leaderboard_math_hard', 'leaderboard_musr'}

sess = requests.Session()
sess.headers.update(H)


def fetch(ds):
    try:
        r = sess.get(f'https://huggingface.co/api/datasets/{ds}', timeout=45)
        if r.status_code != 200:
            return None
        j = r.json()
        files = [s['rfilename'] for s in j.get('siblings', [])
                 if '/results_' in s['rfilename'] and s['rfilename'].endswith('.json')]
        if not files:
            return None
        f = sorted(files)[-1]                      # latest run
        d = sess.get(f'https://huggingface.co/datasets/{ds}/resolve/main/{f}',
                     timeout=60)
        if d.status_code != 200:
            return None
        js = d.json()
        res = js.get('results', {})
        cfg = js.get('config_general', {}) or js.get('config', {})
        row = {'dataset': ds,
               'model': ds[len('open-llm-leaderboard/'):-len('-details')].replace('__', '/'),
               'run_file': f}
        for k, v in cfg.items():
            if k in ('model_name', 'model_dtype', 'model_size', 'start_time'):
                row[f'cfg_{k}'] = v
        for task, vals in res.items():
            if task in DROP or not isinstance(vals, dict):
                continue
            for m in METRICS:
                if m in vals and vals[m] is not None:
                    row[task] = float(vals[m])
                    break
        return row
    except Exception:
        return None


def main():
    ids = [l.strip() for l in open(LIST) if l.strip()]
    print(f'{len(ids)} repos to pull', flush=True)
    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(fetch, d): d for d in ids}
        for fu in as_completed(futs):
            r = fu.result()
            done += 1
            if r:
                rows.append(r)
            if done % 250 == 0:
                el = time.time() - t0
                print(f'  {done}/{len(ids)}  ok={len(rows)}  {el:.0f}s '
                      f'({done/el:.1f}/s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f'\nwrote {OUT}: {df.shape[0]} models x {df.shape[1]} columns', flush=True)
    task_cols = [c for c in df.columns if c.startswith('leaderboard_')]
    print(f'leaf subtasks captured: {len(task_cols)}', flush=True)
    print(df[task_cols].notna().sum().sort_values(ascending=False).head(45).to_string(),
          flush=True)


if __name__ == '__main__':
    main()
