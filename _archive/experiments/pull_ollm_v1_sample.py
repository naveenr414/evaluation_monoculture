"""Scoped feasibility pull: for a sample of models we already have v2-battery
scores for (data/ollm_subtasks.parquet), fetch their OLD (v1, pre-June-2024)
Open LLM Leaderboard scores from the still-archived open-llm-leaderboard-old
org. Static open-weight checkpoints -> theta is literally frozen across this
w-shock, so any change in relative standing between v1 and v2 batteries is
100% attributable to the benchmark/weight swap, not new investment.

This is a feasibility-scale pull (not the full population): ~200 models,
respecting the documented HF rate limit (~3 req/s).
"""
import json, time, sys
import requests
import pandas as pd
import numpy as np

TOKEN = open('hf_token.txt').read().strip()
H = {'Authorization': f'Bearer {TOKEN}'}
API = 'https://huggingface.co/api/datasets/open-llm-leaderboard-old/results/tree/main'

LABS = ['Qwen', 'google', 'meta-llama', 'mistralai', 'microsoft', 'ibm-granite', 'ibm',
        'allenai', 'deepseek-ai', '01-ai', 'tiiuae', 'EleutherAI', 'bigscience',
        'databricks', 'NousResearch', 'teknium', 'HuggingFaceH4', 'stabilityai',
        'openchat', 'upstage', 'berkeley-nest']

v2 = pd.read_parquet('data/ollm_subtasks.parquet')
v2_leaf_cols = [c for c in v2.columns if c.startswith('leaderboard_')]
v2['v2_avg'] = v2[v2_leaf_cols].mean(axis=1)
v2['org'] = v2.model.str.split('/').str[0]
models = v2[v2.org.isin(LABS)][['model', 'org', 'v2_avg']].dropna().drop_duplicates('model')
models = models.reset_index(drop=True)
print(f'{len(models)} lab-affiliated models for v1 lookup (not a random sample -- restricted '
      f'to official releases from known open-weight labs, far more likely to have been '
      f'submitted during v1\'s run through June 2024)')

V1_KEYS = {
    'harness|arc:challenge|25': 'arc',
    'harness|hellaswag|10': 'hellaswag',
    'harness|hendrycksTest-*': 'mmlu',
    'harness|truthfulqa:mc|0': 'truthfulqa',
    'harness|winogrande|5': 'winogrande',
    'harness|gsm8k|5': 'gsm8k',
}


def get_v1(model_id, sess):
    r = sess.get(f'{API}/{model_id}', headers=H, timeout=15)
    if r.status_code != 200:
        return None
    files = [f['path'] for f in r.json() if f['path'].endswith('.json') and 'results_' in f['path']]
    if not files:
        return None
    fpath = sorted(files)[-1]  # latest run
    rr = sess.get(f'https://huggingface.co/datasets/open-llm-leaderboard-old/results/resolve/main/{fpath}',
                  headers=H, timeout=15)
    if rr.status_code != 200:
        return None
    try:
        j = rr.json()
    except Exception:
        return None
    res = j.get('results', {})
    vals = []
    for task, sub in res.items():
        for k, v in sub.items():
            if k.endswith(('_norm', 'acc_norm')) or k in ('acc', 'mc2'):
                if isinstance(v, (int, float)):
                    vals.append(v)
    return float(np.mean(vals)) if vals else None


sess = requests.Session()
rows = []
t0 = time.time()
for i, m in models.iterrows():
    v1 = get_v1(m.model, sess)
    if v1 is not None:
        rows.append(dict(model=m.model, org=m.org, v1_avg=v1, v2_avg=m.v2_avg))
    if (i + 1) % 20 == 0:
        print(f'{i+1}/{len(models)}  matched so far: {len(rows)}  elapsed {time.time()-t0:.0f}s')
    time.sleep(0.35)

out = pd.DataFrame(rows)
out.to_parquet('data/ollm_v1_v2_matched.parquet')
print(f'\nmatched {len(out)}/{len(models)} models with a v1 result')
print(out.head(10))
