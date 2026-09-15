"""Experiment G4: compare production functions across labs at 39 dimensions.

G/G2/G3 compared technology directions A_i in a 6-dimensional aggregate space whose
own effective rank was under 2, so "no detectable difference in direction" was partly
a statement about the instrument. Redo at 39 leaf subtasks.

    theta_{i,k} = a_{i,k} + A_{i,k} log(params)     (controls: instruct, release date)

Theorem 9's conclusion is that the A_i are parallel across firms.
  G4a  pairwise cosine between A_i, bootstrapped
  G4b  share of each A_i along the common direction
  G4c  do residual directions differ by more than estimation noise?
"""
import numpy as np, pandas as pd
from itertools import combinations

LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'bigscience', 'databricks', 'stabilityai',
        'internlm', 'upstage', 'openbmb', 'thudm', 'baichuan-inc']

sub = pd.read_parquet('data/ollm_subtasks.parquet')
sub = sub.drop_duplicates(subset='run_file', keep='first')
TASKS = [c for c in sub.columns if c.startswith('leaderboard_') and c != 'leaderboard_arc_challenge']
sub = sub.dropna(axis=1, thresh=int(0.9 * len(sub)))
TASKS = [c for c in TASKS if c in sub.columns]
sub['key'] = sub.model.str.lower()

meta = pd.read_parquet('data/benchmarks.parquet')
meta['key'] = meta.hf_model_id.str.lower()
meta = meta[meta.params_b.notna() & (meta.params_b > 0) & ~meta.flagged & ~meta.merged]

d = sub.merge(meta[['key', 'params_b', 'upload_date', 'model_type', 'hf_model_id']],
              on='key', how='inner').dropna(subset=TASKS)
d['org'] = d.hf_model_id.str.split('/').str[0].str.lower()
d['lp'] = np.log(d.params_b)
d['t'] = (pd.to_datetime(d.upload_date) - pd.Timestamp('2023-01-01')).dt.days / 365.25
d['is_instruct'] = (d.model_type == 'instruct').astype(float)
d = d.dropna(subset=['t'])
for k in TASKS:
    d[k] = (d[k] - d[k].mean()) / d[k].std()

keep = [o for o, g in d.groupby('org')
        if o in LABS and len(g) >= 8 and g.params_b.nunique() >= 4
        and np.log10(g.params_b.max() / g.params_b.min()) >= 0.6]
d = d[d.org.isin(keep)]
print(f'dimensions={len(TASKS)}  labs={len(keep)}  models={len(d)}')
print(d.groupby('org').agg(n=('lp', 'size'), sizes=('params_b', 'nunique'),
                           lo=('params_b', 'min'), hi=('params_b', 'max')).round(2).to_string())


def fit_A(g):
    X = np.column_stack([np.ones(len(g)), g.lp.values, g.is_instruct.values, g.t.values])
    if np.linalg.matrix_rank(X) < 4:
        X = np.column_stack([np.ones(len(g)), g.lp.values, g.t.values])
        if np.linalg.matrix_rank(X) < 3:
            return None
    return np.array([np.linalg.lstsq(X, g[k].values, rcond=None)[0][1] for k in TASKS])


B, rng = 1000, np.random.default_rng(0)
A, boots = {}, {}
for org, g in d.groupby('org'):
    a = fit_A(g)
    if a is None:
        continue
    A[org] = a
    bs = []
    for _ in range(B):
        r = fit_A(g.iloc[rng.integers(0, len(g), len(g))])
        bs.append(r if r is not None else a)
    boots[org] = np.array(bs)

strong = [o for o in A if np.percentile(np.linalg.norm(boots[o], axis=1), 2.5) > 1.0]
print(f'\nlabs with a measurable scaling response: {len(strong)}/{len(A)}')
for o in sorted(A, key=lambda o: -np.linalg.norm(A[o])):
    nb = np.linalg.norm(boots[o], axis=1)
    print(f'  {o:16s} ||A||={np.linalg.norm(A[o]):6.2f}  '
          f'CI [{np.percentile(nb,2.5):5.2f},{np.percentile(nb,97.5):6.2f}]'
          f'  {"yes" if o in strong else "NO"}')


def cos(u, v):
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return u @ v / n if n > 0 else np.nan


print('\n=== G4a  pairwise cosine between technology directions (39-dim) ===')
pr = [(a, b, cos(A[a], A[b])) for a, b in combinations(strong, 2)]
for a, b, c in sorted(pr, key=lambda x: -x[2]):
    print(f'  {a:15s} {b:15s} cos={c:+.3f}')
cs = np.array([p[2] for p in pr])
print(f'\nmean={cs.mean():+.3f}  median={np.median(cs):+.3f}  min={cs.min():+.3f}')
mb = [np.nanmean([cos(boots[a][t], boots[b][t]) for a, b in combinations(strong, 2)])
      for t in range(B)]
print(f'95% CI on mean cosine = [{np.percentile(mb,2.5):+.3f}, {np.percentile(mb,97.5):+.3f}]')
print(f'  (at 6 dims this was +0.920)')

print('\n=== G4b  share of each A_i along the common direction ===')
U = np.array([A[o] / np.linalg.norm(A[o]) for o in strong])
gdir = U.mean(0); gdir /= np.linalg.norm(gdir)
sh = (U @ gdir) ** 2
for o, s in sorted(zip(strong, sh), key=lambda x: -x[1]):
    print(f'  {o:16s} {s:.3f}')
print(f'  mean = {sh.mean():.3f}   (at 6 dims this was 0.929)')

R = U - np.outer(U @ gdir, gdir)
obs = np.linalg.norm(R, axis=1).mean()
print('\n=== G4c  do residual directions differ beyond estimation noise? ===')
null = []
for t in range(B):
    Un = []
    for o in strong:
        sd = boots[o].std(0)
        v = gdir * np.linalg.norm(A[o]) + rng.standard_normal(len(TASKS)) * sd
        Un.append(v / np.linalg.norm(v))
    Un = np.array(Un)
    gn = Un.mean(0); gn /= np.linalg.norm(gn)
    Rn = Un - np.outer(Un @ gn, gn)
    null.append(np.linalg.norm(Rn, axis=1).mean())
p = (np.sum(np.array(null) >= obs) + 1) / (B + 1)
print(f'  observed mean residual norm = {obs:.3f}')
print(f'  null (identical direction + noise) = {np.mean(null):.3f} '
      f'(95th pct {np.percentile(null,95):.3f})')
print(f'  p = {p:.4f}')
print('  p < 0.05 => labs genuinely differ in technology DIRECTION')
print('  (at 6 dims: observed 0.242, null 0.265, p = 0.75 -> no detectable difference)')
