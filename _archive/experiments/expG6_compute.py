"""Experiment G6: convert scale efficiency from per-parameter to per-FLOP.

||A_i|| from G/G4 is capability gained per log-PARAMETER. That is not efficiency: it
ignores how many tokens a lab trains on. With C = 6ND,

    log C = log N + log D + const,   and empirically  log D = b_i log N + const_i

so   log C = (1 + b_i) log N + const_i,  and the slope of capability on log C is

    A_i^{compute} = A_i^{params} / (1 + b_i).

Epoch AI's notable-models data gives b_i per lab, and it varies a lot (Google 0.14,
Meta 0.81, DeepSeek 1.47), so this correction re-ranks labs rather than just rescaling.

b_i is itself an allocation choice -- how a lab splits a scale increment between
parameters and data -- which is exactly the kind of allocation difference the paper
is about, sitting underneath an apparently shared technology.
"""
import numpy as np, pandas as pd

# Epoch organization -> HF org used in the G experiments
ORGMAP = {'alibaba': 'qwen', 'meta ai': 'meta-llama', 'google': 'google',
          'deepseek': 'deepseek-ai', 'microsoft': 'microsoft', 'nvidia': 'nvidia',
          'deepmind': 'google', 'google research': 'google', 'google brain': 'google',
          'mistral ai': 'mistralai', 'allen institute for ai': 'allenai',
          'technology innovation institute': 'tiiuae', 'eleutherai': 'eleutherai',
          'stability ai': 'stabilityai', '01.ai': '01-ai', 'ibm': 'ibm-granite'}

e = pd.read_csv('data/epoch_notable_models.csv', low_memory=False)
e['org'] = e.Organization.astype(str).str.lower()
e['N'] = pd.to_numeric(e.Parameters, errors='coerce')
e['D'] = pd.to_numeric(e['Training dataset size (total)'], errors='coerce')
e = e[(e.N > 1e8) & (e.D > 1e9)].copy()
e['hf'] = e.org.map(ORGMAP)
e = e.dropna(subset=['hf'])
e['lN'], e['lD'] = np.log(e.N), np.log(e.D)

print('=== token-scaling exponent b_i = dlogD/dlogN, from Epoch ===')
print(f'{"lab":14s} {"n":>3s} {"b_i":>6s} {"1+b_i":>7s} {"median D/N":>11s}')
B = {}
for hf, g in e.groupby('hf'):
    if len(g) < 4:
        continue
    b = np.polyfit(g.lN, g.lD, 1)[0]
    b = float(np.clip(b, 0.0, 2.0))
    B[hf] = b
    print(f'{hf:14s} {len(g):3d} {b:6.2f} {1+b:7.2f} {np.median(g.D/g.N):11.0f}')
pooled = float(np.clip(np.polyfit(e.lN, e.lD, 1)[0], 0, 2))
print(f'{"POOLED":14s} {len(e):3d} {pooled:6.2f} {1+pooled:7.2f}')

# ---- apply the correction to the 39-dim A_i estimates ----
LABS = list(B) + ['ibm-granite', 'stabilityai', 'mistralai', 'allenai', 'tiiuae',
                  'eleutherai', '01-ai']
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


def fit_A(g):
    X = np.column_stack([np.ones(len(g)), g.lp.values, g.is_instruct.values, g.t.values])
    if np.linalg.matrix_rank(X) < 4:
        X = np.column_stack([np.ones(len(g)), g.lp.values, g.t.values])
        if np.linalg.matrix_rank(X) < 3:
            return None
    return np.array([np.linalg.lstsq(X, g[k].values, rcond=None)[0][1] for k in TASKS])


rows = []
for org, g in d.groupby('org'):
    a = fit_A(g)
    if a is None:
        continue
    n = np.linalg.norm(a)
    b = B.get(org, pooled)
    rows.append(dict(lab=org, n_models=len(g), per_param=n, b_i=b,
                     per_flop=n / (1 + b), imputed=org not in B))
r = pd.DataFrame(rows).sort_values('per_param', ascending=False)
r['rank_param'] = r.per_param.rank(ascending=False).astype(int)
r['rank_flop'] = r.per_flop.rank(ascending=False).astype(int)
r['shift'] = r.rank_param - r.rank_flop

print('\n=== efficiency: per log-parameter vs per log-FLOP ===')
print(f'{"lab":14s} {"n":>4s} {"||A||/logN":>11s} {"b_i":>6s} {"||A||/logC":>11s}'
      f' {"rank N":>7s} {"rank C":>7s} {"shift":>6s}  src')
for _, x in r.iterrows():
    print(f'{x.lab:14s} {x.n_models:4d} {x.per_param:11.2f} {x.b_i:6.2f} '
          f'{x.per_flop:11.2f} {x.rank_param:7d} {x.rank_flop:7d} {x["shift"]:+6d}'
          f'  {"imputed" if x.imputed else "Epoch"}')

meas = r[~r.imputed].copy()
print(f'\n--- labs with a genuine Epoch-measured b_i (n={len(meas)}) ---')
if len(meas) > 1:
    meas['rp'] = meas.per_param.rank(ascending=False).astype(int)
    meas['rf'] = meas.per_flop.rank(ascending=False).astype(int)
    for _, x in meas.sort_values('per_flop', ascending=False).iterrows():
        print(f'  {x.lab:14s} per-param rank {x.rp} -> per-FLOP rank {x.rf}'
              f'   ({x.per_param:.2f} -> {x.per_flop:.2f}, b={x.b_i:.2f})')
    print(f'  Spearman among measured labs = '
          f'{meas[["rp","rf"]].corr(method="spearman").iloc[0,1]:.3f}')
    print(f'  spread in (1+b_i) across measured labs = '
          f'{(1+meas.b_i).max()/(1+meas.b_i).min():.2f}x')
print('\nOnly the "Epoch" rows are measurements. The "imputed" rows share the pooled')
print('exponent, so their per-FLOP values are a constant rescaling and their rank')
print('shifts carry no information -- do not report them as findings.')
