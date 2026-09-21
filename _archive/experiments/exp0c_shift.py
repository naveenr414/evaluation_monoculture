"""RQ1.1b: cosine similarity as a function of the BT anchor C.

Arena's `rating` = (400/ln10)*beta + C, where beta is the BT coefficient and C is an
arbitrary display anchor (BT is identified only up to an additive constant). Cosine is
invariant to the common scale but NOT to C. This sweeps C to show that raw-theta cosine
is a free parameter of the display convention, and reports it at natural choices of C.

Sample construction is copied verbatim from exp0_measurement.py so the C=0 row
reproduces the 0.9999 headline.
"""
import pandas as pd, numpy as np
from itertools import combinations

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d = d[d.category.isin(CORE) & (d.dt >= '2024-06-01')]
d = d[d.organization.str.lower().isin(FIRMS)]


def mean_cos(M):
    cs = []
    for a, b in combinations(range(len(M)), 2):
        u, v = M[a], M[b]
        n = np.linalg.norm(u) * np.linalg.norm(v)
        if n > 0:
            cs.append(u @ v / n)
    return np.mean(cs)


# ---- build one flagship-per-firm matrix per snapshot (exp0 construction) ----
mats, dates = [], []
for date, snap in d.groupby('dt'):
    piv = snap.pivot_table(index='model_name', columns='category', values='rating')
    piv = piv.reindex(columns=CORE).dropna()
    if len(piv) < 6:
        continue
    org = snap.drop_duplicates('model_name').set_index('model_name').organization.str.lower()
    keep = piv.mean(axis=1).groupby(org.reindex(piv.index)).idxmax()
    P = piv.loc[keep.values].values
    if len(P) < 4:
        continue
    mats.append(P); dates.append(date)

print(f'snapshots={len(mats)}  mean firms/snapshot={np.mean([len(m) for m in mats]):.1f}')
print(f'rating level: mean={np.mean([m.mean() for m in mats]):.1f}  '
      f'within-snapshot sd={np.mean([m.std() for m in mats]):.1f}\n')

# ---- named choices of C ----
print('=== cosine at natural choices of the anchor C ===')
named = {
    'C = 0            (raw, as published)': lambda P: np.zeros(1),
    'C = 1000         (round anchor)':      lambda P: np.array([1000.0]),
    'C = grand mean   (global centering)':  lambda P: np.array([P.mean()]),
    'C_k = category mean (per-cat anchor)': lambda P: P.mean(0),
}
for name, f in named.items():
    v = [mean_cos(P - f(P)[None, :] if f(P).size > 1 else P - f(P)[0]) for P in mats]
    print(f'{name:38s} mean cos = {np.mean(v):+.4f}   (min {np.min(v):+.4f}, max {np.max(v):+.4f})')

# ---- sweep ----
print('\n=== sweep: mean pairwise cosine vs global anchor C ===')
print(f'{"C":>8s} {"mean cos":>10s}')
for C in [0, 500, 800, 1000, 1100, 1150, 1200, 1250, 1275, 1300, 1325, 1350, 1400]:
    print(f'{C:8d} {np.mean([mean_cos(P - C) for P in mats]):+10.4f}')

# ---- trend over time at each named choice ----
print('\n=== trend (slope/yr) ===')
idx = pd.DatetimeIndex(dates)
t = (idx - idx[0]).days.values / 365.25
for name, f in named.items():
    y = np.array([mean_cos(P - f(P)[None, :] if f(P).size > 1 else P - f(P)[0]) for P in mats])
    b = np.polyfit(t, y, 1)[0]; r = np.corrcoef(t, y)[0, 1]
    print(f'{name:38s} slope/yr={b:+.4f}  r={r:+.3f}')
