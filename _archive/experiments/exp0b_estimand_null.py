"""Exp 0b: is two-way-FE residual cosine mechanically pinned at -1/(N-1)?
And what estimand actually carries signal?
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
        n = np.linalg.norm(u)*np.linalg.norm(v)
        if n > 0: cs.append(u@v/n)
    return np.mean(cs)

rows = []
rng = np.random.default_rng(0)
for date, snap in d.groupby('dt'):
    piv = snap.pivot_table(index='model_name', columns='category', values='rating').reindex(columns=CORE).dropna()
    if len(piv) < 6: continue
    org = snap.drop_duplicates('model_name').set_index('model_name').organization.str.lower()
    keep = piv.mean(axis=1).groupby(org.reindex(piv.index)).idxmax()
    P = piv.loc[keep.values].values
    N, K = P.shape
    if N < 4: continue

    # A. two-way FE residual
    R2 = P - P.mean(1, keepdims=True); R2 = R2 - R2.mean(0, keepdims=True)
    # B. category-FE only (keeps firm level -> dominated by quality again)
    R1 = P - P.mean(0, keepdims=True)
    # C. profile shares: each firm's within-firm allocation, then category-demeaned
    S = P - P.mean(1, keepdims=True)          # within-firm centered (removes quality)
    Sc = S - S.mean(0, keepdims=True)          # == R2 actually

    # mechanical null for two-way: permute residuals within each category independently
    null = []
    for _ in range(200):
        Rp = np.column_stack([rng.permutation(R2[:, k]) for k in range(K)])
        null.append(mean_cos(Rp))
    rows.append(dict(date=date, N=N,
                     resid_cos=mean_cos(R2),
                     null_mean=np.mean(null), null_sd=np.std(null),
                     analytic_null=-1/(N-1),
                     catFE_cos=mean_cos(R1)))

r = pd.DataFrame(rows).set_index('date')
r['excess'] = r.resid_cos - r.null_mean
r['z'] = r.excess / r.null_sd
print(r.resample('QE').mean().round(4).to_string())
print('\nOverall: mean resid_cos=%.4f  mean analytic null=%.4f  mean excess=%.4f  mean z=%.2f'
      % (r.resid_cos.mean(), r.analytic_null.mean(), r.excess.mean(), r.z.mean()))
t = (r.index - r.index[0]).days.values/365.25
for c in ['resid_cos','excess','z','catFE_cos']:
    b = np.polyfit(t, r[c].values, 1)[0]; rr = np.corrcoef(t, r[c].values)[0,1]
    print(f'{c:12s} slope/yr={b:+.4f} r={rr:+.3f}')
