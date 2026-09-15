"""Exp V: is similarity among the TOP-K firms specifically rising over time?

Same two-way-FE + permutation-null estimand as exp0b, restricted at each
snapshot to the top-k firms by mean rating (proxy for overall), for
k in {3,4,5,8}. Tracks excess-over-null and z over time per k.
"""
import pandas as pd, numpy as np
from itertools import combinations

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
KS = [3, 4, 5, 8]

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

rng = np.random.default_rng(0)
all_rows = []
for date, snap in d.groupby('dt'):
    piv = snap.pivot_table(index='model_name', columns='category', values='rating').reindex(columns=CORE).dropna()
    if len(piv) < 6: continue
    org = snap.drop_duplicates('model_name').set_index('model_name').organization.str.lower()
    org_aligned = org.reindex(piv.index)
    # one flagship per firm = highest mean rating for that firm
    keep = piv.mean(axis=1).groupby(org_aligned).idxmax()
    firm_of_keep = org_aligned.loc[keep.values]
    level = piv.loc[keep.values].mean(axis=1)  # proxy for 'overall' level, used to rank firms

    order = level.sort_values(ascending=False).index  # model names, best first

    for k in KS:
        if len(order) < k: continue
        top_models = order[:k]
        P = piv.loc[top_models].values
        N, K = P.shape
        if N < 3: continue

        R2 = P - P.mean(1, keepdims=True); R2 = R2 - R2.mean(0, keepdims=True)

        null = []
        for _ in range(200):
            Rp = np.column_stack([rng.permutation(R2[:, j]) for j in range(K)])
            null.append(mean_cos(Rp))
        resid_cos = mean_cos(R2)
        null_mean, null_sd = np.mean(null), np.std(null)
        all_rows.append(dict(date=date, k=k, N=N,
                              resid_cos=resid_cos, null_mean=null_mean, null_sd=null_sd,
                              excess=resid_cos - null_mean,
                              z=(resid_cos - null_mean)/null_sd if null_sd > 0 else np.nan))

r = pd.DataFrame(all_rows)

print(f'{"k":>3} {"n_snaps":>8} {"mean_resid_cos":>15} {"mean_null":>10} {"mean_excess":>12} {"mean_z":>8} {"slope_excess/yr":>16} {"r":>7} {"slope_z/yr":>11}')
for k, sub in r.groupby('k'):
    sub = sub.sort_values('date')
    t = (sub.date - sub.date.iloc[0]).dt.days.values / 365.25
    b_excess, *_ = np.polyfit(t, sub.excess.values, 1)
    rr = np.corrcoef(t, sub.excess.values)[0, 1] if len(sub) > 2 else np.nan
    b_z, *_ = np.polyfit(t, sub.z.values, 1) if sub.z.notna().sum() > 2 else (np.nan,)
    print(f'{k:>3} {len(sub):>8} {sub.resid_cos.mean():>15.4f} {sub.null_mean.mean():>10.4f} '
          f'{sub.excess.mean():>12.4f} {sub.z.mean():>8.2f} {b_excess:>16.4f} {rr:>7.3f} {b_z:>11.3f}')

print('\nQuarterly excess-over-null by k:')
piv_q = r.pivot_table(index=pd.Grouper(key='date', freq='QE'), columns='k', values='excess')
print(piv_q.round(4).to_string())

print('\nQuarterly z by k:')
piv_qz = r.pivot_table(index=pd.Grouper(key='date', freq='QE'), columns='k', values='z')
print(piv_qz.round(2).to_string())
