"""Exp 0: measurement diagnostic.
(a) raw cosine similarity on theta vs residual cosine after two-way (model x category) FE
(b) does cross-firm dispersion shrink mechanically as votes accumulate? noise-corrected version.
"""
import pandas as pd, numpy as np
from itertools import combinations

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d = d[d.category.isin(CORE) & (d.dt >= '2024-06-01')]

# frontier firms only
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
d = d[d.organization.str.lower().isin(FIRMS)]

def two_way_resid(P):
    """P: models x categories matrix. Remove model + category means."""
    R = P - P.mean(axis=1).values[:, None]
    R = R - R.mean(axis=0).values[None, :]
    return R

rows = []
for date, snap in d.groupby('dt'):
    piv = snap.pivot_table(index='model_name', columns='category', values='rating')
    piv = piv.reindex(columns=CORE).dropna()
    if len(piv) < 6:
        continue
    var = snap.pivot_table(index='model_name', columns='category', values='variance')
    var = var.reindex(index=piv.index, columns=CORE)
    org = snap.drop_duplicates('model_name').set_index('model_name').organization.str.lower()

    # keep one model per firm: the best-rated (frontier flagship)
    ov = piv.mean(axis=1)
    keep = ov.groupby(org.reindex(piv.index)).idxmax()
    piv_f, var_f = piv.loc[keep.values], var.loc[keep.values]
    if len(piv_f) < 4:
        continue

    R = two_way_resid(piv_f)
    raw = piv_f.values

    def mean_cos(M):
        cs = []
        for a, b in combinations(range(len(M)), 2):
            u, v = M[a], M[b]
            n = np.linalg.norm(u) * np.linalg.norm(v)
            if n > 0:
                cs.append(u @ v / n)
        return np.mean(cs)

    # dispersion: cross-firm variance of residual profiles, averaged over categories
    disp = R.values.var(axis=0, ddof=1).mean()
    # expected sampling variance contribution (mean of per-model BT variances)
    noise = np.nanmean(var_f.values)
    rows.append(dict(date=date, n_firms=len(piv_f),
                     raw_cos=mean_cos(raw), resid_cos=mean_cos(R.values),
                     disp=disp, noise=noise, disp_corrected=disp - noise,
                     med_votes=np.nanmedian(snap.vote_count)))

res = pd.DataFrame(rows).set_index('date')
print(res.resample('QE').mean().round(4).to_string())

print('\n--- trend tests (OLS on time index) ---')
import numpy.polynomial.polynomial as P
t = (res.index - res.index[0]).days.values / 365.25
for col in ['raw_cos', 'resid_cos', 'disp', 'disp_corrected']:
    y = res[col].values
    m = np.isfinite(y)
    b1, b0 = np.polyfit(t[m], y[m], 1)[0], np.polyfit(t[m], y[m], 1)[1]
    r = np.corrcoef(t[m], y[m])[0, 1]
    print(f'{col:16s} slope/yr={b1:+.4f}  level0={b0:.4f}  r={r:+.3f}')
