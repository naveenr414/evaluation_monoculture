import numpy as np, pandas as pd
import sys
sys.path.insert(0, 'experiments')
from common import CORE

PANEL = 'data/leaderboard/text/full-00000-of-00001.parquet'
d = pd.read_parquet(PANEL)
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.lower()
d = d[d.dt >= '2024-01-01']

ov = d[d.category == 'overall'].dropna(subset=['rating'])
cat = d[d.category.isin(CORE)]

cat_piv = cat.pivot_table(index=['dt','model_name'], columns='category', values='rating')

snaps = np.sort(d.dt.unique())
rows = []
for t in snaps:
    ov_t = ov[ov.dt == t]
    if ov_t.empty:
        continue
    # best model per org at this snapshot, by overall rating
    best = ov_t.sort_values('rating').groupby('org').tail(1).set_index('org')
    # need complete CORE ratings for that model
    vecs = {}
    for org, row in best.iterrows():
        key = (t, row.model_name)
        if key in cat_piv.index:
            v = cat_piv.loc[key][CORE]
            if not v.isna().any():
                vecs[org] = v.values.astype(float)
    if len(vecs) < 10:
        continue
    # top-10 orgs by overall rating among those with complete category vectors
    orgs_ranked = best.loc[list(vecs.keys())].sort_values('rating', ascending=False).index[:10]
    M = np.array([vecs[o] for o in orgs_ranked])
    U = M / np.linalg.norm(M, axis=1, keepdims=True)
    mean_dir = U.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)
    cos = U @ mean_dir
    rows.append((t, cos.mean(), cos.std(), len(vecs)))

out = pd.DataFrame(rows, columns=['dt','mean_cos','sd_cos','n_avail'])
out.to_csv('experiments/scratch_plots/top10_cosine_over_time.csv', index=False)
print(out.describe())
print(out.head())
print(out.tail())
