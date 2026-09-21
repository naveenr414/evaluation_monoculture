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
rng = np.random.default_rng(0)
for t in snaps:
    ov_t = ov[ov.dt == t]
    if ov_t.empty:
        continue
    best = ov_t.sort_values('rating').groupby('org').tail(1).set_index('org')
    vecs = {}
    for org, row in best.iterrows():
        key = (t, row.model_name)
        if key in cat_piv.index:
            v = cat_piv.loc[key][CORE]
            if not v.isna().any():
                vecs[org] = v.values.astype(float)
    if len(vecs) < 10:
        continue
    orgs_ranked = best.loc[list(vecs.keys())].sort_values('rating', ascending=False).index[:10]
    M = np.array([vecs[o] for o in orgs_ranked])

    # row-demean: strip each firm's own average level, keep only relative category shape
    Mc = M - M.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(Mc, axis=1)
    U = Mc / row_norms[:, None]
    mean_dir = U.mean(axis=0)
    mdn = np.linalg.norm(mean_dir)
    if mdn < 1e-9:
        continue
    mean_dir = mean_dir / mdn
    cos = U @ mean_dir

    # permutation null: shuffle each firm's category labels independently -> destroys
    # any shared category shape while preserving each firm's own spread
    null_coss = []
    for _ in range(200):
        Mp = np.array([rng.permutation(row) for row in Mc])
        Up = Mp / np.linalg.norm(Mp, axis=1, keepdims=True)
        md = Up.mean(axis=0)
        mdnp = np.linalg.norm(md)
        if mdnp < 1e-9:
            null_coss.append(np.nan); continue
        md = md / mdnp
        null_coss.append((Up @ md).mean())
    null_mean = np.nanmean(null_coss)

    rows.append((t, cos.mean(), cos.std(), null_mean, len(vecs)))

out = pd.DataFrame(rows, columns=['dt','mean_cos','sd_cos','null_cos','n_avail'])
out.to_csv('experiments/scratch_plots/top10_cosine_demeaned.csv', index=False)
print(out.describe())
print(out.head())
print(out.tail())
