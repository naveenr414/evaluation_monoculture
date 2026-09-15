import numpy as np, pandas as pd
import sys
sys.path.insert(0, 'experiments')
from common import CORE
from scipy import stats

PANEL = 'data/leaderboard/text/full-00000-of-00001.parquet'
d = pd.read_parquet(PANEL)
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.lower()
d = d[d.dt >= '2024-01-01']

ov = d[d.category == 'overall'].dropna(subset=['rating'])
cat = d[d.category.isin(CORE)]
cat_piv = cat.pivot_table(index=['dt','model_name'], columns='category', values='rating')

snaps = np.sort(d.dt.unique())
rng = np.random.default_rng(0)
N_PERM = 999

rows = []
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
    Mc = M - M.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(Mc, axis=1)
    U = Mc / row_norms[:, None]
    mean_dir = U.mean(axis=0); mean_dir /= np.linalg.norm(mean_dir)
    obs = (U @ mean_dir).mean()

    null_draws = np.empty(N_PERM)
    for p in range(N_PERM):
        Mp = np.array([rng.permutation(row) for row in Mc])
        Up = Mp / np.linalg.norm(Mp, axis=1, keepdims=True)
        md = Up.mean(axis=0); md /= np.linalg.norm(md)
        null_draws[p] = (Up @ md).mean()

    p_emp = (np.sum(null_draws >= obs) + 1) / (N_PERM + 1)
    z = (obs - null_draws.mean()) / null_draws.std(ddof=1)
    rows.append((t, obs, null_draws.mean(), null_draws.std(ddof=1), z, p_emp))

out = pd.DataFrame(rows, columns=['dt','obs','null_mean','null_sd','z','p_emp'])
out.to_csv('experiments/scratch_plots/top10_cosine_test.csv', index=False)

print(f"n snapshots = {len(out)}")
print(f"snapshots with p_emp < 1/(N_PERM+1) (obs exceeds ALL permutations): {(out.p_emp <= 1/(N_PERM+1)).sum()} / {len(out)}")
print(f"median z-score = {out.z.median():.1f}")
print(out[['dt','obs','null_mean','z','p_emp']].describe())

# ---- aggregate test correcting for autocorrelation: thin to ~monthly, roughly independent ----
out2 = out.set_index('dt')
monthly = out2.resample('MS').first().dropna()
print(f"\nthinned to {len(monthly)} ~monthly, roughly-independent snapshots")
diff = monthly.obs - monthly.null_mean
t_stat, p_t = stats.ttest_1samp(diff, 0)
w_stat, p_w = stats.wilcoxon(diff)
print(f"paired t-test (obs - null_mean > 0): t={t_stat:.2f}, p={p_t:.2e}")
print(f"Wilcoxon signed-rank: W={w_stat:.1f}, p={p_w:.2e}")
print(f"mean excess = {diff.mean():.3f}  sd excess = {diff.std():.3f}  Cohen's d = {diff.mean()/diff.std():.2f}")
print(f"min excess across monthly snapshots = {diff.min():.3f}")
