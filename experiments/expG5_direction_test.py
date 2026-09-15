"""Experiment G5: properly calibrated test of whether labs differ in technology direction.

G4c built its null by adding INDEPENDENT per-coordinate noise to each of 39 coefficients.
But the 39 subtasks are heavily correlated, so estimation errors are correlated too, and
an independent-noise null overstates how much direction spread noise alone would create
(hence the degenerate p = 1.0).

Fix: build the null from each lab's own bootstrap DEVIATIONS, which carry the correct
error covariance by construction. Under H0 every lab shares the common direction and
differs only by its own estimation error.
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


def fit_A(g):
    X = np.column_stack([np.ones(len(g)), g.lp.values, g.is_instruct.values, g.t.values])
    if np.linalg.matrix_rank(X) < 4:
        X = np.column_stack([np.ones(len(g)), g.lp.values, g.t.values])
        if np.linalg.matrix_rank(X) < 3:
            return None
    return np.array([np.linalg.lstsq(X, g[k].values, rcond=None)[0][1] for k in TASKS])


B, rng = 1500, np.random.default_rng(0)
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
print(f'dims={len(TASKS)}  labs tested={len(strong)}  models={len(d)}\n')

U = np.array([A[o] / np.linalg.norm(A[o]) for o in strong])
gdir = U.mean(0); gdir /= np.linalg.norm(gdir)
R = U - np.outer(U @ gdir, gdir)
obs = np.linalg.norm(R, axis=1).mean()

# properly calibrated null: reuse each lab's own bootstrap deviations
dev = {o: boots[o] - A[o] for o in strong}
null = []
for t in range(B):
    Un = []
    for o in strong:
        v = gdir * np.linalg.norm(A[o]) + dev[o][rng.integers(0, B)]
        Un.append(v / np.linalg.norm(v))
    Un = np.array(Un)
    gn = Un.mean(0); gn /= np.linalg.norm(gn)
    Rn = Un - np.outer(Un @ gn, gn)
    null.append(np.linalg.norm(Rn, axis=1).mean())
null = np.array(null)
p = (np.sum(null >= obs) + 1) / (B + 1)
print('=== direction test, correlated-error null ===')
print(f'  observed mean residual direction norm = {obs:.4f}')
print(f'  null mean = {null.mean():.4f}  sd = {null.std():.4f}'
      f'  95th pct = {np.percentile(null,95):.4f}')
print(f'  p = {p:.4f}')
print('  p < 0.05 => labs genuinely differ in technology DIRECTION\n')

# same test on the mean pairwise cosine, which is the more readable statistic
def mc(M):
    return np.mean([M[a] @ M[b] / (np.linalg.norm(M[a]) * np.linalg.norm(M[b]))
                    for a, b in combinations(range(len(M)), 2)])


obs_c = mc(U)
nullc = []
for t in range(B):
    Un = np.array([(gdir * np.linalg.norm(A[o]) + dev[o][rng.integers(0, B)]) for o in strong])
    nullc.append(mc(Un))
nullc = np.array(nullc)
pc = (np.sum(nullc <= obs_c) + 1) / (B + 1)
print('=== same test on mean pairwise cosine ===')
print(f'  observed mean cosine = {obs_c:+.4f}')
print(f'  null (identical direction + own estimation error) = {nullc.mean():+.4f}'
      f'  5th pct = {np.percentile(nullc,5):+.4f}')
print(f'  p = {pc:.4f}   (low p => labs LESS aligned than shared-direction noise allows)')

print('\n=== per-lab: how far is each lab from the common direction, vs its own noise? ===')
print(f'{"lab":16s} {"resid norm":>11s} {"own-noise null":>15s} {"z":>7s}')
for i, o in enumerate(strong):
    nb = []
    for t in range(300):
        v = gdir * np.linalg.norm(A[o]) + dev[o][rng.integers(0, B)]
        v = v / np.linalg.norm(v)
        nb.append(np.linalg.norm(v - (v @ gdir) * gdir))
    z = (np.linalg.norm(R[i]) - np.mean(nb)) / (np.std(nb) + 1e-12)
    print(f'{o:16s} {np.linalg.norm(R[i]):11.4f} {np.mean(nb):15.4f} {z:+7.2f}')
