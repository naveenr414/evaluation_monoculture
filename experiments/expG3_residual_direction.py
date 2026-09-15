"""Experiment G3: is the parallelism of A_i real, or an artefact of the benchmarks
sharing one general-capability factor?

If all six benchmarks load on a single general factor g, and scale raises g, then
every lab's slope vector A_i is automatically proportional to g's loadings and the
cosines are near 1 by construction -- the same trap as raw cosine on theta.

Fix: project each normalised A_i onto the orthogonal complement of the common
direction, and ask whether the RESIDUAL directions differ across labs by more than
bootstrap error allows.
"""
import sys, numpy as np, pandas as pd
from itertools import combinations

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged].dropna(subset=DIMS)
b['lp'] = np.log(b.params_b)
b['t'] = (pd.to_datetime(b.upload_date) - pd.Timestamp('2023-01-01')).dt.days / 365.25
for k in DIMS:
    b[k] = (b[k] - b[k].mean()) / b[k].std()
b = b[b.org.isin(LABS)].dropna(subset=['t']).copy()
b['is_instruct'] = (b.model_type == 'instruct').astype(float)
keep = [o for o, g in b.groupby('org')
        if len(g) >= 8 and np.log10(g.params_b.max()/g.params_b.min()) >= 0.6
        and g.params_b.nunique() >= 4]
b = b[b.org.isin(keep)]

# --- how much of the benchmark space is one general factor? ---
C = np.corrcoef(b[DIMS].values.T)
lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
print('benchmark correlation spectrum (share of variance):')
print('  ' + ', '.join(f'{v:.3f}' for v in lam/lam.sum()))
print(f'  leading general factor = {lam[0]/lam.sum():.1%} of benchmark variance')
print(f'  effective dims of the benchmark battery = {lam.sum()**2/(lam**2).sum():.2f} of {len(DIMS)}\n')


def fit_A(g):
    X = np.column_stack([np.ones(len(g)), g.lp.values, g.is_instruct.values, g.t.values])
    if np.linalg.matrix_rank(X) < 4:
        return None
    return np.array([np.linalg.lstsq(X, g[k].values, rcond=None)[0][1] for k in DIMS])


B, rng = 2000, np.random.default_rng(0)
A, boots = {}, {}
for org, g in b.groupby('org'):
    a = fit_A(g)
    if a is None:
        continue
    A[org] = a

boots = {}
for org, g in b.groupby('org'):
    if org not in A:
        continue
    bs = []
    for _ in range(B):
        r = fit_A(g.iloc[rng.integers(0, len(g), len(g))])
        bs.append(r if r is not None else A[org])
    boots[org] = np.array(bs)

strong = [o for o in A if np.percentile(np.linalg.norm(boots[o], axis=1), 2.5) > 0.5]
print(f'labs with a measurable scaling response: {len(strong)}')

U = np.array([A[o]/np.linalg.norm(A[o]) for o in strong])
gdir = U.mean(0); gdir /= np.linalg.norm(gdir)
print(f'common direction: ' + ', '.join(f'{k}={v:+.2f}' for k, v in zip(DIMS, gdir)))

# residual directions, orthogonal to the common direction
R = U - np.outer(U @ gdir, gdir)
share_common = (U @ gdir)**2
print(f'\nshare of each unit A_i lying ALONG the common direction:')
for o, s in sorted(zip(strong, share_common), key=lambda x: -x[1]):
    print(f'  {o:16s} {s:.3f}   residual norm {np.linalg.norm(R[strong.index(o)]):.3f}')

print(f'\nmean share along common direction = {share_common.mean():.3f}')
print(f'mean residual norm                = {np.linalg.norm(R,axis=1).mean():.3f}')

# Is the residual structure real, or bootstrap noise?
obs = np.linalg.norm(R, axis=1).mean()
nb = []
for t in range(B):
    Ub = np.array([boots[o][t]/np.linalg.norm(boots[o][t]) for o in strong])
    gb = Ub.mean(0); gb /= np.linalg.norm(gb)
    Rb = Ub - np.outer(Ub @ gb, gb)
    nb.append(np.linalg.norm(Rb, axis=1).mean())
print(f'  bootstrap 95% CI on mean residual norm = '
      f'[{np.percentile(nb,2.5):.3f}, {np.percentile(nb,97.5):.3f}]')

# permutation-style null: how big would residuals be if labs differed ONLY by noise?
# resample each lab's A around the COMMON direction, using its own bootstrap spread
null = []
for t in range(B):
    Un = []
    for o in strong:
        sd = boots[o].std(0)
        v = gdir*np.linalg.norm(A[o]) + rng.standard_normal(len(DIMS))*sd
        Un.append(v/np.linalg.norm(v))
    Un = np.array(Un)
    gn = Un.mean(0); gn /= np.linalg.norm(gn)
    Rn = Un - np.outer(Un @ gn, gn)
    null.append(np.linalg.norm(Rn, axis=1).mean())
p = (np.sum(np.array(null) >= obs)+1)/(B+1)
print(f'  null (labs identical up to estimation noise): mean {np.mean(null):.3f}'
      f'  95th pct {np.percentile(null,95):.3f}')
print(f'  observed {obs:.3f}   p = {p:.4f}')
print('\n  p < 0.05  => labs genuinely differ in technology DIRECTION, not just scale')
print('  p >= 0.05 => differences are consistent with one shared direction + noise')

# pairwise cosine in the residual subspace
pr = [(o1, o2, R[strong.index(o1)] @ R[strong.index(o2)] /
       (np.linalg.norm(R[strong.index(o1)])*np.linalg.norm(R[strong.index(o2)])))
      for o1, o2 in combinations(strong, 2)]
cs = np.array([x[2] for x in pr])
print(f'\nmean pairwise cosine in the RESIDUAL subspace = {cs.mean():+.3f}'
      f'  (mechanical null approx -1/(n-1) = {-1/(len(strong)-1):+.3f})')
