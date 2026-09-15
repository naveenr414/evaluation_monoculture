"""Experiment G: compare production functions F_i across labs, via scaling.

Theorem 9 concludes F_i(x) = lambda(x) F_j(x): production functions are PARALLEL up
to a pointwise scale. Experiment D tested that indirectly, through (c,w) perturbations,
because investment x is unobserved on the Arena panel.

Here x is (partially) observed. For open-weight labs we see parameter count, and the
Open LLM Leaderboard gives K = 6 objective capability dimensions. Adopting the paper's
log family (Corollary 6, F_i(x) = A_i log(1+x)) with a level term:

        theta_{i,k} = a_{i,k} + A_{i,k} * log(params)

A_i is then firm i's TECHNOLOGY DIRECTION: how it converts scale into capability
across dimensions. Theorem 9's conclusion is exactly that the A_i are parallel.

  G1  estimate A_i per lab, bootstrap
  G2  pairwise cosine between A_i, with bootstrap CIs -> is parallelism rejected?
  G3  how much of the cross-lab variation in A_i lies along ONE common direction?
  G4  ||A_i|| as a scale-efficiency ranking
"""
import sys, numpy as np, pandas as pd
from itertools import combinations

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']
MIN_N, MIN_SPAN = 7, 0.6

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged]
b = b.dropna(subset=DIMS)
b['lp'] = np.log(b.params_b)

# standardise each dimension across the whole pool so the cosine is not dominated
# by whichever benchmark happens to have the widest raw range
for k in DIMS:
    b[k] = (b[k] - b[k].mean()) / b[k].std()

b = b[b.org.isin(LABS)].copy()
b['is_instruct'] = (b.model_type == 'instruct').astype(float)

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)]
print(f'labs={len(keep)}  models={len(b)}')
print(b.groupby('org').agg(n=('lp', 'size'), sizes=('params_b', 'nunique'),
                           lo=('params_b', 'min'), hi=('params_b', 'max')).round(2).to_string())


def fit_A(g):
    """Slope of each standardised dimension on log(params), controlling for instruct."""
    X = np.column_stack([np.ones(len(g)), g.lp.values, g.is_instruct.values])
    if np.linalg.matrix_rank(X) < 3:
        X = X[:, :2]
    A = []
    for k in DIMS:
        coef, *_ = np.linalg.lstsq(X, g[k].values, rcond=None)
        A.append(coef[1])
    return np.array(A)


print('\n=== G1  technology direction A_i (slope per standardised dim on log params) ===')
print(f'{"lab":16s} ' + ' '.join(f'{k:>9s}' for k in DIMS) + f' {"||A||":>7s}')
A, labs_used = {}, []
for org, g in b.groupby('org'):
    a = fit_A(g)
    A[org] = a
    labs_used.append(org)
    print(f'{org:16s} ' + ' '.join(f'{v:9.3f}' for v in a) + f' {np.linalg.norm(a):7.3f}')

M = np.array([A[o] for o in labs_used])

# ---- bootstrap ----
B = 2000
rng = np.random.default_rng(0)
boots = {o: [] for o in labs_used}
for _ in range(B):
    for org, g in b.groupby('org'):
        idx = rng.integers(0, len(g), len(g))
        try:
            boots[org].append(fit_A(g.iloc[idx]))
        except Exception:
            boots[org].append(A[org])
boots = {o: np.array(v) for o, v in boots.items()}

print('\n=== G2  pairwise cosine between technology directions ===')
print('(1.00 = perfectly parallel = Theorem 9 conclusion holds pointwise)')


def cos(u, v):
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return u @ v / n if n > 0 else np.nan


pairs = []
for i, j in combinations(range(len(labs_used)), 2):
    o1, o2 = labs_used[i], labs_used[j]
    c = cos(A[o1], A[o2])
    bc = np.array([cos(boots[o1][t], boots[o2][t]) for t in range(B)])
    lo, hi = np.nanpercentile(bc, [2.5, 97.5])
    pairs.append((o1, o2, c, lo, hi))

pairs.sort(key=lambda p: -p[2])
for o1, o2, c, lo, hi in pairs:
    flag = 'PARALLEL' if hi >= 0.95 else ('rejects parallel' if hi < 0.90 else '')
    print(f'  {o1:15s} {o2:15s} cos={c:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {flag}')

cs = np.array([p[2] for p in pairs])
print(f'\nmean pairwise cosine = {cs.mean():+.3f}   median = {np.median(cs):+.3f}')
n_rej = sum(1 for p in pairs if p[4] < 0.90)
print(f'pairs whose CI excludes near-parallelism (upper < 0.90): {n_rej}/{len(pairs)}')

print('\n=== G3  is there ONE common scaling direction? ===')
Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
u, s, vt = np.linalg.svd(Mn - 0 * Mn.mean(0), full_matrices=False)
ev = s ** 2 / (s ** 2).sum()
print(f'variance share of leading direction : {ev[0]:.3f}')
print(f'  cumulative first two              : {ev[:2].sum():.3f}')
print(f'  participation ratio (eff. dims)   : {(s**2).sum()**2/((s**2)**2).sum():.2f}')
lead = vt[0] * np.sign(vt[0] @ Mn.mean(0))
print('  leading direction loadings: ' + ', '.join(f'{k}={v:+.2f}' for k, v in zip(DIMS, lead)))

print('\n=== G4  scale efficiency ||A_i|| (capability gained per log-param) ===')
for o in sorted(labs_used, key=lambda o: -np.linalg.norm(A[o])):
    nb = np.linalg.norm(boots[o], axis=1)
    print(f'  {o:16s} ||A||={np.linalg.norm(A[o]):.3f}  '
          f'95% CI [{np.percentile(nb,2.5):.3f}, {np.percentile(nb,97.5):.3f}]')
