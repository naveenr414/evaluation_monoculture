"""Experiment I: is the low effective rank a property of the INSTRUMENT or of the MODELS?

Experiment C found Arena's 8 categories collapse to 3.81 effective dimensions.
Experiment G3 found the Open LLM battery's 6 aggregates collapse to 1.84.
Both could be artefacts of coarse aggregation.

Here we use 24 genuinely distinct BBH reasoning subtasks over 44 models -- already
on disk. If 24 fine-grained tasks ALSO collapse to a handful of dimensions, the low
rank is a fact about models, not about how the board aggregates.

Noise works AGAINST the finding: per-subtask binomial error ADDS apparent dimensions,
so a low measured rank despite noise is a conservative result.
"""
import numpy as np, pandas as pd

d = pd.read_csv('data/bbh_subtask_scores.csv').rename(columns={'Unnamed: 0': 'model'})
TASKS = [c for c in d.columns if c not in ('model', 'year', 'bbh_agg')]
X = d[TASKS].astype(float).values
print(f'models={X.shape[0]}  subtasks={X.shape[1]}')
print(f'mean score={np.nanmean(X):.3f}')

X = X[~np.isnan(X).any(axis=1)]
print(f'complete rows: {X.shape[0]}')


def eff_rank(M, demean_both=True):
    R = M - M.mean(1, keepdims=True) if demean_both else M.copy()
    R = R - R.mean(0, keepdims=True)
    C = np.corrcoef(R.T)
    C = np.nan_to_num(C, nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    return lam.sum() ** 2 / (lam ** 2).sum(), lam / lam.sum()


er, spec = eff_rank(X)
print(f'\nnominal dimensions        = {X.shape[1]}')
print(f'effective dimensions      = {er:.2f}')
print('variance shares (top 8)   : ' + ', '.join(f'{v:.3f}' for v in spec[:8]))
print(f'top 3 carry {spec[:3].sum():.1%} of residual variance')

# pure-noise null of the same shape (noise INFLATES rank, so this is an upper bound)
rng = np.random.default_rng(0)
nb = [eff_rank(rng.standard_normal(X.shape))[0] for _ in range(300)]
print(f'effective dims under pure noise = {np.mean(nb):.2f} (sd {np.std(nb):.2f})')

# how much apparent rank does binomial sampling error alone create?
# BBH subtasks have roughly 250 items each
n_items = 250
sim = []
for _ in range(300):
    truth = X.mean(0)[None, :] + (X - X.mean(0)).mean() * 0
    # rank-1 latent model: one general ability + binomial measurement error
    g = rng.standard_normal((X.shape[0], 1)) * X.std()
    P = np.clip(X.mean(0)[None, :] + g, 0.01, 0.99)
    obs = rng.binomial(n_items, P) / n_items
    sim.append(eff_rank(obs)[0])
print(f'effective dims if models truly differed on ONE latent factor'
      f' + binomial noise = {np.mean(sim):.2f} (sd {np.std(sim):.2f})')

print('\n--- comparison across instruments ---')
print(f'  Arena, 8 categories        -> 3.81 effective dims  (Experiment C)')
print(f'  Open LLM, 6 aggregates     -> 1.84 effective dims  (Experiment G3)')
print(f'  BBH, {X.shape[1]} fine subtasks     -> {er:.2f} effective dims  (here)')
print('\nIf the fine-grained battery lands near the coarse ones, aggregation is not')
print('the bottleneck: model capability really is close to low-dimensional, and the')
print('"add more categories" lever cannot work. If it lands much higher, the boards')
print('are throwing away resolution they could keep.')

# per-pair correlations: are ANY subtasks genuinely orthogonal?
R = X - X.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
C = np.corrcoef(R.T)
iu = np.triu_indices_from(C, 1)
print(f'\nresidual subtask correlations: mean={C[iu].mean():+.3f} '
      f'median={np.median(C[iu]):+.3f}  min={C[iu].min():+.3f}  max={C[iu].max():+.3f}')
print(f'  share of pairs with |r| < 0.2 : {np.mean(np.abs(C[iu])<0.2):.1%}')
lo = sorted(zip(C[iu], [(TASKS[i], TASKS[j]) for i, j in zip(*iu)]))[:5]
print('  most orthogonal pairs:')
for r, (a, b) in lo:
    print(f'    r={r:+.3f}  {a} / {b}')
