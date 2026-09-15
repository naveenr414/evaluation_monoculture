"""Experiment J3: does finer measurement buy real dimensions?

Matched comparison -- SAME models, SAME convention, three resolutions:
    6 family aggregates  vs  24 BBH leaves  vs  39 leaves
Effective rank is a participation ratio: 1 = one dimension, N = evenly spread.
LOWER means MORE concentrated. Each resolution is benchmarked against its own
single-latent-factor + binomial-noise null, since noise alone inflates apparent rank.
"""
import numpy as np, pandas as pd

df = pd.read_parquet('data/ollm_subtasks.parquet')
df = df.drop_duplicates(subset='run_file', keep='first')
TASKS = [c for c in df.columns if c.startswith('leaderboard_') and c != 'leaderboard_arc_challenge']
X = df[TASKS].astype(float).dropna(axis=1, thresh=int(0.9 * len(df)))
TASKS = list(X.columns); X = X.dropna()
print(f'models={len(X)}\n')


def fam(t):
    for f in ('bbh', 'math', 'gpqa', 'musr'):
        if f in t:
            return f
    return t.replace('leaderboard_', '')


FAMS = sorted({fam(t) for t in TASKS})
AGG = np.column_stack([X[[t for t in TASKS if fam(t) == f]].mean(axis=1).values for f in FAMS])
BBH = X[[t for t in TASKS if fam(t) == 'bbh']].values
ALL = X.values


def er(M):
    R = M - M.mean(0, keepdims=True)
    C = np.nan_to_num(np.corrcoef(R.T), nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    return lam.sum() ** 2 / (lam ** 2).sum(), lam / lam.sum()


rng = np.random.default_rng(0)


def null(M, n=200):
    out = []
    for _ in range(n):
        g = rng.standard_normal((len(M), 1)) * M.std()
        P = np.clip(M.mean(0)[None, :] + g, 0.01, 0.99)
        out.append(er(rng.binomial(250, P) / 250)[0])
    return np.mean(out), np.std(out)


print(f'{"battery":24s} {"dims":>5s} {"eff rank":>9s} {"1-factor null":>14s} '
      f'{"excess":>8s} {"top-1 share":>12s}')
rows = []
for name, M in [(f'{len(FAMS)} family aggregates', AGG),
                (f'{BBH.shape[1]} BBH leaves', BBH),
                (f'{ALL.shape[1]} leaves (all)', ALL)]:
    o, spec = er(M)
    nm, ns = null(M)
    print(f'{name:24s} {M.shape[1]:5d} {o:9.2f} {nm:11.2f}±{ns:.2f} {o-nm:+8.2f} {spec[0]:12.3f}')
    rows.append((name, M.shape[1], o, nm, o - nm))

print('\nexcess = observed rank above what one latent ability + measurement noise'
      '\nwould produce at that resolution. Excess is the REAL extra structure.')

print('\n=== how much does each extra dimension actually buy? ===')
o, spec = er(ALL)
cum = np.cumsum(spec)
for i in [1, 2, 3, 5, 10, 20]:
    print(f'  first {i:2d} components carry {cum[i-1]:.1%} of variance')

print('\n=== the practical question: can a board of K categories separate firms? ===')
print('A board can only rank firms on dimensions that exist. With a leading component')
print(f'at {spec[0]:.1%} and {o:.2f} effective dimensions from {ALL.shape[1]} nominal tasks,')
print('the ceiling on how much differentiation ANY leaderboard built from this kind of')
print('battery can express is roughly the excess-rank column above.')

# does the extra structure survive removing the general factor?
R = ALL - ALL.mean(0, keepdims=True)
R = R / R.std(0, keepdims=True)
u, s, vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
g = u[:, 0:1] * s[0] @ vt[0:1]
resid = R - g
C = np.corrcoef(resid.T)
lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
print(f'\nafter projecting out the single leading factor:')
print(f'  effective rank of what remains = {lam.sum()**2/(lam**2).sum():.2f}')
print(f'  next 3 components carry {lam[:3].sum()/lam.sum():.1%} of the remaining variance')
