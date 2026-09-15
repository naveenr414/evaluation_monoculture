"""Experiment J: effective rank of a 40-dimension battery, at scale.

Experiment I asked whether the low effective rank of public evaluation is a property
of the INSTRUMENT or of the MODELS, but had only 44 models x 24 BBH subtasks -- too
few to estimate a 24x24 correlation matrix reliably.

Here: ~1000+ models x 40 leaf subtasks pulled from the Open LLM Leaderboard details
repos (24 BBH tasks, 7 MATH subjects, 3 GPQA, 3 MuSR, IFEval, MMLU-Pro, ARC).

J1  effective rank of the full battery, with a properly calibrated null
J2  does rank keep rising as resolution rises? (6 aggregates -> 40 leaves)
J3  which task families are genuinely orthogonal?
"""
import numpy as np, pandas as pd

df = pd.read_parquet('data/ollm_subtasks.parquet')
df = df.drop_duplicates(subset='run_file', keep='first')
TASKS = [c for c in df.columns if c.startswith('leaderboard_') and c != 'leaderboard_arc_challenge']
X = df[TASKS].astype(float)
X = X.dropna(axis=1, thresh=int(0.9 * len(X)))
TASKS = list(X.columns)
X = X.dropna()
print(f'models={len(X)}  leaf subtasks={len(TASKS)}')


def eff_rank(M):
    R = M - M.mean(1, keepdims=True)
    R = R - R.mean(0, keepdims=True)
    C = np.nan_to_num(np.corrcoef(R.T), nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    return lam.sum() ** 2 / (lam ** 2).sum(), lam / lam.sum()


V = X.values
er, spec = eff_rank(V)
print(f'\n=== J1  effective rank ===')
print(f'nominal dimensions   = {len(TASKS)}')
print(f'effective dimensions = {er:.2f}')
print('top-10 variance shares: ' + ', '.join(f'{v:.3f}' for v in spec[:10]))
print(f'top 3 carry {spec[:3].sum():.1%};  top 5 carry {spec[:5].sum():.1%}')

rng = np.random.default_rng(0)
nb = [eff_rank(rng.standard_normal(V.shape))[0] for _ in range(200)]
print(f'pure-noise null (same shape) = {np.mean(nb):.2f} (sd {np.std(nb):.2f})')

# one-latent-factor + binomial measurement error null
sim = []
for _ in range(200):
    g = rng.standard_normal((len(V), 1)) * V.std()
    P = np.clip(V.mean(0)[None, :] + g, 0.01, 0.99)
    sim.append(eff_rank(rng.binomial(250, P) / 250)[0])
print(f'single-latent-factor + noise null = {np.mean(sim):.2f} (sd {np.std(sim):.2f})')

print('\n=== J2  does resolution buy dimensions? ===')
fam = {}
for t in TASKS:
    if 'bbh' in t: fam[t] = 'bbh'
    elif 'math' in t: fam[t] = 'math'
    elif 'gpqa' in t: fam[t] = 'gpqa'
    elif 'musr' in t: fam[t] = 'musr'
    else: fam[t] = t.replace('leaderboard_', '')
fams = sorted(set(fam.values()))
agg = np.column_stack([X[[t for t in TASKS if fam[t] == f]].mean(axis=1).values for f in fams])
er_agg, _ = eff_rank(agg)
print(f'  {len(fams)} family aggregates  -> {er_agg:.2f} effective dims')
print(f'  {len(TASKS)} leaf subtasks     -> {er:.2f} effective dims')
print(f'  gain from finer measurement: {er/er_agg:.2f}x')
print('  (Experiment C, Arena 8 categories -> 3.81;  G3, 6 OLLM aggregates -> 1.84;'
      '\n   Experiment I, 24 BBH subtasks n=44 -> 5.31)')

for f in fams:
    cols = [t for t in TASKS if fam[t] == f]
    if len(cols) > 2:
        e, _ = eff_rank(X[cols].values)
        print(f'    within {f:10s} ({len(cols):2d} tasks): {e:.2f} effective dims')

print('\n=== J3  most orthogonal task pairs (residual correlation) ===')
R = V - V.mean(1, keepdims=True); R = R - R.mean(0, keepdims=True)
C = np.corrcoef(R.T)
iu = np.triu_indices_from(C, 1)
print(f'residual correlations: mean={C[iu].mean():+.3f}  median={np.median(C[iu]):+.3f}'
      f'  min={C[iu].min():+.3f}  max={C[iu].max():+.3f}')
print(f'share of pairs with |r| < 0.2: {np.mean(np.abs(C[iu])<0.2):.1%}')
pairs = sorted(zip(C[iu], [(TASKS[i], TASKS[j]) for i, j in zip(*iu)]))
print('  most NEGATIVELY correlated (genuine trade-offs):')
for r, (a, b) in pairs[:6]:
    print(f'    r={r:+.3f}  {a.replace("leaderboard_","")} / {b.replace("leaderboard_","")}')

# which single tasks add the most independent information?
print('\n  tasks least explained by the rest of the battery (highest unique variance):')
uniq = []
for i, t in enumerate(TASKS):
    others = np.delete(R, i, axis=1)
    coef, *_ = np.linalg.lstsq(others, R[:, i], rcond=None)
    r2 = 1 - ((R[:, i] - others @ coef) ** 2).sum() / (R[:, i] ** 2).sum()
    uniq.append((r2, t))
for r2, t in sorted(uniq)[:8]:
    print(f'    R2={r2:.3f}  {t.replace("leaderboard_","")}')
