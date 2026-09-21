"""Experiment J2: corrected rank analysis.

Two fixes to J:
 (1) Two-way demeaning makes every task the exact negative sum of the others, so the
     "unique variance" regression is degenerate (R^2 = 1 for everything) and the
     extreme negative correlations are that artefact. Redo with TASK demeaning only.
 (2) Report effective rank under both conventions, each against a matched null:
       column-demeaned : total dimensionality, general ability included
       two-way         : dimensionality of SPECIALISATION, general ability removed
     The single-latent-factor + binomial-noise null is the one that matters: it says
     how many dimensions a battery would APPEAR to have if models differed on exactly
     one ability and everything else were measurement error.
"""
import numpy as np, pandas as pd

df = pd.read_parquet('data/ollm_subtasks.parquet')
df = df.drop_duplicates(subset='run_file', keep='first')
TASKS = [c for c in df.columns if c.startswith('leaderboard_') and c != 'leaderboard_arc_challenge']
X = df[TASKS].astype(float)
X = X.dropna(axis=1, thresh=int(0.9 * len(X)))
TASKS = list(X.columns)
X = X.dropna()
V = X.values
print(f'models={len(V)}  leaf subtasks={len(TASKS)}\n')


def er(M, two_way):
    R = M - M.mean(0, keepdims=True)
    if two_way:
        R = M - M.mean(1, keepdims=True)
        R = R - R.mean(0, keepdims=True)
    C = np.nan_to_num(np.corrcoef(R.T), nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    return lam.sum() ** 2 / (lam ** 2).sum(), lam / lam.sum()


rng = np.random.default_rng(0)
NSIM = 200
print(f'{"convention":18s} {"observed":>9s} {"1-factor+noise null":>20s} {"pure noise":>11s}')
for tw in (False, True):
    o, spec = er(V, tw)
    sim, nz = [], []
    for _ in range(NSIM):
        g = rng.standard_normal((len(V), 1)) * V.std()
        P = np.clip(V.mean(0)[None, :] + g, 0.01, 0.99)
        sim.append(er(rng.binomial(250, P) / 250, tw)[0])
        nz.append(er(rng.standard_normal(V.shape), tw)[0])
    name = 'two-way' if tw else 'task-demeaned'
    print(f'{name:18s} {o:9.2f} {np.mean(sim):14.2f} ±{np.std(sim):.2f} {np.mean(nz):11.2f}')
    print(f'{"":18s} top-5 variance shares: ' + ', '.join(f'{v:.3f}' for v in spec[:5]))

print('\nReading: if observed is at or below the 1-factor null, the battery carries no')
print('more independent structure than a single ability plus measurement error.')

# ---- corrected uniqueness: task-demeaned only, so tasks are not collinear ----
print('\n=== which tasks carry information the rest of the battery does not? ===')
R = V - V.mean(0, keepdims=True)
R = R / R.std(0, keepdims=True)
uniq = []
for i, t in enumerate(TASKS):
    others = np.delete(R, i, axis=1)
    coef, *_ = np.linalg.lstsq(others, R[:, i], rcond=None)
    resid = R[:, i] - others @ coef
    r2 = 1 - (resid ** 2).sum() / (R[:, i] ** 2).sum()
    uniq.append((r2, t))
uniq.sort()
print('  least predictable from the others (most unique):')
for r2, t in uniq[:8]:
    print(f'    R2={r2:.3f}  {t.replace("leaderboard_","")}')
print('  most redundant:')
for r2, t in uniq[-5:]:
    print(f'    R2={r2:.3f}  {t.replace("leaderboard_","")}')

# ---- corrected correlations ----
C = np.corrcoef(R.T)
iu = np.triu_indices_from(C, 1)
print(f'\ntask correlations (task-demeaned): mean={C[iu].mean():+.3f}'
      f'  median={np.median(C[iu]):+.3f}  min={C[iu].min():+.3f}  max={C[iu].max():+.3f}')
print(f'  share |r| < 0.2: {np.mean(np.abs(C[iu])<0.2):.1%}')
pairs = sorted(zip(C[iu], [(TASKS[i], TASKS[j]) for i, j in zip(*iu)]))
print('  genuinely least correlated pairs:')
for r, (a, b) in pairs[:6]:
    print(f'    r={r:+.3f}  {a.replace("leaderboard_","")} / {b.replace("leaderboard_","")}')

# ---- how many models do you need to see the rank? (relevant to Experiment I's n=44) ----
print('\n=== sample-size sensitivity (Experiment I had n=44) ===')
for n in [44, 100, 250, 500, len(V)]:
    if n > len(V):
        continue
    vals = [er(V[rng.choice(len(V), n, replace=False)], True)[0] for _ in range(60)]
    print(f'  n={n:5d}: two-way effective rank = {np.mean(vals):.2f} (sd {np.std(vals):.2f})')
print('  -> small samples INFLATE apparent rank; Experiment I\'s 5.31 at n=44 was biased up.')
