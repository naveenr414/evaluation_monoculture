"""Experiment C: is the board too coarse to see differentiation?

C1 factor structure: how many INDEPENDENT dimensions live in the 8 core categories?
   Effective rank via participation ratio of the eigenvalue spectrum of the
   category correlation matrix (computed on two-way residuals, broad model set),
   benchmarked against the spectrum expected from sampling noise alone.
C2 fine vs coarse partition: from 2025-11-05 the 8 industry_* categories slice the
   same battles more finely. If firms differentiate INSIDE what the coarse board
   lumps together, the fine partition should show more cross-firm structure.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import (load_panel, two_way_resid, firm_profiles,
                    CORE, INDUSTRY, FRONTIER)
from itertools import combinations

# ---------- C1: effective dimensionality ----------
print('=== C1  effective dimensionality of the 8-category board ===')
d_all = load_panel(cats=CORE, firms_only=False)
rows = []
for date, snap in d_all.groupby('dt'):
    piv = snap.pivot_table(index='model_name', columns='category', values='rating')
    piv = piv.reindex(columns=CORE).dropna()
    if len(piv) < 30:
        continue
    R = two_way_resid(piv.values)
    C = np.corrcoef(R.T)
    if not np.all(np.isfinite(C)):
        continue
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    lam = np.clip(lam, 0, None)
    part = lam.sum() ** 2 / (lam ** 2).sum()      # participation ratio
    rows.append(dict(date=date, n_models=len(piv), eff_rank=part,
                     top1=lam[0] / lam.sum(), top2=lam[:2].sum() / lam.sum(),
                     top3=lam[:3].sum() / lam.sum()))
c1 = pd.DataFrame(rows)
print(f'snapshots={len(c1)}  median models/snapshot={c1.n_models.median():.0f}')
print(c1[['eff_rank', 'top1', 'top2', 'top3']].describe().loc[
    ['mean', '25%', '50%', '75%']].round(3).to_string())
print(f'\nnominal K = {len(CORE)}   effective K (participation ratio) = '
      f'{c1.eff_rank.mean():.2f}')
print(f'top 3 of 8 components carry {100*c1.top3.mean():.0f}% of residual variance')

# noise benchmark: pure-noise residuals of the same shape
rng = np.random.default_rng(0)
nb = []
for _ in range(200):
    Z = rng.standard_normal((int(c1.n_models.median()), len(CORE)))
    Rz = two_way_resid(Z)
    lz = np.clip(np.sort(np.linalg.eigvalsh(np.corrcoef(Rz.T)))[::-1], 0, None)
    nb.append(lz.sum() ** 2 / (lz ** 2).sum())
print(f'effective K under pure noise (null)      = {np.mean(nb):.2f} '
      f'(sd {np.std(nb):.2f})')
print('  -> observed effective K well BELOW the noise null means the categories'
      ' share a strong common factor and are not independent dimensions')

# ---------- C2: fine vs coarse partition ----------
print('\n=== C2  fine (industry) vs coarse (core) partition, 2025-11-05 onward ===')
d_ind = load_panel(cats=INDUSTRY + CORE)
d_ind = d_ind[d_ind.dt >= '2025-11-05']


def structure(d, cats):
    """Cross-firm dispersion (noise-corrected) and excess cosine over permutation null."""
    rng = np.random.default_rng(1)
    out = []
    for date in sorted(d.dt.unique()):
        P, V, names = firm_profiles(d, cats, date)
        if len(P) < 5:
            continue
        R = two_way_resid(P)
        N = len(P)
        nfac = (N - 1) / N
        disp = R.var(axis=0, ddof=1).mean() - np.nanmean(V) * nfac

        def mc(M):
            cs = [M[a] @ M[b] / (np.linalg.norm(M[a]) * np.linalg.norm(M[b]))
                  for a, b in combinations(range(len(M)), 2)
                  if np.linalg.norm(M[a]) * np.linalg.norm(M[b]) > 0]
            return np.mean(cs)
        obs = mc(R)
        null = [mc(np.column_stack([rng.permutation(R[:, k]) for k in range(R.shape[1])]))
                for _ in range(200)]
        # scale-free: dispersion relative to the sampling noise floor
        out.append(dict(date=date, N=N, disp=disp, noise=np.nanmean(V) * nfac,
                        snr=disp / (np.nanmean(V) * nfac),
                        excess=obs - np.mean(null), z=(obs - np.mean(null)) / np.std(null)))
    return pd.DataFrame(out)


for nm, cats in [('coarse (8 core)', CORE), ('fine (8 industry)', INDUSTRY)]:
    s = structure(d_ind, cats)
    if len(s) == 0:
        print(f'{nm}: no usable snapshots'); continue
    print(f'{nm:20s} snapshots={len(s):3d}  dispersion={s.disp.mean():7.1f}'
          f'  noise={s.noise.mean():6.1f}  signal/noise={s.snr.mean():5.2f}'
          f'  excess-cos={s.excess.mean():+.4f}  z={s.z.mean():+.2f}')
print('\nsignal/noise is the scale-free comparison: higher under the fine partition')
print('would mean firms differentiate INSIDE the coarse categories.')
