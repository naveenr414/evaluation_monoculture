"""Experiment A: does the calibrated model account for the OBSERVED LEVEL of diversity?

Partial adjustment toward the per-category frontier, plus idiosyncratic innovation:
    r'_{i,k} = (1-beta) r_{i,k} + beta * (frontier_k) + eps_{i,k}
Frontier_k is common across firms within a category, so it is absorbed by the
category demeaning; cross-firm dispersion therefore evolves as
    D' = (1-beta)^2 D + sigma^2_innov
    =>  D* = sigma^2_innov / (1 - (1-beta)^2)

We estimate beta and sigma^2_innov from release events, then compare the implied
D* to the dispersion actually observed in the panel. All quantities noise-corrected.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import (load_panel, build_events, fe_regress, two_way_resid,
                    firm_profiles, CORE, FRONTIER)

d = load_panel(cats=CORE)
ev = build_events(d, CORE)
print(f'events={ev.event.nunique()}  firms={ev.firm.nunique()}  obs={len(ev)}\n')

# ---- beta and innovation variance from the release panel ----
X = ev[['gap']].values
groups = [ev.event.values, ev.cat.values]
# gap carries -eps_old and dtheta carries -eps_old  -> spurious +var(eps_old)
ncov = np.array([ev.v_old.mean()])
nvar = np.array([[ev.v_old.mean() + ev.v_front.mean()]])
beta, se, beta_c, resid, dfac = fe_regress(
    ev.dtheta.values, X, groups, ncov, nvar, cluster=ev.event.values)

b_naive, b_corr = beta[0], beta_c[0]
print(f'beta  naive = {b_naive:+.4f}  (cluster SE {se[0]:.4f}, t={b_naive/se[0]:+.2f})')
print(f'beta  noise-corrected = {b_corr:+.4f}')

# innovation variance = residual variance minus the sampling noise it inherits.
# dtheta = theta_new - theta_old, so residual carries eps_new and (1-beta)*eps_old
resid_var = resid.var(ddof=1)
noise_in_resid = (ev.v_new.mean() + (1 - b_corr) ** 2 * ev.v_old.mean()) * dfac
sig2_innov = resid_var - noise_in_resid
print(f'\nresidual var (per release, per category) = {resid_var:8.1f}')
print(f'  sampling noise inside it               = {noise_in_resid:8.1f}')
print(f'  => innovation variance sigma^2_innov   = {sig2_innov:8.1f}')

# ---- predicted steady-state dispersion ----
D_pred = sig2_innov / (1 - (1 - b_corr) ** 2)
print(f'\npredicted steady-state dispersion D* = {D_pred:.1f}'
      f'   [= {sig2_innov:.1f} / (1 - (1-{b_corr:.3f})^2)]')

# ---- observed dispersion, noise-corrected ----
obs = []
for date, snap in d.groupby('dt'):
    P, V, names = firm_profiles(d, CORE, date)
    if len(P) < 5:
        continue
    R = two_way_resid(P)
    nfac = (len(P) - 1) / len(P)          # column demeaning shrinks noise
    obs.append(dict(date=date, n=len(P),
                    raw=R.var(axis=0, ddof=1).mean(),
                    noise=np.nanmean(V) * nfac))
o = pd.DataFrame(obs)
o['corrected'] = o.raw - o.noise
D_obs = o.corrected.mean()
print(f'observed dispersion (noise-corrected, mean over {len(o)} snapshots) = {D_obs:.1f}'
      f'   [raw {o.raw.mean():.1f} - noise {o.noise.mean():.1f}]')
print(f'   snapshot range {o.corrected.min():.0f} to {o.corrected.max():.0f}')

ratio = D_pred / D_obs
print(f'\n>>> predicted / observed = {ratio:.2f}')
if 0.5 < ratio < 2.0:
    verdict = 'MATCH: chasing-vs-innovation accounts for the observed level of diversity'
elif ratio < 0.5:
    verdict = 'UNDER-PREDICTS: something else sustains diversity (different F, or K too coarse)'
else:
    verdict = 'OVER-PREDICTS: an additional homogenising force is missing from the model'
print('>>> ' + verdict)

# ---- half-life and implied dynamics ----
hl = np.log(0.5) / np.log(1 - b_corr)
print(f'\nimplied half-life of a profile deviation = {hl:.2f} releases')
sw = ev.groupby('event').agg(firm=('firm', 'first'), switch=('switch', 'first'))
med_gap = sw.sort_values('switch').groupby('firm').switch.diff().dt.days.median()
print(f'median spacing between one firm\'s releases = {med_gap:.0f} days'
      f'  -> ~{hl*med_gap/30.4:.1f} months')

# ---- direct test of implied persistence ----
# The partial-adjustment model says a firm's residual profile should decay at rate
# (1-beta) from one release to the next. Measure that decay directly: if deviations
# die FASTER than (1-beta), an extra homogenising force is confirmed independently.
print('\n--- direct persistence test ---')
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
prof = []
for e, g in ev.groupby('event'):
    firm, sw_ = g.firm.iloc[0], g.switch.iloc[0]
    P, V, names = firm_profiles(d, CORE, sw_)
    if len(P) < 5 or firm not in names:
        continue
    R = two_way_resid(P)
    prof.append(dict(firm=firm, switch=sw_, r=R[names.index(firm)],
                     v=V[names.index(firm)]))
pf = pd.DataFrame(prof).sort_values(['firm', 'switch'])

pairs = []
for firm, g in pf.groupby('firm'):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        pairs.append((firm, g.r[i], g.r[i + 1], g.v[i], g.v[i + 1]))
if pairs:
    r0 = np.concatenate([p[1] for p in pairs])
    r1 = np.concatenate([p[2] for p in pairs])
    v0 = np.concatenate([p[3] for p in pairs])
    fm = np.repeat([p[0] for p in pairs], len(CORE))
    cm = np.tile(CORE, len(pairs))
    bb, ss, bc, _, dfac2 = fe_regress(r1, r0[:, None], [fm, cm],
                                      np.array([0.0]),
                                      np.array([[np.nanmean(v0)]]),
                                      cluster=fm)
    print(f'  n transitions = {len(pairs)}, obs = {len(r0)}')
    print(f'  measured persistence          = {bb[0]:+.3f} (SE {ss[0]:.3f})')
    print(f'  noise-corrected persistence   = {bc[0]:+.3f}')
    print(f'  predicted by gap-chasing alone= {1-b_corr:+.3f}')
    print(f'  => deviations decay {"FASTER" if bc[0] < 1-b_corr else "SLOWER"}'
          f' than eval-chasing alone explains')

# ---- sensitivity of the verdict to beta ----
print('\nsensitivity:')
for b in [0.30, 0.35, 0.43, 0.50, 0.55]:
    print(f'  beta={b:.2f} -> D*={sig2_innov/(1-(1-b)**2):7.1f}  ratio={sig2_innov/(1-(1-b)**2)/D_obs:5.2f}')
