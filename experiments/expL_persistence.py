"""Exp L: does the innovation residual persist across releases, or is it transitory?

Uses the preferred gap-response model from expK (B approx scalar/low-rank; richer
specs buy only marginal CV R^2 over B=beta*I), so residuals are built from the
scalar, noise-corrected beta already established in exp1/expA.

    u_{i,e,k} = dtheta_{i,e,k} - alpha_k - beta * gap_{i,e,k}

alpha_k is a category intercept (average improvement in category k across all
firms/events); beta uses exp1's event+category-FE, noise-corrected estimate.
u retains the firm's event-level average momentum -- that is exactly the object
whose persistence we are testing, so it should NOT be event-demeaned away.

Track (firm, category) into the firm's NEXT release and regress:
    u_{i,e+1,k} = rho * u_{i,e,k} + eps
plus symmetry (sign split), time-gap interaction, and category/firm splits.
Then feed the measured rho back into the steady-state accounting from expA.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, fe_regress, two_way_resid, firm_profiles, CORE, FRONTIER

d = load_panel(cats=CORE)
ev = build_events(d, CORE)
K = len(CORE)

# ---- beta (scalar, noise-corrected, event+category FE), same spec as expA ----
X = ev[['gap']].values
groups = [ev.event.values, ev.cat.values]
ncov = np.array([ev.v_old.mean()])
nvar = np.array([[ev.v_old.mean() + ev.v_front.mean()]])
beta, se, beta_c, resid, dfac = fe_regress(ev.dtheta.values, X, groups, ncov, nvar, cluster=ev.event.values)
beta_corr = beta_c[0]
print(f'beta (noise-corrected, event+category FE, same spec as expA) = {beta_corr:+.4f}')

# u = the event+category-demeaned residual -- this IS expA's innovation term (the
# object that drives cross-CATEGORY dispersion D, after each release's own average
# improvement is netted out by the event FE). `resid` is row-aligned with `ev`.
ev = ev.copy()
ev['u'] = resid

resid_var = ev.u.var(ddof=1)
noise_in_resid = (ev.v_new.mean() + (1 - beta_corr) ** 2 * ev.v_old.mean()) * dfac
sig2_innov_total = resid_var - noise_in_resid
print(f'residual var = {resid_var:.1f}, sampling noise in it = {noise_in_resid:.1f} '
      f'=> sigma^2_innov = {sig2_innov_total:.1f}  (cf. expA: 232.9)\n')

# ---- pair each event with the SAME firm's next event ----
sw = ev.drop_duplicates('event').set_index('event')[['firm', 'switch']]
pairs = []
for firm, g in sw.sort_values('switch').groupby('firm'):
    events_sorted = g.index.tolist()
    for e0, e1 in zip(events_sorted[:-1], events_sorted[1:]):
        gap_days = (g.loc[e1, 'switch'] - g.loc[e0, 'switch']).days
        pairs.append((firm, e0, e1, gap_days))
pairs = pd.DataFrame(pairs, columns=['firm', 'e0', 'e1', 'gap_days'])
print(f'firm-level consecutive release pairs = {len(pairs)}  (of {sw.firm.value_counts().sub(1).clip(lower=0).sum()} possible)')

u0 = ev.set_index(['event', 'cat']).u
rows = []
for _, p in pairs.iterrows():
    for k in CORE:
        try:
            rows.append(dict(firm=p.firm, cat=k, gap_days=p.gap_days,
                              u0=u0[(p.e0, k)], u1=u0[(p.e1, k)]))
        except KeyError:
            continue
pr = pd.DataFrame(rows)
print(f'(firm,category) transition observations = {len(pr)}\n')

v_u = noise_in_resid  # sampling noise carried by u (from the dfac-scaled BT variance above)

def ar1(sub, label):
    if len(sub) < 5:
        print(f'  {label:28s} n={len(sub):4d}  (too few obs)')
        return np.nan
    U0 = np.asarray(sub.u0.values, dtype=float); U1 = np.asarray(sub.u1.values, dtype=float)
    cov, var0 = np.mean(U0 * U1), np.mean(U0 * U0)
    rho_naive = cov / var0
    # errors-in-variables: u0 measured with noise v_u -> attenuates rho; correct diagonal only
    # (u1's noise is uncorrelated with u0's noise across releases)
    rho_corr = cov / (var0 - v_u) if var0 > v_u else np.nan
    print(f'  {label:28s} n={len(sub):4d}  naive rho={rho_naive:+.3f}  noise-corrected rho={rho_corr:+.3f}')
    return rho_corr

print('=== overall persistence ===')
rho_all = ar1(pr, 'all (firm,cat) transitions')

print('\n=== symmetry: split by sign of u0 ===')
ar1(pr[pr.u0 > 0], 'starting positive (ahead)')
ar1(pr[pr.u0 < 0], 'starting negative (behind)')

print('\n=== does persistence depend on release spacing? ===')
med_gap = pr.gap_days.median()
ar1(pr[pr.gap_days <= med_gap], f'short spacing (<= {med_gap:.0f}d)')
ar1(pr[pr.gap_days > med_gap], f'long spacing (> {med_gap:.0f}d)')

print('\n=== by category (pooled across firms) ===')
for cat, sub in pr.groupby('cat'):
    ar1(sub, cat)

print('\n=== by firm (pooled across categories; small n) ===')
for firm, sub in pr.groupby('firm'):
    ar1(sub, firm)

# ---- feed rho back into the steady-state accounting ----
print('\n=== steady-state accounting with measured persistence ===')
rho = max(-0.95, min(rho_all, 0.95)) if np.isfinite(rho_all) else 0.0
# innovation process: u_{t+1} = rho*u_t + e_t. sigma_e^2 = var of the direct residual.
e_resid_var = np.var(pr.u1.values - rho * pr.u0.values, ddof=1) if np.isfinite(rho_all) else sig2_innov_total
print(f'using rho = {rho:+.3f}, innovation shock variance sigma_e^2 = {e_resid_var:.1f}')

# simulate r_{t+1} = (1-beta) r_t + u_{t+1};  u_{t+1} = rho u_t + e_t
rng = np.random.default_rng(0)
T = 200_000
r = 0.0
u = 0.0
burn = 2000
vals = []
for t in range(T):
    e = rng.normal(0, np.sqrt(max(e_resid_var, 1e-6)))
    u = rho * u + e
    r = (1 - beta_corr) * r + u
    if t > burn:
        vals.append(r)
D_pred_persist = np.var(vals)
D_pred_iid = sig2_innov_total / (1 - (1 - beta_corr) ** 2)
print(f'D* with rho=0 (iid innovation, expA-style) = {D_pred_iid:.1f}')
print(f'D* with measured persistence rho           = {D_pred_persist:.1f}')

# observed dispersion (same as expA)
obs = []
for date, snap in d.groupby('dt'):
    P, V, names = firm_profiles(d, CORE, date)
    if len(P) < 5:
        continue
    R = two_way_resid(P)
    nfac = (len(P) - 1) / len(P)
    obs.append(dict(raw=R.var(axis=0, ddof=1).mean(), noise=np.nanmean(V) * nfac))
o = pd.DataFrame(obs)
D_obs = (o.raw - o.noise).mean()
print(f'observed dispersion (noise-corrected)       = {D_obs:.1f}')
print(f'\nratio iid-model / observed      = {D_pred_iid/D_obs:.2f}')
print(f'ratio persistence-model / observed = {D_pred_persist/D_obs:.2f}')
if abs(D_pred_persist/D_obs - 1) < abs(D_pred_iid/D_obs - 1):
    print('=> incorporating measured persistence moves the prediction CLOSER to observed.')
else:
    print('=> incorporating measured persistence does NOT close the gap; look elsewhere.')
