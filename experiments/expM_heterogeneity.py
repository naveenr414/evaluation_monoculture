"""Exp M: is firm-level heterogeneity in gap-chasing (beta_i) real, or sampling noise?

Same question and estimator as the first version of this script, rewritten to use
closed-form two-way demeaning on the balanced (event x category) matrix instead of
iterative pandas groupby demeaning -- the panel is balanced (every event has all 8
categories, by construction of build_events), so two-way FE has an exact one-shot
formula and bootstrap/permutation reps run in milliseconds instead of ~10ms each.

    tau^2 = Var(beta_i) across firms          [DerSimonian-Laird random-effects estimator]
    H0: beta_1 = ... = beta_N                 [Cochran's Q, permutation null]
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, CORE, FRONTIER
from scipy import stats

rng = np.random.default_rng(0)
d = load_panel(cats=CORE)
ev = build_events(d, CORE)
K = len(CORE)

G_all = ev.pivot_table(index='event', columns='cat', values='gap')[CORE]
D_all = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE]
Vo_all = ev.pivot_table(index='event', columns='cat', values='v_old')[CORE]
Vf_all = ev.pivot_table(index='event', columns='cat', values='v_front')[CORE]
firm_of = ev.drop_duplicates('event').set_index('event').firm.reindex(G_all.index)
assert G_all.index.equals(D_all.index) and G_all.index.equals(Vo_all.index)

print('events per firm:'); print(firm_of.value_counts().to_string())


def two_way_beta(G, D, Vo, Vf):
    """Exact two-way (event+category) FE slope on a balanced n_ev x K panel, plus
    the same diagonal mean-reversion noise correction used throughout."""
    n_ev = G.shape[0]
    if n_ev < 3:
        return np.nan
    Gw = G - G.mean(1, keepdims=True) - G.mean(0, keepdims=True) + G.mean()
    Dw = D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
    vg, cov = np.mean(Gw * Gw), np.mean(Gw * Dw)
    if vg <= 1e-10:
        return np.nan
    n = n_ev * K
    fac = (n - n_ev - K + 1) / n
    return (cov - Vo.mean() * fac) / (vg - (Vo.mean() + Vf.mean()) * fac)


def bootstrap_beta(G, D, Vo, Vf, n_boot=4000):
    n_ev = G.shape[0]
    if n_ev < 3:
        return np.array([])
    idx = rng.integers(0, n_ev, size=(n_boot, n_ev))
    out = np.array([two_way_beta(G[i], D[i], Vo[i], Vf[i]) for i in idx])
    return out[np.isfinite(out)]


firms = [f for f in FRONTIER if f in ev.firm.unique()]
boot = {}
point = {}
for f in firms:
    mask = (firm_of == f).values
    G, D, Vo, Vf = G_all.values[mask], D_all.values[mask], Vo_all.values[mask], Vf_all.values[mask]
    point[f] = two_way_beta(G, D, Vo, Vf)
    boot[f] = bootstrap_beta(G, D, Vo, Vf)

print('\n=== per-firm beta_i (bootstrap over events, n_boot=4000) ===')
print(f'{"firm":12s} {"n_ev":>5s} {"beta_i":>9s} {"boot_SE":>9s} {"95% CI":>20s}')
rows = []
for f in firms:
    b = boot[f]
    n_ev = int((firm_of == f).sum())
    if len(b) < 100:
        print(f'{f:12s} {n_ev:5d}   too few events for bootstrap')
        continue
    se = b.std(ddof=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f'{f:12s} {n_ev:5d} {point[f]:9.3f} {se:9.3f}   [{lo:6.2f}, {hi:6.2f}]')
    rows.append(dict(firm=f, n=n_ev, beta=point[f], se=se))
bt = pd.DataFrame(rows).dropna()


def q_stat(bt_):
    w = 1 / bt_.se.values ** 2
    beta_bar = np.sum(w * bt_.beta.values) / np.sum(w)
    return np.sum(w * (bt_.beta.values - beta_bar) ** 2), beta_bar, w


def tau2_dl(bt_):
    Q, beta_bar, w = q_stat(bt_)
    df = len(bt_) - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return max(0.0, (Q - df) / c), Q, df, beta_bar


tau2, Q, df, beta_bar = tau2_dl(bt)
print(f'\npooled beta (inverse-variance weighted) = {beta_bar:+.4f}')
print(f'Cochran Q = {Q:.2f} on df={df}  (chi2 asymptotic p={1 - stats.chi2.cdf(Q, df):.3f})')
print(f'DerSimonian-Laird tau^2 = {tau2:.4f}   (tau = {np.sqrt(tau2):.3f}, vs mean beta ~{beta_bar:.2f})')

# ---- permutation test: shuffle FIRM labels across events, keep event/category structure ----
print('\n=== permutation test (shuffle which firm each event belongs to) ===')
n_events = len(G_all)
firm_labels = firm_of.values
n_perm = 4000
null_Q = []
Gv, Dv, Vov, Vfv = G_all.values, D_all.values, Vo_all.values, Vf_all.values
se_proxy = bt.set_index('firm').se.to_dict()
for _ in range(n_perm):
    perm = rng.permutation(n_events)
    perm_labels = firm_labels[perm]
    bs, ss = [], []
    for f in firms:
        if f not in se_proxy:
            continue
        mask = perm_labels == f
        if mask.sum() < 3:
            continue
        b = two_way_beta(Gv[mask], Dv[mask], Vov[mask], Vfv[mask])
        if np.isfinite(b):
            bs.append(b); ss.append(se_proxy[f])
    if len(bs) < 3:
        continue
    bs, ss = np.array(bs), np.array(ss)
    w = 1 / ss ** 2
    bb = np.sum(w * bs) / np.sum(w)
    null_Q.append(np.sum(w * (bs - bb) ** 2))
null_Q = np.array(null_Q)
p_perm = np.mean(null_Q >= Q)
print(f'observed Q = {Q:.2f}; permutation null (n={len(null_Q)}): '
      f'mean={null_Q.mean():.2f}, 95th pct={np.percentile(null_Q,95):.2f}')
print(f'permutation p-value = {p_perm:.3f}')

# ---- robustness: leave-one-firm-out ----
print('\n=== leave-one-firm-out tau^2 ===')
for f in bt.firm:
    sub = bt[bt.firm != f]
    if len(sub) < 3:
        continue
    t2, Qs, dfs, bb = tau2_dl(sub)
    print(f'  drop {f:12s} tau^2={t2:.4f}  Q={Qs:.2f} df={dfs}  pooled_beta={bb:+.3f}')

# ---- robustness: drop firms with < 5 events ----
big = bt[bt.n >= 5]
print(f'\n=== firms with >=5 events only (n={len(big)}) ===')
if len(big) >= 3:
    t2, Qb, dfb, bb = tau2_dl(big)
    print(f'tau^2={t2:.4f}  Q={Qb:.2f} df={dfb}  pooled_beta={bb:+.3f}')
else:
    print('too few firms remain')

# ---- shrinkage (empirical Bayes) vs separate OLS ----
print('\n=== empirical Bayes shrinkage (using pooled tau^2) vs raw per-firm beta ===')
print(f'{"firm":12s} {"raw beta_i":>11s} {"shrunk beta_i":>14s} {"shrinkage wt":>13s}')
for _, r in bt.iterrows():
    w_shrink = tau2 / (tau2 + r.se ** 2) if tau2 > 0 else 0.0
    shrunk = w_shrink * r.beta + (1 - w_shrink) * beta_bar
    print(f'{r.firm:12s} {r.beta:11.3f} {shrunk:14.3f} {w_shrink:13.2f}')

print('\n=== interpretation ===')
print(f'DL tau^2 = {tau2:.4f} (clipped at 0 whenever Q < df -- biased toward 0 with only '
      f'{len(bt)} firms); permutation p = {p_perm:.3f}.')
if tau2 < 1e-3 and p_perm > 0.10:
    print('Both signals agree: cannot reject a common beta. Firm-level differences are '
          'consistent with sampling noise given how few release events most firms contribute.')
elif tau2 < 1e-3 and p_perm <= 0.10:
    print('Signals disagree: the DL point estimate clips to 0 (typical small-k downward bias), '
          'but the permutation test (which does not rely on that asymptotic correction) gives '
          f'p={p_perm:.3f}. Read this as weak/borderline evidence of heterogeneity, not a clean '
          'rejection -- and check whether it survives leave-one-out and the >=5-event subset above.')
else:
    print('tau^2 > 0 and the permutation test rejects a common beta: heterogeneous chasing '
          'strength across firms looks real, not just noise.')
