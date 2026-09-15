"""Formal test: do closed labs chase gaps harder than open-weight labs?

exp1_heterogeneous.py's post-fix descriptive split (0.61 closed vs 0.43 open, §3.1b) was
never tested. Here:

    dtheta = beta*gap + gamma*(gap x Closed) + event FE + category FE

gamma is the ADDITIONAL closed-lab response over open-weight. Inference via:
  - event-resampling bootstrap within each group (preserves the firm/event block)
  - EXACT permutation test over all C(8,4)=70 ways to split the 8 frontier firms into
    two groups of 4 (relabeling which firms are "closed"), which is cheap enough to
    enumerate exactly rather than sample -- respects firm-level event-count imbalance
    properly, unlike shuffling individual events.
"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, CORE, FRONTIER

rng = np.random.default_rng(0)
CLOSED = {'openai', 'anthropic', 'google', 'xai'}
OPEN = {'meta', 'mistral', 'deepseek', 'alibaba'}
K = len(CORE)

d = load_panel(cats=CORE)
ev = build_events(d, CORE)

G_all = ev.pivot_table(index='event', columns='cat', values='gap')[CORE]
D_all = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE]
Vo_all = ev.pivot_table(index='event', columns='cat', values='v_old')[CORE]
Vf_all = ev.pivot_table(index='event', columns='cat', values='v_front')[CORE]
firm_of = ev.drop_duplicates('event').set_index('event').firm.reindex(G_all.index)
Gv, Dv, Vov, Vfv = G_all.values, D_all.values, Vo_all.values, Vf_all.values
n_events = len(G_all)
print(f'events: {n_events}')


def two_way_beta(G, D, Vo, Vf):
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


def gamma_for(closed_set, labels=firm_of.values, G=Gv, D=Dv, Vo=Vov, Vf=Vfv):
    m_closed = np.isin(labels, list(closed_set))
    m_open = ~m_closed
    if m_closed.sum() < 3 or m_open.sum() < 3:
        return np.nan, np.nan, np.nan
    b_closed = two_way_beta(G[m_closed], D[m_closed], Vo[m_closed], Vf[m_closed])
    b_open = two_way_beta(G[m_open], D[m_open], Vo[m_open], Vf[m_open])
    return b_closed - b_open, b_closed, b_open


gamma_obs, b_closed_obs, b_open_obs = gamma_for(CLOSED)
print(f'\nbeta closed = {b_closed_obs:+.4f}   beta open = {b_open_obs:+.4f}')
print(f'gamma (closed - open) = {gamma_obs:+.4f}')

# ---- event-resampling bootstrap within each group ----
n_boot = 4000
boot_gamma = []
idx_closed = np.where(np.isin(firm_of.values, list(CLOSED)))[0]
idx_open = np.where(np.isin(firm_of.values, list(OPEN)))[0]
for _ in range(n_boot):
    ic = rng.choice(idx_closed, size=len(idx_closed), replace=True)
    io = rng.choice(idx_open, size=len(idx_open), replace=True)
    bc = two_way_beta(Gv[ic], Dv[ic], Vov[ic], Vfv[ic])
    bo = two_way_beta(Gv[io], Dv[io], Vov[io], Vfv[io])
    if np.isfinite(bc) and np.isfinite(bo):
        boot_gamma.append(bc - bo)
boot_gamma = np.array(boot_gamma)
se_boot = boot_gamma.std(ddof=1)
lo, hi = np.percentile(boot_gamma, [2.5, 97.5])
print(f'\nbootstrap SE(gamma) = {se_boot:.4f}   95% CI = [{lo:+.3f}, {hi:+.3f}]   '
      f'(n={len(boot_gamma)} of {n_boot} finite)')
print(f't-like stat = {gamma_obs/se_boot:+.2f}')

# ---- exact permutation test over all 4-vs-4 firm partitions ----
print('\n=== exact permutation test (all C(8,4)/2=35 distinct 4-vs-4 firm splits) ===')
all_firms = list(FRONTIER)
null_gamma = []
seen = set()
for combo in itertools.combinations(all_firms, 4):
    key = frozenset(combo)
    comp = frozenset(all_firms) - key
    if comp in seen:   # each split counted once (closed=A,open=B is the same split as closed=B,open=A up to sign)
        continue
    seen.add(key)
    g, _, _ = gamma_for(key)
    if np.isfinite(g):
        null_gamma.append(abs(g))
null_gamma = np.array(null_gamma)
p_exact = np.mean(null_gamma >= abs(gamma_obs))
print(f'observed |gamma| = {abs(gamma_obs):.4f}')
print(f'null distribution of |gamma| over {len(null_gamma)} distinct splits: '
      f'mean={null_gamma.mean():.3f}, 95th pct={np.percentile(null_gamma,95):.3f}, max={null_gamma.max():.3f}')
print(f'exact permutation p-value (two-sided, |gamma| >= observed) = {p_exact:.3f}')
print(f'rank of observed split among all {len(null_gamma)}: '
      f'{(null_gamma >= abs(gamma_obs)).sum()} splits are as or more extreme')

print('\n=== interpretation ===')
if p_exact < 0.05:
    print(f'REJECT common beta across open/closed: gamma={gamma_obs:+.3f} is significant '
          f'(exact p={p_exact:.3f}). The open/closed split captures real heterogeneity, not '
          f'just an arbitrary partition of the 8 firms.')
else:
    print(f'CANNOT reject a common beta across open/closed (exact p={p_exact:.3f}). The observed '
          f'0.61-vs-0.43 split is not distinguishable from an arbitrary 4-vs-4 partition of the '
          f'8 firms -- drop the open/closed framing rather than treat it as a finding.')

print('\n=== context: is the observed split special among ALL 35 partitions, or generic? ===')
# where does the ACTUAL open/closed split rank among all 35 by |gamma|?
all_splits = []
for combo in itertools.combinations(all_firms, 4):
    key = frozenset(combo)
    comp = frozenset(all_firms) - key
    if comp in [s for s, _ in all_splits]:
        continue
    g, bc, bo = gamma_for(key)
    if np.isfinite(g):
        all_splits.append((key, g))
all_splits.sort(key=lambda x: -abs(x[1]))
rank = [i for i, (k, g) in enumerate(all_splits) if k == frozenset(CLOSED) or k == frozenset(OPEN)]
print(f'the actual open/closed split ranks #{rank[0]+1 if rank else "?"} of {len(all_splits)} '
      f'partitions by |gamma| (rank 1 = most extreme)')
print('\ntop 5 most extreme splits (by |gamma|), for context:')
for key, g in all_splits[:5]:
    tag = ' <-- ACTUAL OPEN/CLOSED SPLIT' if key == frozenset(CLOSED) or key == frozenset(OPEN) else ''
    print(f'  {sorted(key)!s:55s} gamma={g:+.3f}{tag}')
