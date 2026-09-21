"""Plug the REAL fitted (firm_effect_i, cat_effect_k, u_i, v_k) into the dynamic
investment sim (rescaled from Arena's base-10/400 Elo units to the sim's natural
units), and check not just POOLED beta/persistence but PER-FIRM persistence --
against per-firm persistence computed the same way on the real panel.
"""
import numpy as np, pandas as pd
from exp_dynamic_investment import simulate, N, K, BURN_IN, T

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
ELO_TO_NAT = np.log(10) / 400  # rating points -> natural sigmoid units

# ---------- 1. real per-firm persistence (same two-way-demean-per-quarter recipe
#              as exp_persistence.py, but broken out BY FIRM instead of pooled) ----------
d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['org'] = d.organization.str.lower()
d = d[d.org.isin(FIRMS) & d.category.isin(CORE)]
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['quarter'] = d.dt.dt.to_period('Q')
g = d.groupby(['org', 'category', 'quarter'])['rating'].max().reset_index()

def demean(group):
    grand = group['rating'].mean()
    fm = group.groupby('org')['rating'].transform('mean')
    cm = group.groupby('category')['rating'].transform('mean')
    group['resid'] = group['rating'] - fm - cm + grand
    return group

g = g.groupby('quarter', group_keys=False).apply(demean, include_groups=False if False else True) if False else g
g = g.groupby('quarter', group_keys=False).apply(demean)
g = g.sort_values(['org', 'category', 'quarter'])
g['resid_lag1'] = g.groupby(['org', 'category'])['resid'].shift(1)
pairs = g.dropna(subset=['resid_lag1'])

print('=== REAL per-firm persistence (lag-1 autocorr of two-way-demeaned residual) ===')
real_pers = {}
for f in FIRMS:
    sub = pairs[pairs.org == f]
    if len(sub) >= 5:
        r = sub['resid'].corr(sub['resid_lag1'])
        real_pers[f] = r
        print(f'  {f:12s} n={len(sub):3d}  persistence={r:+.3f}')
    else:
        print(f'  {f:12s} n={len(sub):3d}  (too few obs, skipped)')

# ---------- 2. real fit -> sim inputs ----------
recent_dates = np.sort(d.dt.unique())[-4:]
dd = d[d.dt.isin(recent_dates)]
frontier = dd.groupby(['org', 'category'])['rating'].max().reset_index()
M_df = frontier.pivot(index='org', columns='category', values='rating').reindex(index=FIRMS, columns=CORE).dropna()
M = M_df.values * ELO_TO_NAT
grand = M.mean()
firm_effect = M.mean(axis=1) - grand
cat_effect = M.mean(axis=0) - grand
resid = M - grand - firm_effect[:, None] - cat_effect[None, :]
U, S, Vt = np.linalg.svd(resid)
u1, s1, v1 = U[:, 0] * np.sqrt(S[0]), S[0], Vt[0, :] * np.sqrt(S[0])
logD = np.outer(u1, v1)

A = np.exp(firm_effect)[:, None] * np.exp(cat_effect)[None, :] * np.exp(logD)
A = np.clip(A, 0.05, None)
print(f'\nrescaled A_i,k range: [{A.min():.3f}, {A.max():.3f}]  (calibrated toy run used ~[0.4, 1.7])')
w = np.ones(K) / K

# ---------- 3. run the sim with REAL-derived A, same shock params as the best-fit calibration ----------
theta_hist, X_hist = simulate(A, w, gamma=0.1, shock_sigma=0.75, rho_shock=0.0, seed=1)

def two_way_demean(Mt):
    return Mt - Mt.mean(axis=1, keepdims=True) - Mt.mean(axis=0, keepdims=True) + Mt.mean()

# pooled beta (same recipe as measure(), inlined so we can also split by firm)
gaps, dthetas = [], []
for t in range(BURN_IN, T - 1):
    th_prev, th_next = theta_hist[t], theta_hist[t + 1]
    gap_t = np.array([[np.max(np.delete(th_prev[:, k], i)) - th_prev[i, k] for k in range(K)] for i in range(N)])
    dtheta_t = th_next - th_prev
    gaps.append(two_way_demean(gap_t)); dthetas.append(two_way_demean(dtheta_t))
G = np.array(gaps); Y = np.array(dthetas)  # (T', N, K)
beta_pooled = np.sum(G * Y) / np.sum(G * G)

resid_sim = np.array([two_way_demean(theta_hist[t]) for t in range(BURN_IN, T)])
pers_pooled = np.corrcoef(resid_sim[:-1].ravel(), resid_sim[1:].ravel())[0, 1]
print(f'\nPOOLED sim (real-derived A): beta={beta_pooled:.3f}  persistence={pers_pooled:.3f}'
      f'   (targets: 0.43 / 0.339)')

print('\n=== per-firm persistence: REAL vs SIMULATED (using real-derived A_i,k) ===')
print(f'{"firm":12s} {"real":>8s} {"sim":>8s}')
sim_pers = {}
for i, f in enumerate(FIRMS):
    r0 = resid_sim[:-1, i, :].ravel()
    r1 = resid_sim[1:, i, :].ravel()
    sim_pers[f] = np.corrcoef(r0, r1)[0, 1]
    rv = f'{real_pers[f]:+.3f}' if f in real_pers else '   n/a'
    print(f'{f:12s} {rv:>8s} {sim_pers[f]:+8.3f}')

common = [f for f in FIRMS if f in real_pers]
if len(common) >= 3:
    rc = np.corrcoef([real_pers[f] for f in common], [sim_pers[f] for f in common])[0, 1]
    print(f'\ncross-firm correlation (real persistence vs sim persistence), n={len(common)} firms: r={rc:+.3f}')
