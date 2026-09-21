"""Exp K: multivariate gap-response matrix B.

dtheta_e (K-vector) = B @ gap_e (K-vector) + eps_e, one multivariate obs per event.
Category-level intercepts (alpha_k) absorb average improvement per category, so all
matrices are column (category)-demeaned first -- equivalent to a category FE.

Specifications, ridge-regularized where needed, compared by leave-one-event-out CV R^2:
  1. B = beta*I                          (scalar partial adjustment, 1 param)
  2. B = beta*I + gamma*(11^T - I)       (common cross-category spillover, 2 params)
  3. low-rank B, rank 1/2/3              (reduced-rank regression via ridge + SVD)
  4. unrestricted ridge B                (upper bound on achievable fit)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, CORE, FRONTIER

d = load_panel(cats=CORE)
ev = build_events(d, CORE)
K = len(CORE)

G = ev.pivot_table(index='event', columns='cat', values='gap')[CORE]
D = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE]
firms = ev.drop_duplicates('event').set_index('event').firm.reindex(G.index)
assert G.index.equals(D.index)
n = len(G)
print(f'events (complete 8-cat vectors) = {n}')

Gc = G.values - G.values.mean(0, keepdims=True)
Dc = D.values - D.values.mean(0, keepdims=True)


def r2_oos(pred_fn):
    """Leave-one-event-out CV R^2, pred_fn(Gtrain, Dtrain) -> B (KxK)."""
    sse, sst = 0.0, 0.0
    dbar = Dc.mean(0)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        B = pred_fn(Gc[mask], Dc[mask])
        pred = Gc[i] @ B.T
        sse += np.sum((Dc[i] - pred) ** 2)
        sst += np.sum((Dc[i] - dbar) ** 2)
    return 1 - sse / sst


# ---- 1. scalar B = beta I ----
def fit_scalar(Gt, Dt):
    beta = np.sum(Gt * Dt) / np.sum(Gt * Gt)
    return beta * np.eye(K)

beta_hat = np.sum(Gc * Dc) / np.sum(Gc * Gc)
r2_scalar = r2_oos(fit_scalar)
print(f'\n1. scalar B=beta*I         beta={beta_hat:+.4f}   CV R^2={r2_scalar:+.4f}')

# mean-reversion correction (as in exp1): gap_k and dtheta_k both carry -eps_old,k,
# which is category-specific sampling noise -- biases only DIAGONAL entries of B,
# since sampling error is independent across categories. Off-diagonal cross-effects
# are not subject to this bias.
v_old_m, v_front_m = ev.v_old.mean(), ev.v_front.mean()
dfac = (n - 1) / n  # category-only demeaning shrinkage factor
cov_d, vg_d = np.sum(Gc * Dc) / n, np.sum(Gc * Gc) / n
beta_corr = (cov_d - v_old_m * dfac) / (vg_d - (v_old_m + v_front_m) * dfac)
print(f'   noise-corrected diagonal beta = {beta_corr:+.4f}  (off-diagonals unaffected by this bias)')


# ---- 2. common cross-category: B = beta*I + gamma*(11^T - I) ----
def fit_common(Gt, Dt):
    own = np.sum(Gt * Dt)
    voff = 0.0; coff = 0.0
    for k in range(K):
        other = Gt.sum(1) - Gt[:, k]
        coff += np.sum(other * Dt[:, k])
        voff += np.sum(other * other)
    vown = np.sum(Gt * Gt)
    # solve 2x2: [vown, cross; cross, voff] [beta; gamma] = [own; coff]
    cross = sum(np.sum(Gt[:, k] * (Gt.sum(1) - Gt[:, k])) for k in range(K))
    A = np.array([[vown, cross], [cross, voff]])
    b = np.array([own, coff])
    beta, gamma = np.linalg.solve(A, b)
    return beta * np.eye(K) + gamma * (np.ones((K, K)) - np.eye(K))

r2_common = r2_oos(fit_common)
Bc_full = fit_common(Gc, Dc)
beta2 = Bc_full[0, 0]; gamma2 = Bc_full[0, 1]
print(f'2. common B=bI+g(11^T-I)   beta={beta2:+.4f}  gamma={gamma2:+.4f}   CV R^2={r2_common:+.4f}')


# ---- ridge regression B (full), used as base for low-rank truncation ----
def fit_ridge(Gt, Dt, lam):
    A = Gt.T @ Gt + lam * np.eye(K)
    Bt = np.linalg.solve(A, Gt.T @ Dt).T   # B: Dc = Gc @ B^T
    return Bt

# choose ridge lambda by CV
lams = [1, 3, 10, 30, 100, 300, 1000]
best_lam, best_r2 = None, -np.inf
for lam in lams:
    r2 = r2_oos(lambda Gt, Dt, lam=lam: fit_ridge(Gt, Dt, lam))
    if r2 > best_r2:
        best_r2, best_lam = r2, lam
print(f'\n4. unrestricted ridge B    best lambda={best_lam}   CV R^2={best_r2:+.4f}')
B_ridge = fit_ridge(Gc, Dc, best_lam)

# ---- 3. low-rank truncations of the ridge B, rank 1/2/3 ----
def fit_lowrank(Gt, Dt, lam, rank):
    Bt = fit_ridge(Gt, Dt, lam)
    U, S, Vt = np.linalg.svd(Bt)
    return U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

print()
for rank in [1, 2, 3]:
    r2 = r2_oos(lambda Gt, Dt, r=rank: fit_lowrank(Gt, Dt, best_lam, r))
    print(f'3. rank-{rank} B (ridge+SVD truncation)      CV R^2={r2:+.4f}')

# ---- diagnostics on the full-sample ridge B ----
print('\n--- diagnostics (full-sample ridge B, lambda=%d) ---' % best_lam)
diag = np.diag(B_ridge)
offdiag = B_ridge - np.diag(diag)
print(f'mean diagonal (own-gap response)     = {diag.mean():+.3f}  (range {diag.min():+.3f} to {diag.max():+.3f})')
print(f'mean |off-diagonal|                  = {np.abs(offdiag).mean():.3f}')
print(f'ratio |offdiag|/|diag| (Frobenius)    = {np.linalg.norm(offdiag)/np.linalg.norm(np.diag(diag)):.3f}')

U, S, Vt = np.linalg.svd(B_ridge)
print(f'singular values of B                 = {np.array2string(S, precision=2)}')
print(f'share of Frobenius norm^2 in top 1/2/3 = '
      f'{np.sum(S[:1]**2)/np.sum(S**2):.3f} / {np.sum(S[:2]**2)/np.sum(S**2):.3f} / {np.sum(S[:3]**2)/np.sum(S**2):.3f}')

print('\nB (rows=response category, cols=gap category), ridge-regularized:')
print(pd.DataFrame(B_ridge, index=CORE, columns=CORE).round(2).to_string())

print('\n=== summary ===')
specs = [('scalar (1 param)', r2_scalar), ('common (2 params)', r2_common),
         ('rank-1', r2_oos(lambda Gt, Dt: fit_lowrank(Gt, Dt, best_lam, 1))),
         ('rank-2', r2_oos(lambda Gt, Dt: fit_lowrank(Gt, Dt, best_lam, 2))),
         ('rank-3', r2_oos(lambda Gt, Dt: fit_lowrank(Gt, Dt, best_lam, 3))),
         ('unrestricted ridge (64 params)', best_r2)]
for name, r2 in specs:
    print(f'  {name:32s} CV R^2 = {r2:+.4f}')
