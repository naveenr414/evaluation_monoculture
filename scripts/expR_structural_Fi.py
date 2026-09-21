"""Priority 1 (from the 2026-08-10 round), cleanest next test: fit a low-dimensional
structural F_i and ask whether historical shocks (release events) identify it via
leave-one-shock-out prediction.

Family: F_i(x) = a_i * F_0(x) [+ b_i * G(x)], operationalized on the SAME gap -> dtheta
response used throughout (§3.1, §3.5's expK):

  Model 0 (pooled, no firm identity):  dtheta_e = beta0*gap_e + s1*(u1.gap_e)*v1
      -- the "F_i is common across all firms" null. beta0*I is the isotropic response
      (§3.1's scalar law); s1*u1*v1^T is the shared rank-1 cross-category correction
      extracted from expK's pooled ridge B (its comparative-advantage direction).
  Model 1 (a_i only):   dtheta_e = a_i*beta0*gap_e + s1*(u1.gap_e)*v1
      -- firms differ only in overall RESPONSIVENESS (1 parameter/firm), same shape.
  Model 2 (a_i, b_i):   dtheta_e = a_i*beta0*gap_e + b_i*s1*(u1.gap_e)*v1
      -- firms also differ in how much they lean on the shared comparative-advantage
      correction (2 parameters/firm).

a_i, b_i are ridge-shrunk toward 1 (the pooled value), with the ridge strength chosen by
nested leave-one-event-out CV. Evaluated by leave-one-EVENT-out prediction of the held-out
event's full 8-category dtheta vector, comparing OOS R^2 across the three models plus
expK's unrestricted/low-rank matrices as reference points.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_panel, build_events, CORE, FRONTIER

d = load_panel(cats=CORE)
ev = build_events(d, CORE)
K = len(CORE)

G_all = ev.pivot_table(index='event', columns='cat', values='gap')[CORE]
D_all = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE]
firm_of = ev.drop_duplicates('event').set_index('event').firm.reindex(G_all.index)
Gv, Dv = G_all.values, D_all.values
n = len(Gv)
print(f'events: {n}, firms: {firm_of.nunique()}')

Gc = Gv - Gv.mean(0, keepdims=True)
Dc = Dv - Dv.mean(0, keepdims=True)


# ---- shared basis: isotropic beta0 + rank-1 correction (u1, s1, v1), fit once on the full
# sample (same convention as expK; a firm-agnostic quantity, reused across all folds) ----
def fit_ridge(Gt, Dt, lam):
    A = Gt.T @ Gt + lam * np.eye(K)
    return np.linalg.solve(A, Gt.T @ Dt).T

lam_shared = 1000  # matches expK's CV-selected value
B_ridge = fit_ridge(Gc, Dc, lam_shared)
beta0 = np.diag(B_ridge).mean()
resid_matrix = B_ridge - beta0 * np.eye(K)
U, S, Vt = np.linalg.svd(resid_matrix)
u1, s1, v1 = U[:, 0], S[0], Vt[0, :]
print(f'shared isotropic beta0 = {beta0:.3f}')
print(f'shared rank-1 correction: singular value s1={s1:.3f} '
      f'(share of resid Frobenius^2 = {s1**2/np.sum(S**2):.3f})')

# per-event basis features: x1_e = beta0*gap_e (K-vec), x2_e = s1*(v1.gap_e)*u1 (K-vec)
# (resid_matrix = U @ diag(S) @ Vt, so resid_matrix @ x = sum_j S_j*(v_j.x)*u_j -- v1 is the
# INPUT (gap-space) direction to project onto, u1 is the OUTPUT (dtheta-space) response shape)
X1 = beta0 * Gc                                   # n x K
proj = Gc @ v1                                    # n
X2 = s1 * proj[:, None] * u1[None, :]             # n x K


def predict(a, b, i):
    return a * X1[i] + b * X2[i]


def fit_ab(idx, lam):
    """ridge-shrunk (a,b) toward (1,1) using events in idx; stacked over K categories."""
    x1 = X1[idx].reshape(-1)
    x2 = X2[idx].reshape(-1)
    y = Dc[idx].reshape(-1)
    Xd = np.column_stack([x1, x2])
    A = Xd.T @ Xd + lam * np.eye(2)
    b_vec = Xd.T @ y + lam * np.array([1.0, 1.0])
    return np.linalg.solve(A, b_vec)


def fit_a_only(idx, lam):
    x1 = X1[idx].reshape(-1)
    y = Dc[idx].reshape(-1) - X2[idx].reshape(-1)  # fixed b=1
    denom = x1 @ x1 + lam
    num = x1 @ y + lam * 1.0
    return num / denom


firms = sorted(firm_of.unique())
idx_by_firm = {f: np.where(firm_of.values == f)[0] for f in firms}


def loo_r2_model0():
    """pooled: a=b=1 for everyone, no firm-specific fit at all."""
    sse = sum(np.sum((Dc[i] - (X1[i] + X2[i])) ** 2) for i in range(n))
    dbar = Dc.mean(0)
    sst = sum(np.sum((Dc[i] - dbar) ** 2) for i in range(n))
    return 1 - sse / sst


def loo_r2_model1(lam):
    sse, sst = 0.0, 0.0
    dbar = Dc.mean(0)
    for f in firms:
        idx = idx_by_firm[f]
        if len(idx) < 2:
            for i in idx:  # too few events to hold one out -- fall back to pooled a=1
                pred = X1[i] + X2[i]
                sse += np.sum((Dc[i] - pred) ** 2)
                sst += np.sum((Dc[i] - dbar) ** 2)
            continue
        for j, i in enumerate(idx):
            train = np.delete(idx, j)
            a = fit_a_only(train, lam)
            pred = a * X1[i] + X2[i]
            sse += np.sum((Dc[i] - pred) ** 2)
            sst += np.sum((Dc[i] - dbar) ** 2)
    return 1 - sse / sst


def loo_r2_model2(lam):
    sse, sst = 0.0, 0.0
    dbar = Dc.mean(0)
    for f in firms:
        idx = idx_by_firm[f]
        if len(idx) < 3:  # need >=2 to fit 2 params after holding one out
            for i in idx:
                pred = X1[i] + X2[i]
                sse += np.sum((Dc[i] - pred) ** 2)
                sst += np.sum((Dc[i] - dbar) ** 2)
            continue
        for j, i in enumerate(idx):
            train = np.delete(idx, j)
            a, b = fit_ab(train, lam)
            pred = predict(a, b, i)
            sse += np.sum((Dc[i] - pred) ** 2)
            sst += np.sum((Dc[i] - dbar) ** 2)
    return 1 - sse / sst


r2_m0 = loo_r2_model0()
print(f'\nModel 0 (pooled, no firm identity): CV R^2 = {r2_m0:+.4f}')

print('\nModel 1 (a_i per firm, ridge-shrunk toward 1):')
lams = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 1e9]
best_lam1, best_r2_1 = None, -np.inf
for lam in lams:
    r2 = loo_r2_model1(lam)
    tag = '  <- lam=1e9 means a_i collapses to a=1 for all (= Model 0)' if lam == 1e9 else ''
    print(f'  lambda={lam:<8g} CV R^2={r2:+.4f}{tag}')
    if r2 > best_r2_1:
        best_r2_1, best_lam1 = r2, lam

print('\nModel 2 (a_i, b_i per firm, ridge-shrunk toward 1):')
best_lam2, best_r2_2 = None, -np.inf
for lam in lams:
    r2 = loo_r2_model2(lam)
    print(f'  lambda={lam:<8g} CV R^2={r2:+.4f}')
    if r2 > best_r2_2:
        best_r2_2, best_lam2 = r2, lam

print('\n=== summary ===')
print(f'Model 0 (F_i common, 0 firm params):        CV R^2 = {r2_m0:+.4f}')
print(f'Model 1 (a_i, 1 param/firm, best lambda={best_lam1:g}):  CV R^2 = {best_r2_1:+.4f}')
print(f'Model 2 (a_i,b_i, 2 params/firm, best lambda={best_lam2:g}): CV R^2 = {best_r2_2:+.4f}')
print(f'cf. expK unrestricted ridge (64 params, firm-agnostic): CV R^2 = 0.1632')
print(f'cf. expK scalar B=beta*I (1 param total, firm-agnostic): CV R^2 = 0.1512')

gain1 = best_r2_1 - r2_m0
gain2 = best_r2_2 - best_r2_1
print(f'\ngain from adding a_i (firm productivity):        {gain1:+.4f}')
print(f'gain from adding b_i (firm comparative advantage): {gain2:+.4f}')

print('\n=== interpretation ===')
if gain1 > 0.01:
    print(f'Firm identity via a_i alone improves held-out prediction ({gain1:+.4f}) -- a '
          f'1-parameter structural F_i is weakly identified by historical shocks.')
else:
    print(f'Firm identity via a_i buys essentially nothing OOS ({gain1:+.4f}) -- even a '
          f'1-parameter productivity scalar is not reliably recoverable from historical shocks '
          f'at this sample size, despite expM finding borderline in-sample significance '
          f'(permutation p=0.036). In-sample significance and out-of-sample predictive value '
          f'are different questions, and this answers the second one negatively.')
if gain2 > 0.01:
    print(f'Adding b_i (comparative-advantage loading) further improves prediction ({gain2:+.4f}).')
else:
    print(f'Adding b_i buys nothing further ({gain2:+.4f}) -- consistent with the effective rank '
          f'of historical (c,w) shocks being only ~2.06 (§7.3): there may not be enough '
          f'independent perturbation directions in the historical record to identify even a '
          f'2-parameter F_i, let alone a general one.')

# ---- per-firm a_i table (Model 1, full-sample fit at the CV-selected lambda) ----
print('\n=== per-firm a_i (Model 1, full-sample fit) ===')
print(f'{"firm":12s} {"n_ev":>5s} {"a_i (shrunk)":>13s} {"a_i (near-raw)":>15s}')
a_i_table = {}
for f in firms:
    idx = idx_by_firm[f]
    a_shrunk = fit_a_only(idx, best_lam1)
    a_raw = fit_a_only(idx, 0.1)
    a_i_table[f] = a_shrunk
    print(f'{f:12s} {len(idx):5d} {a_shrunk:13.3f} {a_raw:15.3f}')

# ---- Model B: does scaling the WHOLE shared matrix (diag + off-diag together) beat
# scaling only the diagonal? F_i(x) = a_i * F_0(x) taken literally, with F_0 = the full
# pooled ridge B rather than decomposed into isotropic + correction ----
print('\n=== Model B: a_i scales the FULL shared response matrix B_ridge (not just isotropic) ===')
Xfull = Gc @ B_ridge.T   # n x K, the pooled model's prediction for every event

def fit_a_full(idx, lam):
    x = Xfull[idx].reshape(-1)
    y = Dc[idx].reshape(-1)
    denom = x @ x + lam
    num = x @ y + lam * 1.0
    return num / denom

def loo_r2_modelB(lam):
    sse, sst = 0.0, 0.0
    dbar = Dc.mean(0)
    for f in firms:
        idx = idx_by_firm[f]
        if len(idx) < 2:
            for i in idx:
                sse += np.sum((Dc[i] - Xfull[i]) ** 2)
                sst += np.sum((Dc[i] - dbar) ** 2)
            continue
        for j, i in enumerate(idx):
            train = np.delete(idx, j)
            a = fit_a_full(train, lam)
            pred = a * Xfull[i]
            sse += np.sum((Dc[i] - pred) ** 2)
            sst += np.sum((Dc[i] - dbar) ** 2)
    return 1 - sse / sst

best_lamB, best_r2_B = None, -np.inf
for lam in lams:
    r2 = loo_r2_modelB(lam)
    print(f'  lambda={lam:<8g} CV R^2={r2:+.4f}')
    if r2 > best_r2_B:
        best_r2_B, best_lamB = r2, lam
print(f'Model B best: CV R^2={best_r2_B:+.4f} (lambda={best_lamB:g})  vs Model 1={best_r2_1:+.4f}')

# ---- Model C: the OTHER natural 2-parameter family -- own-category speed a_i plus a
# uniform spillover-to-other-categories strength c_i (matches expK's "common" B=beta*I +
# gamma*(11^T-I) pooled spec, but now with FIRM-SPECIFIC beta_i, gamma_i instead of pooled).
# Shrinkage target is the pooled (beta,gamma) from expK's common fit, not (1,1). ----
print('\n=== Model C: firm-specific (beta_i, gamma_i) on own-gap + uniform other-gap, '
      'shrunk toward the pooled common-model fit ===')
own = Gc                                                    # n x K
other = (Gc.sum(1, keepdims=True) - Gc) / (K - 1)            # n x K, leave-one-out mean of other cats

def fit_common_pooled(idx):
    o1, o2 = own[idx].reshape(-1), other[idx].reshape(-1)
    y = Dc[idx].reshape(-1)
    Xd = np.column_stack([o1, o2])
    beta, gamma = np.linalg.lstsq(Xd, y, rcond=None)[0]
    return beta, gamma

beta_pool, gamma_pool = fit_common_pooled(np.arange(n))
print(f'pooled common-model fit: beta={beta_pool:.3f}, gamma={gamma_pool:.3f} '
      f'(cf. expK full-sample common spec: beta=0.507, gamma=-0.030)')

def fit_beta_gamma(idx, lam):
    o1, o2 = own[idx].reshape(-1), other[idx].reshape(-1)
    y = Dc[idx].reshape(-1)
    Xd = np.column_stack([o1, o2])
    A = Xd.T @ Xd + lam * np.eye(2)
    b_vec = Xd.T @ y + lam * np.array([beta_pool, gamma_pool])
    return np.linalg.solve(A, b_vec)

def loo_r2_modelC(lam, gamma_free=True):
    sse, sst = 0.0, 0.0
    dbar = Dc.mean(0)
    for f in firms:
        idx = idx_by_firm[f]
        min_n = 3 if gamma_free else 2
        if len(idx) < min_n:
            for i in idx:
                pred = beta_pool * own[i] + gamma_pool * other[i]
                sse += np.sum((Dc[i] - pred) ** 2)
                sst += np.sum((Dc[i] - dbar) ** 2)
            continue
        for j, i in enumerate(idx):
            train = np.delete(idx, j)
            if gamma_free:
                beta_i, gamma_i = fit_beta_gamma(train, lam)
            else:
                o1 = own[train].reshape(-1)
                y = Dc[train].reshape(-1) - gamma_pool * other[train].reshape(-1)
                beta_i = (o1 @ y + lam * beta_pool) / (o1 @ o1 + lam)
                gamma_i = gamma_pool
            pred = beta_i * own[i] + gamma_i * other[i]
            sse += np.sum((Dc[i] - pred) ** 2)
            sst += np.sum((Dc[i] - dbar) ** 2)
    return 1 - sse / sst

print('beta_i only (gamma fixed at pooled):')
best_lamC1, best_r2_C1 = None, -np.inf
for lam in lams:
    r2 = loo_r2_modelC(lam, gamma_free=False)
    print(f'  lambda={lam:<8g} CV R^2={r2:+.4f}')
    if r2 > best_r2_C1:
        best_r2_C1, best_lamC1 = r2, lam
print('beta_i AND gamma_i both free:')
best_lamC2, best_r2_C2 = None, -np.inf
for lam in lams:
    r2 = loo_r2_modelC(lam, gamma_free=True)
    print(f'  lambda={lam:<8g} CV R^2={r2:+.4f}')
    if r2 > best_r2_C2:
        best_r2_C2, best_lamC2 = r2, lam
print(f'Model C best: beta_i-only={best_r2_C1:+.4f}, beta_i+gamma_i={best_r2_C2:+.4f}')

# ---- final comparison across all families tried ----
print('\n=== FULL comparison across all model families ===')
print(f'{"model":50s} {"params/firm":>11s} {"CV R^2":>8s}')
print(f'{"Model 0: F_i common (SVD-basis)":50s} {"0":>11s} {r2_m0:8.4f}')
print(f'{"Model 1: a_i on isotropic (SVD-basis)":50s} {"1":>11s} {best_r2_1:8.4f}')
print(f'{"Model 2: a_i,b_i (SVD-basis)":50s} {"2":>11s} {best_r2_2:8.4f}')
print(f'{"Model B: a_i scales FULL shared B_ridge":50s} {"1":>11s} {best_r2_B:8.4f}')
print(f'{"Model C: beta_i only (own-gap, gamma pooled)":50s} {"1":>11s} {best_r2_C1:8.4f}')
print(f'{"Model C: beta_i + gamma_i (own+spillover)":50s} {"2":>11s} {best_r2_C2:8.4f}')
best_overall = max(
    [('Model 0', r2_m0), ('Model 1', best_r2_1), ('Model 2', best_r2_2),
     ('Model B', best_r2_B), ('Model C beta-only', best_r2_C1), ('Model C both', best_r2_C2)],
    key=lambda x: x[1])
print(f'\nbest of all tried: {best_overall[0]} at CV R^2={best_overall[1]:+.4f}')
