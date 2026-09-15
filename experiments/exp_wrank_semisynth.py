"""Semi-synthetic experiment: fix a TRUE heterogeneous ability matrix
A_{i,k} = A_i * A_k * D_{i,k} (log D = rank-2 interaction, fixed throughout --
this is the F_i we're trying to identify), and vary only how the EVALUATOR
WEIGHTS w_t move over time: constant, or confined to a rank-r subspace of the
simplex tangent space, r in {0,1,2,4,7=K-1}, at MATCHED total variance across r
(so we're isolating the effect of rank, not of how much w moves).

For each r, measure:
  1) gap-chasing beta
  2) persistence (comparative-advantage signature)
  3) RECOVERY of the true F_i: OLS slope of investment a_{i,k,t} on the
     w_t deviation in category k, per (firm,category) cell, two-way demeaned,
     cosine-compared to the TRUE two-way-demeaned log D_{i,k}. This is the
     Theorem-9-style identification channel (uses only observable a_{i,k,t}
     and w_t -- NOT X or theta's nonlinear curvature).
  4) dispersion trend
  5) homogeneity H = mean pairwise cosine of RAW theta_i vectors (the paper's H)
"""
import numpy as np
from exp_dynamic_investment import (N, K, T, BURN_IN, best_response, measure,
                                     two_way_demean)

GAMMA = 0.1
SHOCK_SIGMA = 0.5
SIGMA_W_TOTAL = 0.4
SIGMA_LOGD_TRUE = 0.25
D_RANK_TRUE = 2

alpha_i = np.linspace(0.7, 1.3, N)
a_k = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))

rng0 = np.random.default_rng(123)
U_true = rng0.normal(size=(N, D_RANK_TRUE))
V_true = rng0.normal(size=(K, D_RANK_TRUE))
logD_true = U_true @ V_true.T
logD_true -= logD_true.mean(axis=1, keepdims=True)
logD_true *= SIGMA_LOGD_TRUE / logD_true.std()
A_true = np.clip(np.outer(alpha_i, a_k) * np.exp(logD_true), 0.05, None)
D_true_dm = two_way_demean(logD_true)


def build_w_sequence(rank_r, sigma_total, seed):
    rng = np.random.default_rng(seed)
    w_bar = np.ones(K) / K
    if rank_r == 0:
        return np.tile(w_bar, (T, 1))
    raw = rng.normal(size=(K, rank_r))
    raw -= raw.mean(axis=0, keepdims=True)   # zero-sum columns (simplex tangent space)
    Q, _ = np.linalg.qr(raw)
    basis = Q[:, :rank_r]
    sigma_c = sigma_total / np.sqrt(rank_r)  # matched total variance across r
    coeffs = rng.normal(0, sigma_c, size=(T, rank_r))
    pert = coeffs @ basis.T
    logits = np.log(w_bar)[None, :] + pert
    w_t = np.exp(logits)
    return w_t / w_t.sum(axis=1, keepdims=True)


def simulate_wt(A, w_seq, gamma, shock_sigma, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((N, K))
    shock = np.zeros((N, K))
    theta_pub = A * np.log1p(X) + shock
    theta_hist = np.zeros((T, N, K))
    X_hist = np.zeros((T, N, K))
    a_hist = np.zeros((T, N, K))
    for t in range(T):
        wt = w_seq[t]
        a = np.zeros((N, K))
        for i in range(N):
            others = np.arange(N) != i
            for k in range(K):
                a[i, k] = best_response(A[i, k], X[i, k], wt[k], gamma, theta_pub[others, k])
        X = X + a
        shock = shock_sigma * rng.standard_normal((N, K))  # iid, rho=0
        theta_pub = A * np.log1p(X) + shock
        theta_hist[t], X_hist[t], a_hist[t] = theta_pub, X, a
    return theta_hist, X_hist, a_hist


def homogeneity(theta_hist):
    Hs = []
    for t in range(BURN_IN, T):
        Th = theta_hist[t]
        nrm = Th / np.linalg.norm(Th, axis=1, keepdims=True)
        C = nrm @ nrm.T
        iu = np.triu_indices(N, 1)
        Hs.append(C[iu].mean())
    Hs = np.array(Hs)
    trend = np.polyfit(np.arange(len(Hs)), Hs, 1)[0] * len(Hs)
    return Hs.mean(), trend


def recovery(w_seq, a_hist):
    wk = w_seq[BURN_IN:]
    dwk = wk - wk.mean(axis=0, keepdims=True)
    a_post = a_hist[BURN_IN:]
    S_hat = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            x = dwk[:, k]
            if x.std() > 1e-9:
                y = a_post[:, i, k]
                S_hat[i, k] = np.sum(x * (y - y.mean())) / np.sum(x * x)
    S_dm = two_way_demean(S_hat)
    denom = np.linalg.norm(S_dm) * np.linalg.norm(D_true_dm)
    return float(np.sum(S_dm * D_true_dm) / denom) if denom > 1e-12 else np.nan


print(f'true A range: [{A_true.min():.3f}, {A_true.max():.3f}]   '
      f'true log D effective rank check: sigma_logD={SIGMA_LOGD_TRUE}, generative rank={D_RANK_TRUE}')
print()
print(f'{"rank_r":>7} {"beta":>7} {"persist":>8} {"recov_cos":>10} {"mean_disp":>10} {"disp_%chg":>10} {"H_mean":>8} {"H_trend":>8}')
for r in [0, 1, 2, 4, 7]:
    w_seq = build_w_sequence(r, SIGMA_W_TOTAL, seed=77)
    theta_hist, X_hist, a_hist = simulate_wt(A_true, w_seq, GAMMA, SHOCK_SIGMA, seed=1)
    beta, pers, mdisp, rslope = measure(theta_hist)
    rec = recovery(w_seq, a_hist)
    hmean, htrend = homogeneity(theta_hist)
    rec_str = f'{rec:10.3f}' if not np.isnan(rec) else '       n/a'
    print(f'{r:7d} {beta:7.3f} {pers:8.3f} {rec_str} {mdisp:10.3f} {rslope:+9.1%} {hmean:8.3f} {htrend:+8.3f}')
