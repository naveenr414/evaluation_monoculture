"""log D_{i,k} = sum_{r=1}^d u_i^r v_k^r  (rank-d bilinear interaction), zero-mean
per firm, rescaled to a fixed target std so we compare RANK at matched total
variance, not variance at matched rank. A_{i,k} = A_i * A_k * D_{i,k}.

Two questions:
  (1) do beta/persistence/dispersion depend on d at fixed total variance of logD?
      (they are pooled second-moment statistics -- no a priori reason they should
      be rank-sensitive, unlike an SVD-based statistic)
  (2) if we run the REAL pipeline (fit A_hat_{i,k} from the simulated panel, then
      take its participation ratio / effective rank), do we recover d -- and how
      much does shock noise attenuate it? This is the toy-model analog of
      Experiment C / J's effective-rank methodology.
"""
import numpy as np
from exp_dynamic_investment import simulate, measure, N, K, BURN_IN, T

rng = np.random.default_rng(42)
a_k = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))
alpha_i = np.linspace(0.7, 1.3, N)
w = np.ones(K) / K
gamma = 0.1


def make_logD(d, sigma_target, seed):
    r = np.random.default_rng(seed)
    U = r.normal(0, 1, size=(N, d))
    V = r.normal(0, 1, size=(K, d))
    raw = U @ V.T
    raw -= raw.mean(axis=1, keepdims=True)          # zero-mean per firm (as specified)
    raw *= sigma_target / raw.std()                  # match total variance across d
    return raw


def two_way_demean(M):
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def participation_ratio(M):
    Md = two_way_demean(M)
    s = np.linalg.svd(Md, compute_uv=False)
    lam = s ** 2
    return (lam.sum() ** 2) / (lam ** 2).sum()


def recover_A(theta_hist, X_hist):
    """OLS-through-origin per (i,k) cell, using post-burn-in time variation:
    theta_t ~ A_ik * log1p(X_t)."""
    lx = np.log1p(X_hist[BURN_IN:])          # (T', N, K)
    th = theta_hist[BURN_IN:]
    num = (th * lx).sum(axis=0)
    den = (lx * lx).sum(axis=0)
    return num / den


print('=== (1) beta / persistence / dispersion vs rank d, at MATCHED total variance ===')
print(f'{"d":>3} {"sigma_logD":>10} {"beta":>7} {"persistence":>12} {"mean_disp":>10} {"disp_%chg":>10}')
for sigma_target in [0.2, 0.35]:
    for d in [1, 2, 3]:
        logD = make_logD(d, sigma_target, seed=100 + d)
        A = np.outer(alpha_i, a_k) * np.exp(logD)
        A = np.clip(A, 0.05, None)
        theta_hist, X_hist = simulate(A, w, gamma, shock_sigma=1.0, rho_shock=0.0, seed=1)
        beta, pers, mdisp, rslope = measure(theta_hist)
        print(f'{d:3d} {sigma_target:10.2f} {beta:7.3f} {pers:12.3f} {mdisp:10.3f} {rslope:+9.1%}')

print()
print('=== (2) true vs RECOVERED effective rank (participation ratio) of log A_ik, from the simulated panel ===')
print(f'{"d":>3} {"sigma_logD":>10} {"shock_sig":>10} {"PR_true":>8} {"PR_recovered":>13}')
for sigma_target in [0.2, 0.35]:
    for d in [1, 2, 3]:
        for shock_sigma in [0.5, 1.5]:
            logD = make_logD(d, sigma_target, seed=100 + d)
            A = np.outer(alpha_i, a_k) * np.exp(logD)
            A = np.clip(A, 0.05, None)
            theta_hist, X_hist = simulate(A, w, gamma, shock_sigma, rho_shock=0.0, seed=1)
            A_hat = recover_A(theta_hist, X_hist)
            pr_true = participation_ratio(np.log(A))
            pr_hat = participation_ratio(np.log(np.clip(A_hat, 1e-6, None)))
            print(f'{d:3d} {sigma_target:10.2f} {shock_sigma:10.2f} {pr_true:8.2f} {pr_hat:13.2f}')
