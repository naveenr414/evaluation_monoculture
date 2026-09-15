"""A_{i,k} = A_i * A_k * D_{i,k}, with mean_k log D_{i,k} = 0 per firm --
i.e. log A_{i,k} = log A_i + log a_k + log D_{i,k}, log D zero-mean per firm.
Structurally the same move as alpha_i + delta_{i,k}, just multiplicative on
the level instead of additive -- check it behaves the same, since A enters
log(1+X) nonlinearly so it isn't guaranteed a priori.
"""
import numpy as np
from exp_dynamic_investment import simulate, measure, N, K

rng = np.random.default_rng(42)
a_k = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))
alpha_i = np.linspace(0.7, 1.3, N)
w = np.ones(K) / K
gamma = 0.1

def run(label, A, shock_sigma):
    A = np.clip(A, 0.05, None)
    theta_hist, X_hist = simulate(A, w, gamma, shock_sigma, rho_shock=0.0, seed=1)
    beta, pers, mdisp, rslope = measure(theta_hist)
    print(f'{label:30s} shock_sig={shock_sigma:4.1f}  beta={beta:6.3f}  persistence={pers:7.3f}'
          f'  mean_disp={mdisp:6.3f}  disp_%chg={rslope:+7.1%}')

print('=== A_i * A_k * D_{i,k}, log D zero-mean per firm (multiplicative analog of delta_ik) ===')
for sigma_logD in [0.1, 0.2, 0.35, 0.5]:
    logD = rng.normal(0, sigma_logD, size=(N, K))
    logD -= logD.mean(axis=1, keepdims=True)  # zero-mean per firm, in log space
    D = np.exp(logD)
    A = np.outer(alpha_i, a_k) * D
    for s in [1.0]:
        run(f'A_i*A_k*D_ik (sig_logD={sigma_logD})', A, s)

print()
print('=== reference: additive delta_ik at matched sigma (from last run) ===')
for delta_sigma in [0.1, 0.2, 0.35]:
    delta_ik = rng.normal(0, delta_sigma, size=(N, K))
    delta_ik -= delta_ik.mean(axis=1, keepdims=True)
    A3 = np.outer(alpha_i, np.ones(K)) + delta_ik
    run(f'alpha_i+delta_ik (sig={delta_sigma})', A3, 1.0)
