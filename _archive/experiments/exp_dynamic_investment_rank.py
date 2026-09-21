"""Same dynamic investment game as exp_dynamic_investment.py, but comparing
THREE structural forms of the ability matrix A_{i,k} at matched heterogeneity
scale, all with purely IID (rho=0) shocks so persistence isn't smuggled in
via shock autocorrelation:

  (1) A_k only        : A[i,k] = a_k          -- firms are literally identical
  (2) A_i * A_k        : A[i,k] = alpha_i * a_k -- rank-1 / "parallel technology,
                          differ only in scale" (Theorem 9's F_i parallel case)
  (3) alpha_i + delta_{i,k} : additive, delta zero-summed per firm (F3's winning form)
"""
import numpy as np
from exp_dynamic_investment import simulate, measure, N, K, BURN_IN

rng = np.random.default_rng(42)
a_k = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))  # fixed category "shape"
alpha_i = np.linspace(0.7, 1.3, N)                                     # fixed firm "scale"
w = np.ones(K) / K
gamma = 0.1

def run(label, A, shock_sigma):
    A = np.clip(A, 0.05, None)
    theta_hist, X_hist = simulate(A, w, gamma, shock_sigma, rho_shock=0.0, seed=1)
    beta, pers, mdisp, rslope = measure(theta_hist)
    print(f'{label:28s} shock_sig={shock_sigma:4.1f}  beta={beta:6.3f}  persistence={pers:7.3f}'
          f'  mean_disp={mdisp:6.3f}  disp_%chg={rslope:+7.1%}')

print('=== (1) A_k only: firms are IDENTICAL, all heterogeneity is transient shock noise ===')
A1 = np.tile(a_k, (N, 1))
for s in [0.5, 1.0, 2.0]:
    run('A_k only', A1, s)

print()
print('=== (2) A_i * A_k : rank-1, same "technology direction" for every firm, differ only in scale ===')
A2 = np.outer(alpha_i, a_k)
for s in [0.5, 1.0, 2.0]:
    run('A_i * A_k (rank-1)', A2, s)

print()
print('=== (3) alpha_i + delta_{i,k} : genuine per-firm category profile (delta zero-summed per firm) ===')
for delta_sigma in [0.1, 0.2, 0.35]:
    delta_ik = rng.normal(0, delta_sigma, size=(N, K))
    delta_ik -= delta_ik.mean(axis=1, keepdims=True)
    A3 = np.outer(alpha_i, np.ones(K)) + delta_ik
    for s in [1.0]:
        run(f'alpha_i+delta_ik (sig={delta_sigma})', A3, s)
