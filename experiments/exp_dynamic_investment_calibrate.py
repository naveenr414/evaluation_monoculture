"""Grid search: can A_i * A_k * D_{i,k}, log D_{i,k}=u_i v_k (d=1, since d didn't
matter at matched variance), jointly hit the REAL targets
    beta = 0.43 (exp1_evalchasing), persistence = 0.339 (exp_persistence),
and keep dispersion roughly flat over time (not the real absolute level --
units are arbitrary here -- just the STABILITY, i.e. no strong trend)?
"""
import numpy as np
from exp_dynamic_investment import simulate, measure, N, K, BURN_IN, T

BETA_TARGET, PERS_TARGET = 0.43, 0.339

a_k = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))
alpha_i = np.linspace(0.7, 1.3, N)
w = np.ones(K) / K
gamma = 0.1


def make_logD(sigma_target, seed):
    r = np.random.default_rng(seed)
    u = r.normal(0, 1, N)
    v = r.normal(0, 1, K)
    raw = np.outer(u, v)
    raw -= raw.mean(axis=1, keepdims=True)
    raw *= sigma_target / raw.std()
    return raw


def dispersion_by_quartile(theta_hist):
    disp = theta_hist[BURN_IN:].std(axis=1).mean(axis=1)
    q = np.array_split(disp, 4)
    return [x.mean() for x in q]


rows = []
for sigma_logD in [0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
    for shock_sigma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        logD = make_logD(sigma_logD, seed=7)
        A = np.outer(alpha_i, a_k) * np.exp(logD)
        A = np.clip(A, 0.05, None)
        theta_hist, X_hist = simulate(A, w, gamma, shock_sigma, rho_shock=0.0, seed=1)
        beta, pers, mdisp, rslope = measure(theta_hist)
        loss = ((beta - BETA_TARGET) / BETA_TARGET) ** 2 + ((pers - PERS_TARGET) / PERS_TARGET) ** 2
        rows.append(dict(sigma_logD=sigma_logD, shock_sigma=shock_sigma, beta=beta,
                          persistence=pers, mean_disp=mdisp, disp_pct=rslope, loss=loss))

import pandas as pd
df = pd.DataFrame(rows).sort_values('loss')
pd.set_option('display.width', 120)
print('=== best joint fits to (beta=0.43, persistence=0.339) ===')
print(df.head(8).round(4).to_string(index=False))

best = df.iloc[0]
print()
print(f"Best fit: sigma_logD={best.sigma_logD}, shock_sigma={best.shock_sigma}  "
      f"-> beta={best.beta:.3f} (target 0.43), persistence={best.persistence:.3f} (target 0.339), "
      f"disp drift={best.disp_pct:+.1%} over the measured window")

logD = make_logD(best.sigma_logD, seed=7)
A = np.outer(alpha_i, a_k) * np.exp(logD)
A = np.clip(A, 0.05, None)
theta_hist, X_hist = simulate(A, w, gamma, best.shock_sigma, rho_shock=0.0, seed=1)
quarts = dispersion_by_quartile(theta_hist)
print(f"Dispersion (std across firms, avg over categories) by quartile of the measured window: "
      f"{[round(q,3) for q in quarts]}")
print(f"  -> ratio last/first quartile: {quarts[-1]/quarts[0]:.2f}x")

# also check robustness across 5 random seeds for u,v at the best (sigma, shock)
print()
print('=== robustness across 5 random u,v draws at the best-fit (sigma_logD, shock_sigma) ===')
for seed in range(5):
    logD = make_logD(best.sigma_logD, seed=seed)
    A = np.outer(alpha_i, a_k) * np.exp(logD)
    A = np.clip(A, 0.05, None)
    theta_hist, X_hist = simulate(A, w, gamma, best.shock_sigma, rho_shock=0.0, seed=1)
    beta, pers, mdisp, rslope = measure(theta_hist)
    print(f'  seed={seed}  beta={beta:.3f}  persistence={pers:.3f}  disp_%chg={rslope:+.1%}')
