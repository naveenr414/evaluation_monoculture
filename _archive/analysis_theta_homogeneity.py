"""
Homogeneity in OUTPUT LEVELS theta_{i,k} = f(x_{i,k}), not in allocations x.

For f=log(x): x_{i,k} = B_i * w_k (exact, derived analytically and confirmed
numerically), so
    theta_{i,k} = log(x_{i,k}) = log(B_i) + log(w_k)
The firm-specific part log(B_i) is an ADDITIVE shift common to all k. So the
*difference* theta_{i,1} - theta_{i,2} = log(w_1) - log(w_2) = log(w_1/w_2)
is EXACTLY the same across firms (perfect homogeneity in theta-differences),
even though raw theta and norm-normalized theta are NOT the same across firms
(they differ by the additive log(B_i) shift / are not simply rescaled).

For f=sqrt(x): x_{i,k} is not simply proportional to B_i in general (only
w_k in the log case decouples like that), so we check numerically whether
theta differences / normalized theta vectors are similar across firms.
"""

import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
EPS = 1e-6

BUDGETS = {
    "Symmetric":            [B_TOTAL/3]*3,
    "Moderate hetero":      [5.0, 3.0, 1.0],
    "Extreme hetero":       [7.5, 1.2, 0.3],
    "Structure A (2 tiny)": [0.05, 0.05, B_TOTAL - 0.1],
    "Structure B (2 big)":  [4.4, 4.3, 0.3],
}


def normalize(row):
    n = np.linalg.norm(row)
    return row / n if n > 0 else row


def analyze(f, f_prime, label):
    print(f"\n{'#'*70}\n#  f(x) = {label}\n{'#'*70}")
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, f_prime)
        X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        diff = theta[:, 0] - theta[:, 1]                  # theta_1 - theta_2 per firm
        theta_norm = np.array([normalize(row) for row in theta])

        print(f"\n--- {name}  (B={np.round(B,3)}) ---")
        print(f"theta:\n{np.round(theta, 4)}")
        print(f"theta_1 - theta_2 per firm: {np.round(diff, 4)}  "
              f"(spread max-min = {diff.max()-diff.min():.4f})")
        print(f"normalized theta (theta/||theta||):\n{np.round(theta_norm, 4)}")


# f = sqrt(x)
analyze(lambda x: np.sqrt(np.clip(x, 0.0, None)),
        lambda x: 0.5 / np.sqrt(np.clip(x, 1e-12, None)),
        "sqrt(x)")

# f = log(x)
analyze(lambda x: np.log(np.clip(x, EPS, None)),
        lambda x: 1.0 / np.clip(x, EPS, None),
        "log(x)")

print("\nDone.")
