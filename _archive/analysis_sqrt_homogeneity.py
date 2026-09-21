"""
Does f(x) = sqrt(x) (i.e. power law with alpha=0.5) lead to OUTPUT
homogeneity across firms -- i.e. is the normalized output vector
(f(x_{i,1}), f(x_{i,2})) / ||.|| the same for every firm i at Nash,
even when firms have very different budgets B_i?

This is distinct from the f'(x) marginal-homogeneity question in
analysis_fprime_scatter.py: here we look at normalized *levels* theta_i,
not marginals.
"""

import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0


def f(x):
    return np.sqrt(np.clip(x, 0.0, None))


def f_prime(x):
    return 0.5 / np.sqrt(np.clip(x, 1e-12, None))


def normalize(row):
    n = np.linalg.norm(row)
    return row / n if n > 0 else row


def run_and_report(B, label):
    sim = CompetitionSimulation(N, K, np.array(B), W, W, f, f_prime)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-8, verbose=False)
    theta = sim.theta(X)                      # raw outputs f(x_{i,k})
    theta_norm = np.array([normalize(row) for row in theta])
    ratio = theta[:, 0] / (theta[:, 1] + 1e-12)   # theta_1 / theta_2 per firm

    print(f"\n=== {label}  (B = {np.round(B, 3)}) ===")
    print(f"Allocations X:\n{np.round(X, 4)}")
    print(f"Raw outputs theta = f(x):\n{np.round(theta, 4)}")
    print(f"Normalized outputs theta/||theta||:\n{np.round(theta_norm, 4)}")
    print(f"theta_1/theta_2 ratio per firm: {np.round(ratio, 4)}")
    print(f"  max/min ratio spread: {ratio.max()/ratio.min():.4f}")
    return theta, theta_norm, ratio


# -----------------------------------------------------------------------
# 1. Symmetric budgets -- trivially homogeneous by symmetry, sanity check
# -----------------------------------------------------------------------
run_and_report([B_TOTAL/3]*3, "Symmetric budgets")

# -----------------------------------------------------------------------
# 2. Heterogeneous budgets, moderate spread
# -----------------------------------------------------------------------
run_and_report([5.0, 3.0, 1.0], "Heterogeneous budgets (moderate)")

# -----------------------------------------------------------------------
# 3. Heterogeneous budgets, extreme spread
# -----------------------------------------------------------------------
run_and_report([7.5, 1.2, 0.3], "Heterogeneous budgets (extreme)")

# -----------------------------------------------------------------------
# 4. Extreme structure A from fprime_scatter (two tiny, one huge)
# -----------------------------------------------------------------------
eps = 0.05
run_and_report([eps, eps, B_TOTAL - 2*eps], "Extreme structure A (2 tiny, 1 huge)")

# -----------------------------------------------------------------------
# 5. Extreme structure B (two large/close, one tiny)
# -----------------------------------------------------------------------
run_and_report([4.4, 4.3, B_TOTAL - 8.7 - 0.05 + 0.05], "Extreme structure B (2 large, 1 tiny)")

print("\nDone.")
