"""
Same output-homogeneity question as analysis_sqrt_homogeneity.py, but with
f(x) = log(1 + x) (log needs an offset to stay finite / concave on x>=0
with f(0)=0, so we use log(1+x) rather than log(x)).
"""

import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0


def f(x):
    return np.log1p(np.clip(x, 0.0, None))


def f_prime(x):
    return 1.0 / (1.0 + np.clip(x, 0.0, None))


def normalize(row):
    n = np.linalg.norm(row)
    return row / n if n > 0 else row


def run_and_report(B, label):
    sim = CompetitionSimulation(N, K, np.array(B), W, W, f, f_prime)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-8, verbose=False)
    theta = sim.theta(X)
    theta_norm = np.array([normalize(row) for row in theta])
    ratio = theta[:, 0] / (theta[:, 1] + 1e-12)

    print(f"\n=== {label}  (B = {np.round(B, 3)}) ===")
    print(f"Allocations X:\n{np.round(X, 4)}")
    print(f"Raw outputs theta = f(x):\n{np.round(theta, 4)}")
    print(f"Normalized outputs theta/||theta||:\n{np.round(theta_norm, 4)}")
    print(f"theta_1/theta_2 ratio per firm: {np.round(ratio, 4)}")
    print(f"  max/min ratio spread: {ratio.max()/ratio.min():.4f}")
    return theta, theta_norm, ratio


run_and_report([B_TOTAL/3]*3, "Symmetric budgets")
run_and_report([5.0, 3.0, 1.0], "Heterogeneous budgets (moderate)")
run_and_report([7.5, 1.2, 0.3], "Heterogeneous budgets (extreme)")

eps = 0.05
run_and_report([eps, eps, B_TOTAL - 2*eps], "Extreme structure A (2 tiny, 1 huge)")
run_and_report([4.4, 4.3, B_TOTAL - 8.7 - 0.05 + 0.05], "Extreme structure B (2 large, 1 tiny)")

print("\nDone.")
