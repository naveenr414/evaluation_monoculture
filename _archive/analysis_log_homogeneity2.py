"""
Redo with f(x) = log(x) (not log(1+x)). log(x) has f'(x) = 1/x -> infinity
as x->0, just like sqrt's f'(x) = 0.5/sqrt(x) -> infinity. That blocks the
corner solutions seen with log(1+x) (which has finite f'(0)=1).

Known closed-form intuition: when win-rate differences are small, sigma'(diff)
~ sigma'(0) = 1/4 is roughly constant across k, so the FOC
    w_k * sigma'(diff) * f'(x_ik) = lambda_i  for all k
becomes (for f=log):  w_k / x_ik = const  =>  x_ik = w_k * B_i.
i.e. every firm allocates budget in *exactly the evaluator-weight proportions*,
regardless of B_i -- a much stronger/cleaner homogeneity result than sqrt's
"approximately similar" ratios.
"""

import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
EPS = 1e-6  # floor to keep log finite numerically


def f(x):
    return np.log(np.clip(x, EPS, None))


def f_prime(x):
    return 1.0 / np.clip(x, EPS, None)


def run_and_report(B, label):
    sim = CompetitionSimulation(N, K, np.array(B), W, W, f, f_prime)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
    theta = sim.theta(X)
    x_ratio = X[:, 0] / (X[:, 1] + 1e-12)
    predicted_ratio = W[0] / W[1]

    print(f"\n=== {label}  (B = {np.round(B, 3)}) ===")
    print(f"Allocations X:\n{np.round(X, 4)}")
    print(f"X normalized by B_i (should all equal w = {W} if x_ik = B_i*w_k):")
    print(np.round(X / np.array(B)[:, None], 4))
    print(f"x_{{i,1}}/x_{{i,2}} ratio per firm: {np.round(x_ratio, 4)}  (predicted w1/w2 = {predicted_ratio:.4f})")
    print(f"Raw outputs theta = f(x):\n{np.round(theta, 4)}")


run_and_report([B_TOTAL/3]*3, "Symmetric budgets")
run_and_report([5.0, 3.0, 1.0], "Heterogeneous budgets (moderate)")
run_and_report([7.5, 1.2, 0.3], "Heterogeneous budgets (extreme)")

eps = 0.05
run_and_report([eps, eps, B_TOTAL - 2*eps], "Extreme structure A (2 tiny, 1 huge)")
run_and_report([4.4, 4.3, B_TOTAL - 8.7 - 0.05 + 0.05], "Extreme structure B (2 large, 1 tiny)")

print("\nDone.")
