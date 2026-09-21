"""
Same broad search as analysis_p02_search.py but at p=0.5 (sqrt).
"""

import numpy as np
from itertools import combinations
from competition_simulation import CompetitionSimulation

K = 2
W = np.array([0.7, 0.3])
EPS = 1e-6
P = 0.5


def make_power(p):
    f = lambda x: np.clip(x, EPS, None) ** p
    fp = lambda x: p * np.clip(x, EPS, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def min_pairwise_cos(theta, N):
    return min(cos_sim(theta[i], theta[j]) for i, j in combinations(range(N), 2))


f, fp = make_power(P)

worst_val, worst_B = 1.0, None

rng = np.random.default_rng(0)
for _ in range(400):
    N = 3
    B_total = rng.uniform(1.0, 20.0)
    shares = rng.dirichlet(rng.uniform(0.05, 2.0, size=N))
    B = shares * B_total
    B = np.clip(B, 1e-4, None)
    sim = CompetitionSimulation(N, K, B, W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
    theta = sim.theta(X)
    v = min_pairwise_cos(theta, N)
    if v < worst_val:
        worst_val, worst_B = v, B.copy()

print(f"Random search (N=3): worst min cos_sim = {worst_val:.6f}  at B={np.round(worst_B,4)}")

best2, bestB2 = worst_val, worst_B
for eps in np.logspace(-5, -0.5, 40):
    B_total = 9.0
    if 2 * eps >= B_total:
        continue
    B = np.array([eps, eps, B_total - 2 * eps])
    sim = CompetitionSimulation(3, K, B, W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
    theta = sim.theta(X)
    v = min_pairwise_cos(theta, 3)
    if v < best2:
        best2, bestB2 = v, B.copy()

print(f"Extreme structure sweep: worst min cos_sim = {best2:.6f}  at B={np.round(bestB2,6)}")

best3, bestB3 = best2, bestB2
for ratio_pow in range(1, 13):
    B_small = 10.0 ** (-ratio_pow)
    B = np.array([B_small, 20.0])
    sim = CompetitionSimulation(2, K, B, W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
    theta = sim.theta(X)
    v = min_pairwise_cos(theta, 2)
    print(f"  N=2, B={np.round(B,8)}: X={np.round(X,6)}  theta={np.round(theta,6)}  cos_sim={v:.8f}")
    if v < best3:
        best3, bestB3 = v, B.copy()

print(f"\nOverall worst min cos_sim found at p=0.5: {best3:.8f}  at B={np.round(bestB3,6)}")
