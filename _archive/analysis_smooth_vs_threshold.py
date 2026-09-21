"""
Is there a hard threshold p* below which parallelism is EXACT, or is it a
smooth decay where even tiny p has some (vanishingly small) deviation that
just needs a sufficiently extreme budget ratio to reveal?

Strategy: for several small p, push the "extreme structure" (two tiny budgets,
one huge) sweep to VERY small eps (down to 1e-8, way more extreme than
before) and see if worst-case cos_sim keeps degrading as eps shrinks, or
saturates at exactly 1.0.
"""

import numpy as np
from itertools import combinations
from competition_simulation import CompetitionSimulation

K = 2
W = np.array([0.7, 0.3])
EPS_FLOOR = 1e-9
B_TOTAL = 9.0


def make_power(p):
    f = lambda x: np.clip(x, EPS_FLOOR, None) ** p
    fp = lambda x: p * np.clip(x, EPS_FLOOR, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


P_VALUES = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
EPS_SWEEP = np.logspace(-1, -8, 15)   # budget of the two tiny firms

for p in P_VALUES:
    f, fp = make_power(p)
    worst = 1.0
    worst_eps = None
    for eps in EPS_SWEEP:
        B = np.array([eps, eps, B_TOTAL - 2 * eps])
        sim = CompetitionSimulation(3, K, B, W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=400, tol=1e-11, verbose=False)
        theta = sim.theta(X)
        v = min(cos_sim(theta[i], theta[j]) for i, j in combinations(range(3), 2))
        if v < worst:
            worst, worst_eps = v, eps
    print(f"p={p:<6} worst cos_sim over eps in [1e-1,1e-8] = {worst:.10f}  (at eps={worst_eps:.1e})  deviation from 1 = {1-worst:.3e}")
