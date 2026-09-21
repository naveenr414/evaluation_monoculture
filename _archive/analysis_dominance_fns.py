"""
Check full dominance (x_{h,k} > x_{l,k} for all k) across all 5 production functions,
many random configs.
"""

import math
import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 4, 5
N_CONFIGS = 20
RNG = np.random.default_rng(42)

FUNCTIONS = [
    ("log(x+1)",   lambda x: math.log(max(x,0)+1),      lambda x: 1/(max(x,0)+1)),
    ("log(x+100)", lambda x: math.log(max(x,0)+100),    lambda x: 1/(max(x,0)+100)),
    ("x^0.5",      lambda x: max(x,0)**0.5,              lambda x: 0.5*max(x,1e-12)**-0.5),
    ("x^0.99",     lambda x: max(x,1e-12)**0.99,         lambda x: 0.99*max(x,1e-12)**-0.01),
    ("x",          lambda x: max(x,0),                   lambda x: 1.0),
    ("1-e^-x",     lambda x: 1-math.exp(-max(x,0)),      lambda x: math.exp(-max(x,0))),
]

print(f"{'Function':<12} {'violations':>10} {'near_ties':>10} {'total_pairs':>12} {'violation_rate':>15}")

for label, f_fn, fp_fn in FUNCTIONS:
    violations, near_ties, total_pairs = 0, 0, 0
    for cfg in range(N_CONFIGS):
        B = np.sort(RNG.uniform(0.5, 5.0, size=N))[::-1]
        w = np.sort(RNG.dirichlet(np.ones(K)))[::-1]
        sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
        X, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
        for i in range(N):
            for j in range(i+1, N):
                for k in range(K):
                    total_pairs += 1
                    diff = X[i,k] - X[j,k]
                    if abs(diff) < 1e-4:
                        near_ties += 1
                    elif diff < 0:
                        violations += 1
    rate = violations / total_pairs
    print(f"{label:<12} {violations:>10} {near_ties:>10} {total_pairs:>12} {rate:>15.4%}")
