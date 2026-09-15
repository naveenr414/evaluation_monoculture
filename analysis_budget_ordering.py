"""
Verify: at Nash, does higher-budget firm invest more in ALL dims (full dominance)?
Check: for all pairs (i,j) with B_i > B_j, is x_{i,k} > x_{j,k} for all k?
"""

import math
import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 4, 5
N_CONFIGS = 15
RNG = np.random.default_rng(42)
a_vals = [0.1, 1.0, 5.0]

print(f"{'cfg':>4}  {'a':>5}  {'top_holds':>10}  {'full_holds':>10}  B  w")
for cfg in range(N_CONFIGS):
    B = np.sort(RNG.uniform(0.5, 5.0, size=N))[::-1]
    w = np.sort(RNG.dirichlet(np.ones(K)))[::-1]

    for a in a_vals:
        f_fn  = lambda x, a=a: math.log(max(x, 0) + a)
        fp_fn = lambda x, a=a: 1.0 / (max(x, 0) + a)
        sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
        X, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)

        top_ok  = all(X[i,0] >= X[j,0] - 1e-4 for i in range(N) for j in range(i+1,N))
        full_ok = all(X[i,k] >= X[j,k] - 1e-4
                      for i in range(N) for j in range(i+1,N) for k in range(K))
        print(f"{cfg:>4}  {a:>5}  {'YES' if top_ok else 'NO':>10}  {'YES' if full_ok else 'NO':>10}  "
              f"B={np.round(B,2)}  w={np.round(w,2)}")
