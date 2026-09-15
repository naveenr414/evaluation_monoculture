"""Does pure gap-chasing (diagonal B, independent iid innovation per firm/category,
no shared technology or correlated shocks) reproduce the observed shape-cosine excess?

theta_{i,k}(t+1) = theta_{i,k}(t) + beta*(frontier_k(t) - theta_{i,k}(t)) + eta_{i,k}(t)
frontier_k(t) = max_j theta_{j,k}(t)   (endogenous -- the only channel coupling firms/categories)
eta iid across firm AND category -- this is the null: no genuine shared strengths.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import CORE

rng = np.random.default_rng(1)
K = len(CORE)
N = 10          # firms
BETA = 0.343    # calibrated scalar beta from expK/exp1
SIG_INNOV = np.sqrt(233.0)   # from expA's calibrated sigma^2_innov ~ 233 per category
T_BURNIN = 300
T_MEASURE = 200
N_REPS = 30

def run_once():
    theta = rng.normal(1480, 30, size=(N, K))  # iid initial profile, no shared shape
    coss = []
    for t in range(T_BURNIN + T_MEASURE):
        frontier = theta.max(axis=0)
        theta = theta + BETA * (frontier - theta) + rng.normal(0, SIG_INNOV, size=(N, K))
        if t >= T_BURNIN:
            Mc = theta - theta.mean(axis=1, keepdims=True)
            U = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
            md = U.mean(axis=0); mdn = np.linalg.norm(md)
            if mdn > 1e-9:
                md = md / mdn
                coss.append((U @ md).mean())
    return np.mean(coss)

sims = [run_once() for _ in range(N_REPS)]
print(f"simulated (gap-chasing only, iid shocks) shape-cosine: "
      f"mean={np.mean(sims):.3f}  sd={np.std(sims):.3f}  range=[{min(sims):.3f},{max(sims):.3f}]")
print(f"observed real shape-cosine (top-10 firms, row-demeaned): 0.780")
print(f"permutation null (no shared shape at all): 0.305")

# ---- robustness: rivals' frontier only (excludes self, matching the paper's actual gap definition) ----
def run_once_rivals():
    theta = rng.normal(1480, 30, size=(N, K))
    coss = []
    for t in range(T_BURNIN + T_MEASURE):
        # frontier_i,k = max over OTHER firms (rivals), so each firm chases a slightly different target
        frontier_excl = np.array([np.delete(theta, i, axis=0).max(axis=0) for i in range(N)])
        theta = theta + BETA * (frontier_excl - theta) + rng.normal(0, SIG_INNOV, size=(N, K))
        if t >= T_BURNIN:
            Mc = theta - theta.mean(axis=1, keepdims=True)
            U = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
            md = U.mean(axis=0); mdn = np.linalg.norm(md)
            if mdn > 1e-9:
                md = md / mdn
                coss.append((U @ md).mean())
    return np.mean(coss)

sims2 = [run_once_rivals() for _ in range(N_REPS)]
print(f"\nrobustness (rivals'-frontier, excludes self): mean={np.mean(sims2):.3f} sd={np.std(sims2):.3f}")

# sensitivity to beta
print("\nsensitivity to beta:")
for b in [0.15, 0.25, 0.343, 0.5, 0.7]:
    BETA_local = b
    def run_b():
        theta = rng.normal(1480, 30, size=(N, K))
        coss = []
        for t in range(T_BURNIN + T_MEASURE):
            frontier = theta.max(axis=0)
            theta = theta + BETA_local * (frontier - theta) + rng.normal(0, SIG_INNOV, size=(N, K))
            if t >= T_BURNIN:
                Mc = theta - theta.mean(axis=1, keepdims=True)
                U = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
                md = U.mean(axis=0); mdn = np.linalg.norm(md)
                if mdn > 1e-9:
                    md = md/mdn
                    coss.append((U@md).mean())
        return np.mean(coss)
    vals = [run_b() for _ in range(10)]
    print(f"  beta={b:.2f}  shape-cosine={np.mean(vals):.3f}")
