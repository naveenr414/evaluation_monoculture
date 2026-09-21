"""
Price of Anarchy: Nash social welfare vs. optimal social welfare.
Social welfare = sum_k w_k * max_i f(x_{i,k}).

Optimal: enumerate all N^K assignments of dims to firms, for each solve
each firm's within-assignment allocation via scipy, take best.
Nash: find_nash_br then evaluate.
"""

import math
import itertools
import numpy as np
from scipy.optimize import minimize
from competition_simulation import CompetitionSimulation

# ── problem setup ─────────────────────────────────────────────────────────────
N, K = 5, 6
B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
w = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
w = w / w.sum()

FUNCTIONS = [
    ("log(x+1)",       lambda x: math.log(max(x,0)+1),        lambda x: 1/(max(x,0)+1)),
    ("log(x+10)",      lambda x: math.log(max(x,0)+10),       lambda x: 1/(max(x,0)+10)),
    ("sqrt(x)",        lambda x: math.sqrt(max(x,0)),          lambda x: 0.5/math.sqrt(max(x,1e-12))),
    ("1-e^{-x}",       lambda x: 1-math.exp(-max(x,0)),       lambda x: math.exp(-max(x,0))),
    ("1-(1+x)^{-2}",   lambda x: 1-(1+max(x,0))**(-2),        lambda x: 2*(1+max(x,0))**(-3)),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def social_welfare(X, w, f_fn):
    """sum_k w_k * max_i f(x_{i,k})"""
    return sum(w[k] * max(f_fn(X[i, k]) for i in range(len(X)))
               for k in range(len(w)))

def optimal_allocation(assignment, B, w, f_fn):
    """
    Given assignment: list of length K, assignment[k] = firm index.
    For each firm i, maximize sum_{k: assignment[k]==i} w_k f(x_{i,k})
    subject to sum x_{i,k} = B[i], x_{i,k} >= 0.
    Returns X (N x K), value.
    """
    N = len(B)
    K = len(w)
    X = np.zeros((N, K))

    for i in range(N):
        dims = [k for k in range(K) if assignment[k] == i]
        if not dims:
            continue
        wi = np.array([w[k] for k in dims])
        bi = B[i]

        if len(dims) == 1:
            X[i, dims[0]] = bi
            continue

        # maximize sum wi * f(xi) s.t. sum xi = bi, xi >= 0
        def neg_obj(x):
            return -sum(wi[j] * f_fn(x[j]) for j in range(len(dims)))

        x0 = wi / wi.sum() * bi
        cons = {'type': 'eq', 'fun': lambda x: x.sum() - bi}
        bounds = [(0, None)] * len(dims)
        res = minimize(neg_obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 500})
        for j, k in enumerate(dims):
            X[i, k] = max(res.x[j], 0)

    return X

def find_optimal(B, w, f_fn):
    """Enumerate all N^K assignments, return best X and welfare."""
    N, K = len(B), len(w)
    best_val = -np.inf
    best_X = None
    for assignment in itertools.product(range(N), repeat=K):
        X = optimal_allocation(assignment, B, w, f_fn)
        val = social_welfare(X, w, f_fn)
        if val > best_val:
            best_val = val
            best_X = X.copy()
    return best_X, best_val

# ── run ───────────────────────────────────────────────────────────────────────
print(f"N={N}, K={K}, B={B}, w={np.round(w,3)}")
print(f"Enumerating {N**K} assignments per function...\n")
print(f"{'Function':<18}  {'Nash SW':>9}  {'Opt SW':>9}  {'PoA':>7}  {'Opt assignment'}")
print("-" * 75)

for name, f_fn, fp_fn in FUNCTIONS:
    # Nash
    sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=300, tol=1e-7, verbose=False)
    nash_sw = social_welfare(X_nash, w, f_fn)

    # Optimal
    X_opt, opt_sw = find_optimal(B, w, f_fn)

    # Which firm covers which dims at optimum
    assignment_str = " ".join(
        f"d{k+1}→F{np.argmax(X_opt[:,k])+1}" for k in range(K)
    )

    poa = nash_sw / opt_sw if opt_sw > 0 else float('nan')
    print(f"{name:<18}  {nash_sw:>9.4f}  {opt_sw:>9.4f}  {poa:>7.3f}  {assignment_str}")
