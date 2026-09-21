"""
Does f'(x_i) = (f'(x_{i,1}), f'(x_{i,2})) lie on a line through the origin
across firms i, for f(x) = log(x), f(x) = 1[x>0] (approximated by a steep
logistic), and f(x) = sqrt(x)?

Lessons applied from the theta investigation:
 - enforce x_{i,k} >= 1 (floor) so log(x) and sqrt(x) derivatives don't
   blow up / hit domain issues near 0, and so results are comparable
   across all three f's on the same budget grid.
 - use a properly-seeded best-response solver: at extreme budget disparity
   the flat-gradient region near saturation causes SLSQP to stall at the
   initial guess if only warm-started from the previous iterate, so we
   always also try x0 = clip(B_i * w, floor, ...) and the equal split.
 - use several increasing disparity levels (10x, 100x, 1000x) to see if
   the line-through-origin property holds as disparity grows, not just at
   one point.
 - measure "lies on a line through the origin" via the normalized
   (unit-vector) row space: take v_i = f'(x_i)/||f'(x_i)||, stack into an
   (N,K) matrix, and look at sigma_2/sigma_1 of its SVD. 0 = all v_i
   identical (exact ray).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

N, K = 3, 2
W = np.array([0.7, 0.3])
FLOOR = 1.0

BUDGETS = {
    "Wide (10x)":   [200.0, 60.0, 20.0],
    "Wide (100x)":  [2000.0, 60.0, 20.0],
    "Wide (1000x)": [20000.0, 60.0, 20.1],
}


def firm_objective(x_i, i, X, f, w):
    th_i = np.array([f(x_i[k]) for k in range(K)])
    th = np.array([[f(X[j, k]) for k in range(K)] for j in range(N)])
    total = 0.0
    for j in range(N):
        if j != i:
            total += float(np.sum(w * expit(th_i - th[j])))
    return total


def best_response(i, X, B, f, w, floor):
    Bi = B[i]
    free = Bi - floor * K
    bounds = [(floor, Bi - floor * (K - 1))] * K
    constraints = [{"type": "ineq", "fun": lambda x: Bi - np.sum(x)}]
    rng = np.random.default_rng(0)
    x0s = [X[i].copy(), np.clip(Bi * w, floor, Bi - floor * (K - 1)), np.full(K, Bi / K)]
    for _ in range(6):
        alpha = rng.dirichlet(np.ones(K))
        x0s.append(floor + alpha * free)
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -firm_objective(x, i, X, f, w), x0,
                        method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"ftol": 1e-14, "maxiter": 1000})
        if res.success and -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, floor, None)
    return best_x


def find_nash(B, f, w, floor, max_iter=300, tol=1e-11):
    X = np.array([np.clip(Bi * w, floor, Bi - floor * (K - 1)) for Bi in B])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, B, f, w, floor)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X


def through_origin_ratio(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    if np.any(norms < 1e-300):
        return None, M  # degenerate (zero vector) -- can't normalize a direction
    unit = M / norms
    try:
        s = np.linalg.svd(unit, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, unit
    return float(s[1] / s[0]) if len(s) > 1 and s[0] > 1e-12 else 0.0, unit


def make_step(beta, c=0.0):
    f = lambda x: float(expit(beta * (x - c)))
    fp = lambda x: float(beta * expit(beta * (x - c)) * (1 - expit(beta * (x - c))))
    return f, fp


FUNCS = {
    "log(x)":  (lambda x: np.log(x), lambda x: 1.0 / x),
    "sqrt(x)": (lambda x: np.sqrt(x), lambda x: 0.5 / np.sqrt(x)),
}
step_f, step_fp = make_step(1000.0, c=0.0)
FUNCS["1[x>0] (step, beta=1000)"] = (step_f, step_fp)


for label, (f, fp) in FUNCS.items():
    print(f"\n########## f = {label} ##########")
    for name, B in BUDGETS.items():
        X = find_nash(B, f, W, FLOOR)
        fprime = np.array([[fp(X[i, k]) for k in range(K)] for i in range(N)])
        ratio, unit = through_origin_ratio(fprime)
        print(f"-- {name}: B={B}")
        print(f"   X=\n{np.round(X, 5)}")
        print(f"   f'(x)=\n{fprime}")
        if ratio is None:
            print(f"   f prime vectors are ~0 (degenerate / below float precision) -- cannot form a direction")
        else:
            print(f"   f' unit vectors:\n{np.round(unit, 6)}")
            print(f"   through-origin ratio sigma2/sigma1 = {ratio:.4e}")
