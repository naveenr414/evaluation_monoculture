"""
Fix requested by user:
 - enforce x_{i,k} >= 1 in every dimension (avoids log(x) < 0 / log(x)->-inf
   as x->0, which was making raw theta cross zero and confusing the
   "line through the origin" read)
 - use much larger budget disparity
 - test whether theta_i = f(x_i) lies on a line THROUGH THE ORIGIN across
   firms i (i.e. is every theta_i a positive scalar multiple of a single
   common direction vector) -- this is the real homogenization definition,
   not just affine-collinearity.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

N, K = 3, 2
W = np.array([0.7, 0.3])
EPS = 1e-6
L = 1.0  # minimum investment per dimension

# Much larger disparity, all firms comfortably feasible with x_k>=1 each (need B_i > K*L = 2)
BUDGETS = {
    "Wide (10x)":     [200.0, 60.0, 20.0],
    "Wide (100x)":    [2000.0, 60.0, 20.0],
    "Wide (1000x)":   [20000.0, 60.0, 20.1],
    "Extreme wide":   [50000.0, 500.0, 3.0],
}


def firm_objective(x_i, i, X, f, N, K, w):
    th_i = np.array([f(x_i[k]) for k in range(K)])
    th = np.array([[f(X[j, k]) for k in range(K)] for j in range(N)])
    total = 0.0
    for j in range(N):
        if j != i:
            total += float(np.sum(w * expit(th_i - th[j])))
    return total


def best_response(i, X, B, f, N, K, w, L):
    Bi = B[i]
    bounds = [(L, Bi - L * (K - 1))] * K
    constraints = [{"type": "ineq", "fun": lambda x: Bi - np.sum(x)}]
    x0s = [X[i].copy(), np.full(K, Bi / K)]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -firm_objective(x, i, X, f, N, K, w), x0,
                        method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"ftol": 1e-12, "maxiter": 500})
        if res.success and -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, L, None)
    return best_x


def find_nash(B, f, N, K, w, L, max_iter=300, tol=1e-10):
    X = np.array([np.full(K, Bi / K) for Bi in B])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, B, f, N, K, w, L)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X


def line_through_origin_ratio(theta):
    """
    Check if all theta_i are (positive) scalar multiples of one direction.
    Normalize each row, then look at spread of the normalized directions
    (0 = all point in exactly the same direction from the origin).
    """
    norms = np.linalg.norm(theta, axis=1, keepdims=True)
    unit = theta / norms
    # sigma2/sigma1 of the (uncentered!) unit vectors -- if all unit vectors
    # are identical, this matrix has rank 1 without even centering.
    s = np.linalg.svd(unit, compute_uv=False)
    return float(s[1] / s[0]) if len(s) > 1 and s[0] > 1e-12 else 0.0, unit


f_log = lambda x: np.log(x)


for name, B in BUDGETS.items():
    X = find_nash(B, f_log, N, K, W, L)
    theta = np.array([[f_log(X[i, k]) for k in range(K)] for i in range(N)])
    ratio, unit = line_through_origin_ratio(theta)
    print(f"\n-- {name}: B={B}")
    print(f"   X=\n{np.round(X, 5)}")
    print(f"   theta=\n{np.round(theta, 5)}")
    print(f"   theta unit vectors (theta_i / ||theta_i||):\n{np.round(unit, 6)}")
    print(f"   through-origin collinearity ratio sigma2/sigma1 = {ratio:.3e}")
