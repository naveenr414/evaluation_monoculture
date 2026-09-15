"""
Introduce a symmetric battle-rate matrix T over pairs.
For N=3, T = (t01, t02, t12) with t01+t02+t12=1 (simplex).

Firm i's objective:
  U_i = sum_{j≠i} T_{ij} * sum_k w_k * sigma(f(x_{i,k}) - f(x_{j,k}))

Sigma_{i,k} = sum_{j≠i} T_{ij} * sigma'(f(x_{i,k}) - f(x_{j,k}))

We grid-search the (t01, t02, t12) simplex at two budget settings:
  (A) normal:  B = [5, 3, 1]
  (B) extreme: B = [0.01, 0.01, 8.98]  (alpha<=0.6 optimal)
               B = [4.425, 4.325, 0.2] (alpha=0.75 optimal)

For each setting, find T that maximizes max|log R| over all firm pairs.
Also plot the heatmap over the simplex.

We use alpha = 0.5 for the normal-budget case and show all four alphas
for the extreme budget.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.special import expit
from scipy.optimize import minimize

N, K = 3, 2
W = np.array([0.7, 0.3])


def make_power(alpha):
    f  = lambda x: max(x, 0.0) ** alpha
    fp = lambda x: alpha * max(x, 1e-12) ** (alpha - 1)
    return f, fp


def sigma_prime(z):
    s = expit(z); return s * (1.0 - s)


# -----------------------------------------------------------------------
# Nash equilibrium with T-weighted objectives
# -----------------------------------------------------------------------

def best_response_T(i, X, B, w, f_fn, T_edge):
    """
    T_edge = [t01, t02, t12]: symmetric pair weights summing to 1.
    T_{ij} for firm i, opponent j:
      firm 0: T_{01}=t01, T_{02}=t02
      firm 1: T_{10}=t01, T_{12}=t12
      firm 2: T_{20}=t02, T_{21}=t12
    """
    t01, t02, t12 = T_edge
    T_row = [
        [0, t01, t02],
        [t01, 0, t12],
        [t02, t12, 0],
    ]

    def neg_obj(xi):
        fxi = np.array([f_fn(xi[k]) for k in range(K)])
        total = 0.0
        for j in range(N):
            if j == i:
                continue
            fxj = np.array([f_fn(X[j, k]) for k in range(K)])
            total += T_row[i][j] * float(np.sum(w * expit(fxi - fxj)))
        return -total

    B_i = B[i]
    bounds = [(0.0, B_i)] * K
    constraints = [{"type": "ineq", "fun": lambda x: B_i - np.sum(x)}]

    rng = np.random.default_rng(i)
    candidates = [X[i].copy()]
    for _ in range(3):
        a = rng.dirichlet(np.ones(K))
        candidates.append(a * B_i)

    best_x, best_v = X[i].copy(), np.inf
    for x0 in candidates:
        res = minimize(neg_obj, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"ftol": 1e-10, "maxiter": 500})
        if res.success and res.fun < best_v:
            best_v, best_x = res.fun, np.clip(res.x, 0, None)
    return best_x


def find_nash_T(B, alpha, T_edge, max_iter=150, tol=1e-6):
    f_fn, _ = make_power(alpha)
    X = np.stack([np.full(K, B[i] / K) for i in range(N)])
    for _ in range(max_iter):
        X_prev = X.copy()
        for i in range(N):
            X[i] = best_response_T(i, X, B, W, f_fn, T_edge)
        if np.max(np.abs(X - X_prev)) < tol:
            break
    return X


def compute_Sigma_T(X, alpha, T_edge):
    f_fn, _ = make_power(alpha)
    t01, t02, t12 = T_edge
    T_row = [
        [0, t01, t02],
        [t01, 0, t12],
        [t02, t12, 0],
    ]
    fx = np.array([[f_fn(X[i, k]) for k in range(K)] for i in range(N)])
    Sigma = np.zeros((N, K))
    for i in range(N):
        for j in range(N):
            if j == i: continue
            diff = fx[i] - fx[j]
            Sigma[i] += T_row[i][j] * np.array([sigma_prime(d) for d in diff])
    return Sigma


def max_log_ror(Sigma):
    pairs = [(0, 1), (0, 2), (1, 2)]
    vals = []
    for i, j in pairs:
        ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
        rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
        vals.append(abs(np.log(max(ri / (rj + 1e-12), 1e-12))))
    return max(vals)


def eval_T(B, alpha, T_edge):
    X = find_nash_T(B, alpha, T_edge)
    Sigma = compute_Sigma_T(X, alpha, T_edge)
    return X, Sigma, max_log_ror(Sigma)


# -----------------------------------------------------------------------
# Simplex grid
# -----------------------------------------------------------------------

def simplex_grid(n=25, t_min=0.05):
    """
    Returns (t01, t02, t12) triples on a uniform simplex grid,
    restricted to the interior where all entries >= t_min.
    This ensures every firm battles every other firm at least t_min
    of the time, avoiding degenerate Sigma=0 cases.
    """
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            t = np.array([i / n, j / n, k / n])
            if np.all(t >= t_min):
                pts.append(tuple(t))
    return np.array(pts)


def is_valid_Sigma(Sigma, tol=1e-6):
    """Check that no firm's Sigma row is near-zero (degenerate)."""
    return np.all(np.abs(Sigma).sum(axis=1) > tol)


GRID_N = 25
T_pts = simplex_grid(GRID_N)

# -----------------------------------------------------------------------
# Settings to evaluate
# -----------------------------------------------------------------------

SETTINGS = [
    {"label": "Normal B=[5,3,1]",         "B": np.array([5.0, 3.0, 1.0]),    "alpha": 0.5},
    {"label": "Normal B=[5,3,1]",         "B": np.array([5.0, 3.0, 1.0]),    "alpha": 0.75},
    {"label": "Extreme B=[.01,.01,8.98]", "B": np.array([0.01, 0.01, 8.98]), "alpha": 0.5},
    {"label": "Extreme B=[4.43,4.33,.2]", "B": np.array([4.43, 4.33, 0.2]),  "alpha": 0.75},
]

all_results = []

for setting in SETTINGS:
    B, alpha, label = setting["B"], setting["alpha"], setting["label"]
    print(f"\n{'='*60}")
    print(f"{label},  alpha={alpha}")
    print("="*60)

    scores = []
    for T_edge in T_pts:
        X, Sigma, v = eval_T(B, alpha, T_edge)
        if not is_valid_Sigma(Sigma):
            scores.append(np.nan)
        else:
            scores.append(v)
    scores = np.array(scores)

    best_idx = np.nanargmax(scores)
    T_best = T_pts[best_idx]
    X_best, Sigma_best, best_val = eval_T(B, alpha, T_best)

    print(f"  Best T=(t01={T_best[0]:.3f}, t02={T_best[1]:.3f}, t12={T_best[2]:.3f})")
    print(f"  max|log R| = {best_val:.5f}")

    pairs = [(0,1),(0,2),(1,2)]
    for i, j in pairs:
        ri = Sigma_best[i,0]/(Sigma_best[i,1]+1e-12)
        rj = Sigma_best[j,0]/(Sigma_best[j,1]+1e-12)
        ror = ri/(rj+1e-12)
        print(f"    R({i},{j}) = {ror:.5f}")

    # Uniform T baseline
    T_unif = np.array([1/3, 1/3, 1/3])
    _, Sigma_unif, v_unif = eval_T(B, alpha, T_unif)
    print(f"  Uniform T: max|log R| = {v_unif:.5f}")

    all_results.append({
        "label": label, "alpha": alpha, "B": B,
        "scores": scores, "T_pts": T_pts,
        "T_best": T_best, "best_val": best_val,
        "v_unif": v_unif,
    })


# -----------------------------------------------------------------------
# Plot: simplex heatmap for each setting
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(r"max$|\log R|$ over battle-rate simplex $(t_{01}, t_{02}, t_{12})$",
             fontsize=12, fontweight="bold")

for ax, res in zip(axes, all_results):
    T_pts_ = res["T_pts"]
    scores_ = res["scores"]

    # Barycentric coords: use (t01, t02) as 2D embedding (t12 = 1-t01-t02)
    # Map simplex to equilateral triangle
    # Vertices: t01=1 at (0,0), t02=1 at (1,0), t12=1 at (0.5, sqrt(3)/2)
    t01 = T_pts_[:, 0]
    t02 = T_pts_[:, 1]
    t12 = T_pts_[:, 2]

    # Cartesian from barycentric
    v0 = np.array([0.0, 0.0])       # t01=1: pair (0,1) only
    v1 = np.array([1.0, 0.0])       # t02=1: pair (0,2) only
    v2 = np.array([0.5, np.sqrt(3)/2])  # t12=1: pair (1,2) only
    xy = np.outer(t01, v0) + np.outer(t02, v1) + np.outer(t12, v2)
    x_cart, y_cart = xy[:, 0], xy[:, 1]

    triang = mtri.Triangulation(x_cart, y_cart)
    tpc = ax.tripcolor(triang, scores_, shading="gouraud", cmap="viridis")
    plt.colorbar(tpc, ax=ax, label="|log R|")

    # Best point
    T_b = res["T_best"]
    xy_best = T_b[0]*v0 + T_b[1]*v1 + T_b[2]*v2
    ax.scatter(*xy_best, color="red", s=120, marker="*", zorder=5,
               label=f"opt: ({T_b[0]:.2f},{T_b[1]:.2f},{T_b[2]:.2f})")

    # Uniform point
    xy_unif = (1/3)*v0 + (1/3)*v1 + (1/3)*v2
    ax.scatter(*xy_unif, color="white", s=60, marker="o", zorder=5,
               edgecolors="black", lw=1, label=f"uniform: {res['v_unif']:.3f}")

    # Triangle edges and vertex labels
    tri_verts = np.array([v0, v1, v2, v0])
    ax.plot(tri_verts[:,0], tri_verts[:,1], "k-", lw=1)
    offset = 0.06
    ax.text(v0[0]-offset, v0[1]-offset, "$t_{01}=1$\n(0v1 only)", ha="center", fontsize=7)
    ax.text(v1[0]+offset, v1[1]-offset, "$t_{02}=1$\n(0v2 only)", ha="center", fontsize=7)
    ax.text(v2[0], v2[1]+offset, "$t_{12}=1$\n(1v2 only)", ha="center", fontsize=7)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{res['label']}\nα={res['alpha']}  best={res['best_val']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig("plot_battle_T.pdf", bbox_inches="tight")
plt.savefig("plot_battle_T.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_battle_T.pdf / .png")
