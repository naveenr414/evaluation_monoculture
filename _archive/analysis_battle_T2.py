"""
T is a row-stochastic matrix: each row sums to 1 (each firm must battle someone).
For N=3:
  Row 0: [0, T01, 1-T01]      free param: T01 in [0,1]
  Row 1: [T10, 0, 1-T10]      free param: T10 in [0,1]
  Row 2: [T20, 1-T20, 0]      free param: T20 in [0,1]

Search space: 3D unit cube [eps, 1-eps]^3.
Objective for firm i: U_i = sum_{j!=i} T[i,j] * sum_k w_k * sigma(f(x_{i,k}) - f(x_{j,k}))

We evaluate at:
  (A) normal budget B=[5,3,1], alpha=0.5 and 0.75
  (B) extreme budgets, alpha=0.5 and 0.75

Plot: for each setting, 3 pairwise 2D heatmaps (marginalizing the third param)
plus the best T and uniform-T baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def T_from_params(t01, t10, t20):
    """Build full 3x3 row-stochastic T matrix."""
    return np.array([
        [0.0,      t01,      1.0-t01],
        [t10,      0.0,      1.0-t10],
        [t20,      1.0-t20,  0.0    ],
    ])


def best_response_T(i, X, B, w, f_fn, T):
    def neg_obj(xi):
        fxi = np.array([f_fn(xi[k]) for k in range(K)])
        total = 0.0
        for j in range(N):
            if j == i: continue
            fxj = np.array([f_fn(X[j, k]) for k in range(K)])
            total += T[i, j] * float(np.sum(w * expit(fxi - fxj)))
        return -total

    B_i = B[i]
    bounds = [(0.0, B_i)] * K
    constraints = [{"type": "ineq", "fun": lambda x: B_i - np.sum(x)}]
    res = minimize(neg_obj, X[i].copy(), method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"ftol": 1e-8, "maxiter": 200})
    return np.clip(res.x, 0, None) if res.success else X[i].copy()


def find_nash_T(B, alpha, T, max_iter=30, tol=1e-5):
    f_fn, _ = make_power(alpha)
    X = np.stack([np.full(K, B[i] / K) for i in range(N)])
    for _ in range(max_iter):
        X_prev = X.copy()
        for i in range(N):
            X[i] = best_response_T(i, X, B, W, f_fn, T)
        if np.max(np.abs(X - X_prev)) < tol:
            break
    return X


def compute_Sigma_T(X, alpha, T):
    f_fn, _ = make_power(alpha)
    fx = np.array([[f_fn(X[i, k]) for k in range(K)] for i in range(N)])
    Sigma = np.zeros((N, K))
    for i in range(N):
        for j in range(N):
            if j == i: continue
            diff = fx[i] - fx[j]
            Sigma[i] += T[i, j] * np.array([sigma_prime(d) for d in diff])
    return Sigma


def max_log_ror(Sigma):
    if np.any(np.abs(Sigma).sum(axis=1) < 1e-6):
        return np.nan
    pairs = [(0,1),(0,2),(1,2)]
    vals = []
    for i, j in pairs:
        ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
        rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
        vals.append(abs(np.log(max(ri / (rj + 1e-12), 1e-12))))
    return max(vals)


def eval_T_params(t01, t10, t20, B, alpha):
    T = T_from_params(t01, t10, t20)
    X = find_nash_T(B, alpha, T)
    Sigma = compute_Sigma_T(X, alpha, T)
    return X, Sigma, max_log_ror(Sigma)


# -----------------------------------------------------------------------
# Grid search: 3D cube [eps, 1-eps]^3
# -----------------------------------------------------------------------

EPS = 0.05
GRID_N = 10   # per axis => 10^3 = 1000 evals per setting (~10s total)

t_vals = np.linspace(EPS, 1 - EPS, GRID_N)

SETTINGS = [
    {"label": "Normal B=[5,3,1]",         "B": np.array([5.0, 3.0, 1.0]),    "alpha": 0.5},
    {"label": "Normal B=[5,3,1]",         "B": np.array([5.0, 3.0, 1.0]),    "alpha": 0.75},
    {"label": "Extreme B=[.01,.01,8.98]", "B": np.array([0.01, 0.01, 8.98]), "alpha": 0.5},
    {"label": "Extreme B=[4.43,4.33,.2]", "B": np.array([4.43, 4.33, 0.20]), "alpha": 0.75},
]

all_results = []

for setting in SETTINGS:
    B, alpha, label = setting["B"], setting["alpha"], setting["label"]
    print(f"\n{'='*60}\n{label},  alpha={alpha}\n{'='*60}")

    scores = np.full((GRID_N, GRID_N, GRID_N), np.nan)

    for ii, t01 in enumerate(t_vals):
        for jj, t10 in enumerate(t_vals):
            for kk, t20 in enumerate(t_vals):
                _, _, v = eval_T_params(t01, t10, t20, B, alpha)
                scores[ii, jj, kk] = v if v is not None else np.nan

    best_idx = np.unravel_index(np.nanargmax(scores), scores.shape)
    t01_best = t_vals[best_idx[0]]
    t10_best = t_vals[best_idx[1]]
    t20_best = t_vals[best_idx[2]]
    best_val = scores[best_idx]
    T_best = T_from_params(t01_best, t10_best, t20_best)

    # Uniform T baseline: each firm splits 50/50
    _, Sigma_unif, v_unif = eval_T_params(0.5, 0.5, 0.5, B, alpha)

    _, Sigma_best, _ = eval_T_params(t01_best, t10_best, t20_best, B, alpha)
    pairs = [(0,1),(0,2),(1,2)]

    print(f"  Best T:")
    print(f"    T01={t01_best:.3f} T02={1-t01_best:.3f}")
    print(f"    T10={t10_best:.3f} T12={1-t10_best:.3f}")
    print(f"    T20={t20_best:.3f} T21={1-t20_best:.3f}")
    print(f"  max|log R| = {best_val:.5f}  (uniform: {v_unif:.5f})")
    for i, j in pairs:
        ri = Sigma_best[i,0]/(Sigma_best[i,1]+1e-12)
        rj = Sigma_best[j,0]/(Sigma_best[j,1]+1e-12)
        print(f"    R({i},{j}) = {ri/rj:.5f}")

    all_results.append({
        "label": label, "alpha": alpha, "B": B,
        "scores": scores,
        "t_vals": t_vals,
        "best": (t01_best, t10_best, t20_best),
        "best_val": best_val,
        "v_unif": v_unif,
        "T_best": T_best,
        "Sigma_best": Sigma_best,
    })


# -----------------------------------------------------------------------
# Plot: 4 settings x 3 marginal heatmaps = 4x3 grid
# -----------------------------------------------------------------------

fig, axes = plt.subplots(4, 3, figsize=(13, 16))
fig.suptitle("Row-stochastic T: max|log R| over $(T_{01}, T_{10}, T_{20})$\n"
             "Heatmaps marginalize (max over) the third parameter",
             fontsize=12, fontweight="bold")

param_pairs = [
    (0, 1, 2, "T_{01}", "T_{10}"),
    (0, 2, 1, "T_{01}", "T_{20}"),
    (1, 2, 0, "T_{10}", "T_{20}"),
]

for row, res in enumerate(all_results):
    scores = res["scores"]   # shape (GRID_N, GRID_N, GRID_N)
    t_vals = res["t_vals"]
    t01b, t10b, t20b = res["best"]

    for col, (ax_i, ax_j, marg_k, xlabel, ylabel) in enumerate(param_pairs):
        ax = axes[row, col]
        # Marginalize (take max) over the third axis
        heatmap = np.nanmax(scores, axis=marg_k)
        # heatmap axes correspond to ax_i (rows) and ax_j (cols)
        # but we need to check the axis ordering matches (ii=t01, jj=t10, kk=t20)
        # axis 0=t01, axis 1=t10, axis 2=t20
        # marginalizing axis marg_k leaves axes ax_i and ax_j
        # imshow: rows=first remaining axis (y), cols=second (x)
        # We want x=xlabel param, y=ylabel param
        im = ax.imshow(heatmap.T, origin="lower", aspect="auto",
                       extent=[t_vals[0], t_vals[-1], t_vals[0], t_vals[-1]],
                       cmap="viridis")
        plt.colorbar(im, ax=ax, label="|log R|")

        # Best point (project onto these two axes)
        best_coords = [t01b, t10b, t20b]
        ax.scatter(best_coords[ax_i], best_coords[ax_j],
                   color="red", s=100, marker="*", zorder=5,
                   label=f"opt=({best_coords[ax_i]:.2f},{best_coords[ax_j]:.2f})")
        # Uniform point
        ax.scatter(0.5, 0.5, color="white", s=60, marker="o",
                   edgecolors="black", lw=1, zorder=5)

        ax.set_xlabel(f"${xlabel}$", fontsize=9)
        ax.set_ylabel(f"${ylabel}$", fontsize=9)
        if col == 0:
            ax.set_title(f"{res['label']}\nα={res['alpha']}  best={res['best_val']:.3f}",
                         fontsize=8, fontweight="bold")
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("plot_battle_T2.pdf", bbox_inches="tight")
plt.savefig("plot_battle_T2.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_battle_T2.pdf / .png")
