"""
K=2 allocation comparison across sigma variants (fast, self-contained).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize

N, K = 4, 2
EPS  = 1e-8

rng = np.random.default_rng(42)
B   = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
w   = np.array([1/3, 2/3])

f_v  = np.vectorize(lambda x: math.log(max(x, 1e-12)))
fp_v = np.vectorize(lambda x: 1.0 / max(x, 1e-12))

# ---------------------------------------------------------------------------
# Win function families — operate on scalar floats
# ---------------------------------------------------------------------------

def make_absolute():
    win  = lambda a, b: expit(a - b)
    winp = lambda a, b: expit(a-b) * (1-expit(a-b))   # d/da
    return win, winp, "absolute\n$\\sigma(f_i{-}f_j)$"

def make_rel_mean():
    def win(a, b):
        d = (abs(a)+abs(b))/2 + EPS; return expit((a-b)/d)
    def winp(a, b):
        d = (abs(a)+abs(b))/2 + EPS; z = (a-b)/d
        sa = 1. if a >= 0 else -1.
        dz = (d - (a-b)*(sa/2)) / d**2
        v = expit(z); return v*(1-v)*dz
    return win, winp, "rel/mean\n$\\sigma(\\Delta/\\bar{|f|})$"

def make_rel_opp():
    def win(a, b):
        d = max(abs(b), EPS); return expit((a-b)/d)
    def winp(a, b):
        d = max(abs(b), EPS); z = (a-b)/d
        v = expit(z); return v*(1-v)/d
    return win, winp, "rel/opp\n$\\sigma(\\Delta/|f_j|)$"

VARIANTS = [make_absolute(), make_rel_mean(), make_rel_opp()]

# ---------------------------------------------------------------------------
# Fast best-response using vectorised objective
# ---------------------------------------------------------------------------

def firm_obj(x_i, i, X, win_fn):
    th_i = f_v(x_i)          # shape (K,)
    th   = f_v(X)             # shape (N, K)
    total = 0.0
    for ip in range(N):
        if ip != i:
            for k in range(K):
                total += w[k] * win_fn(th_i[k], th[ip, k])
    return total

def best_response(i, X, win_fn, n_restarts=2):
    Bi = B[i]
    bounds = [(0, Bi)] * K
    cons   = [{"type": "ineq", "fun": lambda x: Bi - x.sum()}]
    best_x, best_v = X[i].copy(), -np.inf
    x0s = [X[i].copy(), np.array([Bi*w[0], Bi*w[1]])]
    for x0 in x0s:
        res = minimize(lambda x: -firm_obj(x, i, X, win_fn), x0,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-8, "maxiter": 200})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(win_fn, max_iter=40, tol=1e-5):
    X = np.column_stack([B * w[0], B * w[1]])   # start at proportional
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, win_fn)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X

def social_welfare(X):
    th = f_v(X)
    return float(np.dot(w, np.max(th, axis=0)))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"B = {np.round(B,2)},  w = {np.round(w,3)}\n")

X_prop = np.column_stack([B * w[0], B * w[1]])
sw_prop = social_welfare(X_prop)
print(f"Proportional SW = {sw_prop:.4f}  (Nash = proportional for absolute)\n")

results = []
for win_fn, winp_fn, name in VARIANTS:
    X_nash = find_nash(win_fn)
    sw     = social_welfare(X_nash)
    results.append({"name": name, "X": X_nash, "sw": sw})
    share0 = X_nash[:, 0] / B
    label  = name.replace("\n", " ")
    print(f"{label}")
    for i in range(N):
        print(f"  firm {i+1} (B={B[i]:.2f}): ({X_nash[i,0]:.3f}, {X_nash[i,1]:.3f})"
              f"  share_0={share0[i]:.3f}  [prop={w[0]:.3f}]")
    print(f"  SW={sw:.4f}  vs prop {sw_prop:.4f}\n")

# ---------------------------------------------------------------------------
# Plot — K=2 scatter (style of plot_poa_scatter.pdf)
# ---------------------------------------------------------------------------

colors = plt.cm.plasma(np.linspace(0.1, 0.85, N))
t = np.linspace(0, B.max(), 200)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(
    f"Nash allocations by $\\sigma$ variant  |  $K=2$,  "
    f"$w$=[{w[0]:.2f}, {w[1]:.2f}],  $N={N}$ firms\n"
    "(dashed = proportional ray; faint lines = budget constraints)",
    fontsize=11, fontweight="bold"
)

for ax, res in zip(axes, results):
    X = res["X"]
    # proportional ray and budget constraint lines
    ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.5, alpha=0.25)
    # Nash points per firm
    for i in range(N):
        ax.scatter(X[i, 0], X[i, 1], color=colors[i], s=90, marker="o", zorder=5)
        # proportional benchmark as star
        ax.scatter(X_prop[i, 0], X_prop[i, 1], color=colors[i], s=200,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.4)
        ax.annotate(f"B={B[i]:.1f}", (X[i, 0], X[i, 1]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color=colors[i])
    ax.set_xlabel(f"$x_{{i,1}}$  ($w_1={w[0]:.2f}$)", fontsize=9)
    ax.set_ylabel(f"$x_{{i,2}}$  ($w_2={w[1]:.2f}$)", fontsize=9) if ax is axes[0] else None
    ax.set_title(f"{res['name']}\nSW={res['sw']:.4f}  (prop={sw_prop:.4f})", fontsize=9, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

axes[0].scatter([], [], color="gray", s=90,  marker="o", label="Nash")
axes[0].scatter([], [], color="gray", s=200, marker="*", edgecolors="k",
                linewidths=0.4, label="Proportional")
axes[0].legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_relative_k2.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("Saved: plot_relative_k2.pdf")
