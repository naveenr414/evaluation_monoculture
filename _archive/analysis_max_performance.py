"""
Compare Nash allocations under three evaluation objectives:
  (A) Current:      sum_{i'≠i} sum_k w_k σ(f_{i,k} - f_{i',k})
  (B) Beat-best:    sum_k w_k σ(f_{i,k} - max_{i'≠i} f_{i',k})
  (C) Product:      sum_k w_k prod_{i'≠i} σ(f_{i,k} - f_{i',k})

K=2, f=log(x), w=[1/3, 2/3], N=4 firms.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize

N, K = 20, 2
B    = np.linspace(5.0, 1.0, N)
w    = np.array([1/3, 2/3])

f_v  = np.vectorize(lambda x: math.log(max(x, 1e-12)))

print(f"B = {np.round(B, 2)},  w = {np.round(w, 3)}\n")

# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def obj_current(x_i, i, X):
    th_i = f_v(x_i); th = f_v(X)
    return sum(w[k] * float(expit(th_i[k] - th[ip, k]))
               for ip in range(N) if ip != i
               for k in range(K))

def obj_beat_best(x_i, i, X):
    th_i = f_v(x_i); th = f_v(X)
    others = [ip for ip in range(N) if ip != i]
    total = 0.0
    for k in range(K):
        best_other = max(th[ip, k] for ip in others)
        total += w[k] * float(expit(th_i[k] - best_other))
    return total

def obj_product(x_i, i, X):
    th_i = f_v(x_i); th = f_v(X)
    total = 0.0
    for k in range(K):
        p = 1.0
        for ip in range(N):
            if ip != i:
                p *= float(expit(th_i[k] - th[ip, k]))
        total += w[k] * p
    return total

OBJECTIVES = [
    ("Current\n$\\sum_{i'} \\sum_k w_k \\sigma(f_{i,k}{-}f_{i',k})$", obj_current),
    ("Beat-best\n$\\sum_k w_k \\sigma(f_{i,k}{-}\\max_{i'} f_{i',k})$", obj_beat_best),
    ("Product\n$\\sum_k w_k \\prod_{i'} \\sigma(f_{i,k}{-}f_{i',k})$", obj_product),
]

# ---------------------------------------------------------------------------
# Nash solver
# ---------------------------------------------------------------------------

def best_response(i, X, obj_fn, n_restarts=3):
    Bi = B[i]
    bounds = [(0, Bi)] * K
    cons   = [{"type": "ineq", "fun": lambda x: Bi - x.sum()}]
    x0s    = [X[i].copy(),
               np.array([Bi * w[0], Bi * w[1]]),
               np.array([Bi * 0.9, Bi * 0.1]),
               np.array([Bi * 0.1, Bi * 0.9])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s[:n_restarts+1]:
        res = minimize(lambda x: -obj_fn(x, i, X), x0,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 300})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(obj_fn, max_iter=60, tol=1e-5):
    X = np.column_stack([B * w[0], B * w[1]])   # start at proportional
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, obj_fn)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X

def sw(X):
    return float(np.dot(w, np.max(f_v(X), axis=0)))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

X_prop  = np.column_stack([B * w[0], B * w[1]])
sw_prop = sw(X_prop)

results = []
for name, obj_fn in OBJECTIVES:
    X_nash = find_nash(obj_fn)
    shares = X_nash[:, 0] / B
    print(f"{name.replace(chr(10), ' ')}")
    for i in range(N):
        print(f"  firm {i+1} (B={B[i]:.2f}): ({X_nash[i,0]:.3f}, {X_nash[i,1]:.3f})"
              f"  share_0={shares[i]:.3f}  [prop={w[0]:.3f}]")
    print(f"  SW={sw(X_nash):.4f}  (prop={sw_prop:.4f})\n")
    results.append({"name": name, "X": X_nash, "sw": sw(X_nash), "shares": shares})

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Nash allocations: current vs max-performance objectives  |  "
    f"$K=2$,  $w=[1/3,\\,2/3]$,  $N={N}$,  $f(x)=\\log x$\n"
    "(circles = Nash; stars = proportional benchmark; dashed = proportional ray)",
    fontsize=11, fontweight="bold"
)

for ax, res in zip(axes, results):
    X = res["X"]
    ax.plot(w[0] * t, w[1] * t, "k--", lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.5, alpha=0.2)
    for i in range(N):
        ax.scatter(X[i, 0], X[i, 1], color=colors[i], s=100, marker="o", zorder=5)
        ax.scatter(X_prop[i, 0], X_prop[i, 1], color=colors[i], s=220,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.4)
        if i < 5 or i == N-1:   # label only top firms + smallest
            ax.annotate(f"B={B[i]:.1f}", (X[i, 0], X[i, 1]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=colors[i])
    ax.set_xlabel("$x_{i,1}$  (dim 0)", fontsize=10)
    ax.set_ylabel("$x_{i,2}$  (dim 1)", fontsize=10) if ax is axes[0] else None
    ax.set_title(f"{res['name']}\nSW={res['sw']:.4f}  (prop={sw_prop:.4f})",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

axes[0].scatter([], [], color="gray", s=100, marker="o", label="Nash")
axes[0].scatter([], [], color="gray", s=220, marker="*",
                edgecolors="k", linewidths=0.4, label="Proportional")
axes[0].legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_max_performance.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("Saved: plot_max_performance.pdf")
