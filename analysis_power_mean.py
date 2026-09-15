"""
Continuous interpolation between current (p=1) and product (p->0) objectives
via the power mean of per-opponent win-rates:

  score_{i,k}(p) = (1/(N-1) * sum_{i'!=i} sigma(f_{i,k} - f_{i',k})^p)^(1/p)

  p =  1   : arithmetic mean  = current objective (up to 1/(N-1) scaling)
  p -> 0   : geometric mean   ~ product objective (same Nash)
  p = -1   : harmonic mean    (heavily penalises weak matchups)
  p -> -inf: min win-rate     (only worst matchup matters)

Objective: sum_k w_k * score_{i,k}(p)

We sweep p in {1, 0.5, 0.1, -0.5, -1, -2} and plot Nash allocations.
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

EPS = 1e-12

def power_mean_obj(x_i, i, X, p):
    th_i = f_v(x_i); th = f_v(X)
    others = [ip for ip in range(N) if ip != i]
    n = len(others)
    total = 0.0
    for k in range(K):
        wins = np.array([float(expit(th_i[k] - th[ip, k])) for ip in others])
        wins = np.clip(wins, EPS, 1 - EPS)
        if abs(p) < 1e-3:   # geometric mean (p -> 0 limit)
            score = math.exp(np.mean(np.log(wins)))
        else:
            score = (np.mean(wins ** p)) ** (1.0 / p)
        total += w[k] * score
    return total

def best_response(i, X, p, n_restarts=3):
    Bi = B[i]
    x0s = [X[i].copy(),
            np.array([Bi*w[0], Bi*w[1]]),
            np.array([Bi*0.95, Bi*0.05]),
            np.array([Bi*0.05, Bi*0.95])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s[:n_restarts+1]:
        res = minimize(lambda x: -power_mean_obj(x, i, X, p), x0,
                       method="SLSQP",
                       bounds=[(0, Bi)]*K,
                       constraints=[{"type": "ineq", "fun": lambda x: Bi - x.sum()}],
                       options={"ftol": 1e-9, "maxiter": 300})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(p, max_iter=60, tol=1e-5):
    X = np.column_stack([B*w[0], B*w[1]])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, p)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X

def sw(X):
    return float(np.dot(w, np.max(f_v(X), axis=0)))

P_VALS = [1.0, 0.5, 0.1, -0.5, -1.0, -2.0]
P_LABELS = ["$p=1$\n(arithmetic/current)", "$p=0.5$", "$p\\to 0$\n(geometric/product)",
            "$p=-0.5$", "$p=-1$\n(harmonic)", "$p=-2$"]

X_prop  = np.column_stack([B*w[0], B*w[1]])
sw_prop = sw(X_prop)

print(f"{'p':>6}  {'SW':>8}  {'n_corners':>10}  (prop SW={sw_prop:.4f})")
print("-" * 40)

results = []
for p, label in zip(P_VALS, P_LABELS):
    X = find_nash(p)
    shares = X[:, 0] / B
    n_corners = int(np.sum((shares < 0.05) | (shares > 0.95)))
    print(f"{p:>6.1f}  {sw(X):>8.4f}  {n_corners:>10d}/20")
    results.append({"p": p, "label": label, "X": X, "sw": sw(X), "shares": shares})

# ---------------------------------------------------------------------------
# Plot 1: K=2 scatter grid
# ---------------------------------------------------------------------------

colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
fig.suptitle(
    f"Nash allocations under power-mean objective  |  "
    f"$K=2$, $w=[1/3,\\,2/3]$, $N={N}$ firms, $f(x)=\\log x$\n"
    "circles = Nash;  stars = proportional;  dashed = proportional ray",
    fontsize=11, fontweight="bold"
)

for ax, res in zip(axes, results):
    X = res["X"]
    ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.4, alpha=0.15)
    for i in range(N):
        ax.scatter(X[i,0], X[i,1], color=colors[i], s=60, marker="o", zorder=5)
        ax.scatter(X_prop[i,0], X_prop[i,1], color=colors[i], s=140,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.3, alpha=0.5)
    ax.set_xlabel("$x_{i,1}$", fontsize=9)
    ax.set_ylabel("$x_{i,2}$", fontsize=9) if ax in axes[::3] else None
    ax.set_title(f"{res['label']}\nSW={res['sw']:.4f}  ({res['shares'].min():.2f}–{res['shares'].max():.2f} dim-0 share)",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

axes[0].scatter([], [], color="gray", s=60,  marker="o", label="Nash")
axes[0].scatter([], [], color="gray", s=140, marker="*", edgecolors="k",
                linewidths=0.3, alpha=0.5, label="Proportional")
axes[0].legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_power_mean_scatter.pdf", format="pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: dim-0 share per firm vs budget rank, one line per p
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(P_VALS)))

ax = axes2[0]
for res, c in zip(results, cmap):
    ax.plot(B, res["shares"], "o-", color=c, lw=1.5, ms=5, label=res["label"].replace("\n"," "))
ax.axhline(w[0], ls="--", color="black", lw=1, label=f"proportional (w₀={w[0]:.2f})")
ax.set_xlabel("Budget $B_i$", fontsize=10)
ax.set_ylabel("Dim-0 share  $x_{i,1}/B_i$", fontsize=10)
ax.set_title("Specialization by budget: dim-0 share", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, ncol=2)

ax = axes2[1]
sw_vals = [r["sw"] for r in results]
ax.plot(P_VALS, sw_vals, "o-", color="steelblue", lw=2, ms=8)
ax.axhline(sw_prop, ls="--", color="black", lw=1, label=f"proportional SW={sw_prop:.3f}")
ax.set_xlabel("Power mean exponent $p$", fontsize=10)
ax.set_ylabel("Social welfare (SW)", fontsize=10)
ax.set_title("SW vs $p$: moving from current to product", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_power_mean_sw.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("\nSaved: plot_power_mean_scatter.pdf, plot_power_mean_sw.pdf")
