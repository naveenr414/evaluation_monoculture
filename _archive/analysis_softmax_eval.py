"""
Softmax scoring with temperature tau:

  U_i = sum_k w_k * exp(f_{i,k}/tau) / sum_j exp(f_{j,k}/tau)

  tau -> inf : uniform (1/N per category, no incentive)
  tau -> 0   : winner-takes-all per category

With f=log(x): exp(f/tau) = x^(1/tau), so firms compete on x^(1/tau) shares.

At proportional, softmax_i,k = B_i^(1/tau) / sum B_j^(1/tau) — k-independent,
so proportional is always a critical point. But for low tau, there may be
better Nash equilibria that are specialised.

We try multiple initialisations (proportional + corners) to find the best Nash.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import softmax as sp_softmax

N, K = 20, 2
B    = np.linspace(5.0, 1.0, N)
w    = np.array([1/3, 2/3])
f_v  = np.vectorize(lambda x: math.log(max(x, 1e-12)))

def obj_softmax(x_i, i, X, tau):
    # Replace row i with x_i
    X2 = X.copy(); X2[i] = x_i
    th  = f_v(X2)          # (N, K)
    score = 0.0
    for k in range(K):
        sm = sp_softmax(th[:, k] / tau)
        score += w[k] * sm[i]
    return score

def best_response(i, X, tau):
    Bi  = B[i]
    # Initialisations: proportional, all-dim0, all-dim1, midpoints
    x0s = [X[i].copy(),
            np.array([Bi*w[0], Bi*w[1]]),
            np.array([Bi*0.98, Bi*0.02]),
            np.array([Bi*0.02, Bi*0.98]),
            np.array([Bi*0.7,  Bi*0.3]),
            np.array([Bi*0.3,  Bi*0.7])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -obj_softmax(x, i, X, tau), x0,
                       method="SLSQP",
                       bounds=[(0, Bi)] * K,
                       constraints=[{"type": "ineq", "fun": lambda x: Bi - x.sum()}],
                       options={"ftol": 1e-9, "maxiter": 300})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(tau, max_iter=80, tol=1e-5):
    X = np.column_stack([B*w[0], B*w[1]])   # start at proportional
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, tau)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X

def sw(X):
    return float(np.dot(w, np.max(f_v(X), axis=0)))

TAU_VALS = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]

X_prop  = np.column_stack([B*w[0], B*w[1]])
sw_prop = sw(X_prop)

print(f"{'tau':>6}  {'SW':>8}  {'corners/20':>11}  share_0 range")
print("-" * 55)

results = []
for tau in TAU_VALS:
    X = find_nash(tau)
    shares = X[:, 0] / B
    n_corn = int(np.sum((shares < 0.05) | (shares > 0.95)))
    print(f"  {tau:>4.1f}  {sw(X):>8.4f}  {n_corn:>11d}  [{shares.min():.3f}, {shares.max():.3f}]")
    results.append({"tau": tau, "X": X, "sw": sw(X), "shares": shares, "n_corn": n_corn})

# ---------------------------------------------------------------------------
# Plot 1: scatter at each tau
# ---------------------------------------------------------------------------

colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
fig.suptitle(
    r"Nash allocations under softmax scoring $U_i = \sum_k w_k \cdot \mathrm{softmax}(f_{\cdot,k}/\tau)_i$"
    f"\n$N={N}$, $K=2$, $w=[1/3,\\,2/3]$, $f(x)=\\log x$  |  circles=Nash, stars=proportional",
    fontsize=11, fontweight="bold"
)
for ax, res in zip(axes, results):
    X = res["X"]
    ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.4, alpha=0.15)
    for i in range(N):
        ax.scatter(X[i,0], X[i,1], color=colors[i], s=60, marker="o", zorder=5)
        ax.scatter(X_prop[i,0], X_prop[i,1], color=colors[i], s=130,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.3, alpha=0.5)
    ax.set_xlabel("$x_{i,1}$", fontsize=9)
    if ax in (axes[0], axes[3]): ax.set_ylabel("$x_{i,2}$", fontsize=9)
    ax.set_title(f"$\\tau={res['tau']}$\nSW={res['sw']:.4f}   {res['n_corn']}/20 specialised",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
axes[0].scatter([], [], color="gray", s=60,  marker="o", label="Nash")
axes[0].scatter([], [], color="gray", s=130, marker="*", edgecolors="k",
                linewidths=0.3, alpha=0.5, label="Proportional")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("plot_softmax_scatter.pdf", format="pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: dim-0 share vs budget rank and SW vs tau
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(TAU_VALS)))

ax = axes2[0]
for res, c in zip(results, cmap):
    ax.plot(B, res["shares"], "o-", color=c, lw=1.5, ms=5,
            label=f"$\\tau={res['tau']}$")
ax.axhline(w[0], ls="--", color="black", lw=1, label=f"proportional (w₀={w[0]:.2f})")
ax.set_xlabel("Budget $B_i$", fontsize=10)
ax.set_ylabel("Dim-0 share  $x_{i,1}/B_i$", fontsize=10)
ax.set_title("Specialisation by budget as $\\tau$ falls", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, ncol=2)

ax = axes2[1]
ax.plot(TAU_VALS, [r["sw"] for r in results], "o-", color="steelblue", lw=2, ms=8)
ax.axhline(sw_prop, ls="--", color="black", lw=1, label=f"proportional SW={sw_prop:.3f}")
ax.set_xlabel("Temperature $\\tau$", fontsize=10)
ax.set_ylabel("Social welfare", fontsize=10)
ax.set_title("SW vs $\\tau$", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

plt.suptitle("Softmax scoring: specialisation onset", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_softmax_sw.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("\nSaved: plot_softmax_scatter.pdf, plot_softmax_sw.pdf")
