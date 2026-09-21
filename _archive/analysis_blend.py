"""
Linear blend between current and product objectives:

  U_i(alpha) = (1-alpha) * sum_{i'!=i} sum_k w_k sigma(f_{i,k} - f_{i',k})
             +     alpha  * sum_k w_k  prod_{i'!=i} sigma(f_{i,k} - f_{i',k})

alpha=0 -> current (proportional Nash)
alpha=1 -> product (specialised Nash)

Sweep alpha in [0, 1] and track: Nash allocations, SW, n_corners.
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

def obj_blend(x_i, i, X, alpha):
    th_i = f_v(x_i); th = f_v(X)
    others = [ip for ip in range(N) if ip != i]
    # current term
    cur = sum(w[k] * float(expit(th_i[k] - th[ip, k]))
              for ip in others for k in range(K))
    # product term
    prod_term = 0.0
    for k in range(K):
        p = 1.0
        for ip in others:
            p *= float(expit(th_i[k] - th[ip, k]))
        prod_term += w[k] * p
    return (1 - alpha) * cur + alpha * prod_term

def best_response(i, X, alpha):
    Bi = B[i]
    x0s = [X[i].copy(), np.array([Bi*w[0], Bi*w[1]]),
            np.array([Bi*0.97, Bi*0.03]), np.array([Bi*0.03, Bi*0.97])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -obj_blend(x, i, X, alpha), x0,
                       method="SLSQP", bounds=[(0, Bi)]*K,
                       constraints=[{"type": "ineq", "fun": lambda x: Bi - x.sum()}],
                       options={"ftol": 1e-9, "maxiter": 300})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(alpha, max_iter=60, tol=1e-5):
    X = np.column_stack([B*w[0], B*w[1]])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N): X[i] = best_response(i, X, alpha)
        if np.max(np.abs(X - X_old)) < tol: break
    return X

def sw(X):
    return float(np.dot(w, np.max(f_v(X), axis=0)))

ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

X_prop  = np.column_stack([B*w[0], B*w[1]])
sw_prop = sw(X_prop)

print(f"{'alpha':>7}  {'SW':>8}  {'n_corners/20':>13}  {'share_0 range':>20}")
print("-" * 58)

results = []
for alpha in ALPHAS:
    X = find_nash(alpha)
    shares = X[:, 0] / B
    n_corn = int(np.sum((shares < 0.05) | (shares > 0.95)))
    print(f"  {alpha:>5.2f}  {sw(X):>8.4f}  {n_corn:>13d}  "
          f"  [{shares.min():.2f}, {shares.max():.2f}]")
    results.append({"alpha": alpha, "X": X, "sw": sw(X),
                    "shares": shares, "n_corn": n_corn})

# ---------------------------------------------------------------------------
# Plot 1: SW and n_corners vs alpha
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(ALPHAS, [r["sw"] for r in results], "o-", color="steelblue", lw=2, ms=7)
ax.axhline(sw_prop, ls="--", color="black", lw=1, label=f"proportional SW={sw_prop:.3f}")
ax.set_xlabel(r"Blend $\alpha$  (0=current, 1=product)", fontsize=10)
ax.set_ylabel("Social welfare", fontsize=10)
ax.set_title("SW vs blend parameter", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(ALPHAS, [r["n_corn"] for r in results], "o-", color="darkorange", lw=2, ms=7)
ax.set_xlabel(r"Blend $\alpha$", fontsize=10)
ax.set_ylabel("Firms at corner (specialised)", fontsize=10)
ax.set_title("Specialisation onset vs $\\alpha$", fontsize=10, fontweight="bold")
ax.set_ylim(-0.5, N + 0.5)

plt.suptitle(
    r"$(1-\alpha)\cdot\mathrm{current} + \alpha\cdot\mathrm{product}$ objective  |  "
    f"$N={N}$, $K=2$, $f(x)=\\log x$",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("plot_blend_summary.pdf", format="pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: scatter at key alpha values
# ---------------------------------------------------------------------------

key = [r for r in results if r["alpha"] in {0.0, 0.1, 0.4, 1.0}]
colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle(
    r"Nash allocations at selected $\alpha$ values  |  "
    "circles=Nash, stars=proportional, dashed=proportional ray",
    fontsize=11, fontweight="bold"
)
for ax, res in zip(axes, key):
    X = res["X"]
    ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.4, alpha=0.15)
    for i in range(N):
        ax.scatter(X[i,0], X[i,1], color=colors[i], s=60, marker="o", zorder=5)
        ax.scatter(X_prop[i,0], X_prop[i,1], color=colors[i], s=130,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.3, alpha=0.5)
    ax.set_xlabel("$x_{i,1}$", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("$x_{i,2}$", fontsize=9)
    ax.set_title(f"$\\alpha={res['alpha']}$\nSW={res['sw']:.3f}  "
                 f"{res['n_corn']}/20 specialised", fontsize=9, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("plot_blend_scatter.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("\nSaved: plot_blend_summary.pdf, plot_blend_scatter.pdf")
