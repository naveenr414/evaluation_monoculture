"""
k-way contest sweep: heterogeneity vs k, one line per tau, one plot per f.

Heterogeneity = std of dim-0 shares x_{i,1}/B_i across firms.
(At proportional: all shares = w[0], std=0. At fully specialised: max std.)

N=8, K=2, B=linspace(5,1,8), w=[1/3,2/3].
k in {2,3,4,6,8},  tau in {2.0, 1.0, 0.5, 0.2, 0.1}.
5 production functions.
"""

import math, numpy as np, matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import softmax as sp_softmax

N, K = 8, 2
B    = np.linspace(5.0, 1.0, N)
w    = np.array([1/3, 2/3])

F_SPECS = [
    (r"$\log(x)$",       lambda x: math.log(max(x, 1e-12))),
    (r"$\log(1+x)$",     lambda x: math.log(1 + max(x, 0))),
    (r"$\log(x+100)$",   lambda x: math.log(max(x, 0) + 100)),
    (r"$x^{0.75}$",      lambda x: max(x, 0)**0.75),
    (r"$\sqrt{x}$",      lambda x: math.sqrt(max(x, 0))),
    (r"$1 - e^{-x}$",    lambda x: 1 - math.exp(-max(x, 0))),
    (r"$1 - e^{-5x}$",   lambda x: 1 - math.exp(-5 * max(x, 0))),
]

K_VALS   = [2, 3, 4, 6, 8]
TAU_VALS = [2.0, 1.0, 0.5, 0.2, 0.1]

# Precompute subsets
SUBSETS = {(i, k): list(combinations([j for j in range(N) if j != i], k-1))
           for k in K_VALS for i in range(N)}

fv = np.vectorize

def obj_kway(x_i, i, X, f_fn, k, tau):
    X2 = X.copy(); X2[i] = x_i
    th  = fv(f_fn)(X2)
    subs = SUBSETS[(i, k)]
    total = 0.0
    for d in range(K):
        s = 0.0
        for S in subs:
            idx = [i] + list(S)
            s += sp_softmax(th[idx, d] / tau)[0]
        total += w[d] * s / len(subs)
    return total

def best_response(i, X, f_fn, k, tau):
    Bi = B[i]
    # Warm start: current + proportional + two corners
    x0s = [X[i].copy(), np.array([Bi*w[0], Bi*w[1]]),
            np.array([Bi*.96, Bi*.04]), np.array([Bi*.04, Bi*.96])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -obj_kway(x, i, X, f_fn, k, tau), x0,
                       method="SLSQP", bounds=[(0, Bi)]*K,
                       constraints=[{"type":"ineq","fun":lambda x: Bi - x.sum()}],
                       options={"ftol":1e-8, "maxiter":150})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(f_fn, k, tau, X_init=None, max_iter=40, tol=1e-5):
    X = X_init.copy() if X_init is not None else np.column_stack([B*w[0], B*w[1]])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N):
            X[i] = best_response(i, X, f_fn, k, tau)
        if np.max(np.abs(X - X_old)) < tol:
            break
    return X

def heterogeneity(X):
    shares = X[:, 0] / B
    return float(np.std(shares))

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
# results[f_label][tau][k] = het value
results = {fl: {tau: {} for tau in TAU_VALS} for fl, _ in F_SPECS}

total = len(F_SPECS) * len(TAU_VALS) * len(K_VALS)
done  = 0

for f_label, f_fn in F_SPECS:
    print(f"\nf = {f_label}")
    for tau in TAU_VALS:
        X_prev = None
        for k in K_VALS:
            # Warm start from previous k (smaller k usually closer to proportional)
            X_nash = find_nash(f_fn, k, tau, X_init=X_prev)
            het = heterogeneity(X_nash)
            results[f_label][tau][k] = het
            shares = X_nash[:, 0] / B
            nc = int(np.sum((shares < .05) | (shares > .95)))
            print(f"  tau={tau}  k={k}  het={het:.4f}  corners={nc}/{N}")
            X_prev = X_nash
            done += 1
        print(f"  [{done}/{total} done]")

# ---------------------------------------------------------------------------
# Plot: 5 figures (one per f), lines = tau values, x-axis = k
# ---------------------------------------------------------------------------

tau_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(TAU_VALS)))
prop_het = float(np.std(np.full(N, w[0])))  # = 0 by construction

fig, axes = plt.subplots(1, len(F_SPECS), figsize=(3.5 * len(F_SPECS), 4.5), sharey=False)

for ax, (f_label, _) in zip(axes, F_SPECS):
    for tau, color in zip(TAU_VALS, tau_colors):
        hets = [results[f_label][tau][k] for k in K_VALS]
        ax.plot(K_VALS, hets, "o-", color=color, lw=2, ms=6, label=f"$\\tau={tau}$")
    ax.axhline(0, ls="--", color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("$k$ (group size)", fontsize=10)
    ax.set_ylabel("Heterogeneity  (std of dim-0 shares)", fontsize=9)
    ax.set_title(f_label, fontsize=11, fontweight="bold")
    ax.set_xticks(K_VALS)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(bottom=-0.01)

fig.suptitle(
    r"Heterogeneity vs $k$-way contest size  |  $N=8$, $K=2$, $w=[1/3,\,2/3]$"
    "\nHeterogeneity = std of dim-0 shares across firms (0 = proportional)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("plot_kway_sweep.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("\nSaved: plot_kway_sweep.pdf")
