"""
k-way match evaluation: firm i competes in groups of size k drawn uniformly at random.
Score = average softmax probability of being best across all C(N-1,k-1) possible groups.

  k=2,  tau=1  =>  exactly current pairwise objective
  k=N,  tau=1  =>  full softmax (all models compete simultaneously)

Sweep k in {2, 3, 4, 6, 8, 12} for two tau values:
  tau=1.0  (k=2 recovers current)
  tau=0.5  (where k=N already specialises)

N=12, K=2, B=linspace(5,1,12), w=[1/3,2/3], f in {log(1+x), sqrt(x), 1-e^{-x}}.
"""

import math, numpy as np, matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import softmax as sp_softmax

N, K = 12, 2
B    = np.linspace(5.0, 1.0, N)
w    = np.array([1/3, 2/3])
fv   = np.vectorize

F_SPECS = [
    (r"$\log(1{+}x)$", lambda x: math.log(1+max(x,0)),   lambda x: 1/(1+max(x,0))),
    (r"$\sqrt{x}$",    lambda x: math.sqrt(max(x,0)),     lambda x: 0.5/math.sqrt(max(x,1e-12))),
    (r"$1-e^{-x}$",    lambda x: 1-math.exp(-max(x,0)),   lambda x: math.exp(-max(x,0))),
]

K_VALS  = [2, 3, 4, 6, 8, 12]
TAU_VALS = [1.0, 0.5]

# ---------------------------------------------------------------------------
# Precompute subset indices for each (k, i): list of frozensets of rivals
# ---------------------------------------------------------------------------

def get_subsets(i, k):
    others = [j for j in range(N) if j != i]
    return list(combinations(others, k-1))

SUBSETS = {(i, k): get_subsets(i, k) for k in K_VALS for i in range(N)}

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def obj_kway(x_i, i, X, f_fn, k, tau):
    X2 = X.copy(); X2[i] = x_i
    th  = fv(f_fn)(X2)          # (N, K)
    subs = SUBSETS[(i, k)]
    total = 0.0
    for k_dim in range(K):
        score = 0.0
        for S in subs:
            idx = [i] + list(S)
            sm  = sp_softmax(th[idx, k_dim] / tau)
            score += sm[0]      # position 0 = firm i
        total += w[k_dim] * score / len(subs)
    return total

def best_response(i, X, f_fn, k, tau):
    Bi  = B[i]
    x0s = [X[i].copy(),
            np.array([Bi*w[0], Bi*w[1]]),
            np.array([Bi*.97,  Bi*.03]),
            np.array([Bi*.03,  Bi*.97])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -obj_kway(x, i, X, f_fn, k, tau), x0,
                       method="SLSQP", bounds=[(0, Bi)]*K,
                       constraints=[{"type":"ineq","fun":lambda x: Bi-x.sum()}],
                       options={"ftol":1e-9, "maxiter":200})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(f_fn, k, tau, max_iter=50, tol=1e-5):
    X = np.column_stack([B*w[0], B*w[1]])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N): X[i] = best_response(i, X, f_fn, k, tau)
        if np.max(np.abs(X - X_old)) < tol: break
    return X

def sw(X, f_fn):
    return float(np.dot(w, np.max(fv(f_fn)(X), axis=0)))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# results[tau][f_label][k] = (X_nash, sw)
results = {tau: {fl: {} for fl,_,__ in F_SPECS} for tau in TAU_VALS}

for tau in TAU_VALS:
    for f_label, f_fn, _ in F_SPECS:
        print(f"\ntau={tau}  f={f_label}")
        print(f"  {'k':>3}  {'SW':>8}  {'corners':>8}")
        for k in K_VALS:
            X = find_nash(f_fn, k, tau)
            shares = X[:,0]/B
            nc = int(np.sum((shares<.05)|(shares>.95)))
            results[tau][f_label][k] = (X.copy(), sw(X, f_fn))
            print(f"  {k:>3}  {sw(X,f_fn):>8.4f}  {nc:>8}/12")

# ---------------------------------------------------------------------------
# Plot: one figure per tau, rows=f functions, cols=k values (scatter)
# ---------------------------------------------------------------------------

pt_colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)
X_prop = np.column_stack([B*w[0], B*w[1]])

for tau in TAU_VALS:
    nrows, ncols = len(F_SPECS), len(K_VALS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.8*nrows))
    fig.suptitle(
        f"$k$-way match evaluation  |  $\\tau={tau}$,  $K=2$, $w=[1/3,\\,2/3]$, $N=12$\n"
        "circles=Nash,  stars=proportional,  dashed=proportional ray\n"
        r"($k=2,\tau=1$ recovers current pairwise; $k=N$ = full softmax)",
        fontsize=11, fontweight="bold"
    )

    for row, (f_label, f_fn, _) in enumerate(F_SPECS):
        for col, k in enumerate(K_VALS):
            ax = axes[row, col]
            X_nash, sw_val = results[tau][f_label][k]
            shares = X_nash[:,0]/B
            nc = int(np.sum((shares<.05)|(shares>.95)))

            ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
            for bi in B:
                ax.plot([bi,0],[0,bi],"-",color="gray",lw=0.4,alpha=0.15)
            for i in range(N):
                ax.scatter(X_nash[i,0], X_nash[i,1],
                           color=pt_colors[i], s=50, marker="o", zorder=5)
                ax.scatter(X_prop[i,0], X_prop[i,1],
                           color=pt_colors[i], s=120, marker="*",
                           zorder=6, edgecolors="k", linewidths=0.3, alpha=0.5)
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)
            ax.set_xlabel("$x_{i,1}$", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{f_label}\n$x_{{i,2}}$", fontsize=9, fontweight="bold")
            title = f"$k={k}$"
            if row == 0: title = f"$k={k}$\n"
            ax.set_title(f"{title}SW={sw_val:.3f}  ({nc}/12)",
                         fontsize=8, fontweight="bold")

    axes[0,0].scatter([],[],color="gray",s=50, marker="o",label="Nash")
    axes[0,0].scatter([],[],color="gray",s=120,marker="*",
                      edgecolors="k",linewidths=0.3,alpha=0.5,label="Proportional")
    axes[0,0].legend(fontsize=7)

    plt.tight_layout()
    fname = f"plot_kway_tau{str(tau).replace('.','p')}.pdf"
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fname}")
