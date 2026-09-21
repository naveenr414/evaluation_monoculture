"""
Three evaluation schemes × three production functions × two settings (intermediate / extreme).

Schemes:
  (A) Power mean  p:   intermediate p=-0.5,  extreme p=0.01 (geometric)
  (B) Softmax     tau: intermediate tau=0.5,  extreme tau=0.1
  (C) Nearest     m:   intermediate m=4,      extreme m=1 (only adjacent firms)

Functions:
  log(1+x),  sqrt(x),  1-exp(-x)

N=12, K=2, B=linspace(5,1,12), w=[1/3,2/3].
Output: two plots (intermediate, extreme), each a 3×3 grid of dim-0 share vs budget.
"""

import math, numpy as np, matplotlib.pyplot as plt
from scipy.special import expit, softmax as sp_softmax
from scipy.optimize import minimize

N, K = 12, 2
B    = np.linspace(5.0, 1.0, N)
w    = np.array([1/3, 2/3])

# production functions
F_SPECS = [
    (r"$\log(1+x)$",  lambda x: math.log(1+max(x,0)),        lambda x: 1/(1+max(x,0))),
    (r"$\sqrt{x}$",   lambda x: math.sqrt(max(x,0)),          lambda x: 0.5/math.sqrt(max(x,1e-12))),
    (r"$1-e^{-x}$",   lambda x: 1-math.exp(-max(x,0)),        lambda x: math.exp(-max(x,0))),
]

def fv(f, X): return np.vectorize(f)(X)

# ---------------------------------------------------------------------------
# Objective builders
# ---------------------------------------------------------------------------

def obj_power_mean(x_i, i, X, f, p):
    th_i = fv(f, x_i); th = fv(f, X)
    others = [ip for ip in range(N) if ip != i]
    total = 0.0
    for k in range(K):
        wins = np.clip([float(expit(th_i[k]-th[ip,k])) for ip in others], 1e-9, 1-1e-9)
        if abs(p) < 1e-3:
            score = math.exp(np.mean(np.log(wins)))
        else:
            score = (np.mean(wins**p))**(1/p)
        total += w[k]*score
    return total

def obj_softmax(x_i, i, X, f, tau):
    X2 = X.copy(); X2[i] = x_i
    th  = fv(f, X2)
    return sum(w[k]*sp_softmax(th[:,k]/tau)[i] for k in range(K))

def obj_nearest(x_i, i, X, f, m):
    # m nearest rivals by budget rank
    ranks = np.argsort(np.abs(np.arange(N)-i))
    rivals = [r for r in ranks if r != i][:m]
    th_i = fv(f, x_i); th = fv(f, X)
    return sum(w[k]*float(expit(th_i[k]-th[ip,k]))
               for ip in rivals for k in range(K))

# ---------------------------------------------------------------------------
# Nash solver
# ---------------------------------------------------------------------------

def best_response(i, X, obj_fn, n_restarts=4):
    Bi = B[i]
    x0s = [X[i].copy(), np.array([Bi*w[0],Bi*w[1]]),
            np.array([Bi*.97,Bi*.03]), np.array([Bi*.03,Bi*.97]),
            np.array([Bi*.6, Bi*.4]),  np.array([Bi*.4, Bi*.6])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s[:n_restarts+1]:
        res = minimize(lambda x: -obj_fn(x, i, X), x0,
                       method="SLSQP", bounds=[(0,Bi)]*K,
                       constraints=[{"type":"ineq","fun":lambda x: Bi-x.sum()}],
                       options={"ftol":1e-9,"maxiter":300})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x,0,None)
    return best_x

def find_nash(obj_fn, max_iter=60, tol=1e-5):
    X = np.column_stack([B*w[0], B*w[1]])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N): X[i] = best_response(i, X, obj_fn)
        if np.max(np.abs(X-X_old)) < tol: break
    return X

def sw(X, f): return float(np.dot(w, np.max(fv(f,X), axis=0)))

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SETTINGS = {
    "intermediate": {"power_p": -0.5, "softmax_tau": 0.5, "nearest_m": 4},
    "extreme":      {"power_p":  0.01, "softmax_tau": 0.1, "nearest_m": 1},
}

SCHEMES = [
    ("Power mean",    "p"),
    ("Softmax",       "τ"),
    ("Nearest rivals","m"),
]

# ---------------------------------------------------------------------------
# Run all combinations
# ---------------------------------------------------------------------------

data = {}  # data[setting][scheme_name][f_label] = shares array

for sname, sparams in SETTINGS.items():
    data[sname] = {s[0]: {} for s in SCHEMES}
    for f_label, f_fn, fp_fn in F_SPECS:
        print(f"\n{sname}  |  f={f_label}")

        # Power mean
        p = sparams["power_p"]
        X = find_nash(lambda x,i,X: obj_power_mean(x,i,X,f_fn,p))
        shares = X[:,0]/B
        data[sname]["Power mean"][f_label] = (X.copy(), shares, sw(X,f_fn))
        print(f"  power(p={p:.2f}): corners={int(np.sum((shares<.05)|(shares>.95)))}/N  SW={sw(X,f_fn):.4f}")

        # Softmax
        tau = sparams["softmax_tau"]
        X = find_nash(lambda x,i,X: obj_softmax(x,i,X,f_fn,tau))
        shares = X[:,0]/B
        data[sname]["Softmax"][f_label] = (X.copy(), shares, sw(X,f_fn))
        print(f"  softmax(τ={tau}):  corners={int(np.sum((shares<.05)|(shares>.95)))}/N  SW={sw(X,f_fn):.4f}")

        # Nearest rivals
        m = sparams["nearest_m"]
        X = find_nash(lambda x,i,X: obj_nearest(x,i,X,f_fn,m))
        shares = X[:,0]/B
        data[sname]["Nearest rivals"][f_label] = (X.copy(), shares, sw(X,f_fn))
        print(f"  nearest(m={m}):    corners={int(np.sum((shares<.05)|(shares>.95)))}/N  SW={sw(X,f_fn):.4f}")

# ---------------------------------------------------------------------------
# Plot: K=2 scatter, rows=schemes, cols=f functions
# ---------------------------------------------------------------------------

pt_colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)

for sname, sparams in SETTINGS.items():
    params_str = (f"p={sparams['power_p']},  "
                  f"$\\tau$={sparams['softmax_tau']},  "
                  f"m={sparams['nearest_m']}")
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle(
        f"Nash allocations — {sname} setting  ({params_str})\n"
        r"$K=2$, $w=[1/3,\,2/3]$, $N=12$, $f(x)$ varies  |  "
        "circles=Nash, stars=proportional, dashed=proportional ray",
        fontsize=11, fontweight="bold"
    )

    for row, (scheme_name, param_sym) in enumerate(SCHEMES):
        for col, (f_label, f_fn, _) in enumerate(F_SPECS):
            ax = axes[row, col]
            X_nash, shares, sw_val = data[sname][scheme_name][f_label]
            X_prop = np.column_stack([B*w[0], B*w[1]])

            # proportional ray + budget constraint lines
            ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
            for bi in B:
                ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.4, alpha=0.15)

            for i in range(N):
                ax.scatter(X_nash[i,0], X_nash[i,1],
                           color=pt_colors[i], s=55, marker="o", zorder=5)
                ax.scatter(X_prop[i,0], X_prop[i,1],
                           color=pt_colors[i], s=130, marker="*",
                           zorder=6, edgecolors="k", linewidths=0.3, alpha=0.5)

            n_corn = int(np.sum((shares < .05) | (shares > .95)))
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)
            ax.set_xlabel("$x_{i,1}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{scheme_name}\n$x_{{i,2}}$", fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(f"{f_label}\nSW={sw_val:.3f}  ({n_corn}/12 corners)",
                             fontsize=9, fontweight="bold")
            else:
                ax.set_title(f"SW={sw_val:.3f}  ({n_corn}/12 corners)", fontsize=9)

    # shared legend in first panel
    axes[0,0].scatter([], [], color="gray", s=55,  marker="o", label="Nash")
    axes[0,0].scatter([], [], color="gray", s=130, marker="*",
                      edgecolors="k", linewidths=0.3, alpha=0.5, label="Proportional")
    axes[0,0].legend(fontsize=8)

    plt.tight_layout()
    fname = f"plot_specialization_{sname}.pdf"
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
