"""
Dead-zone sigma: models a 3-star eval scheme ("similar" / "A wins" / "B wins").

  sigma_dz(z) = 0.5 * [sigma(beta*(z+delta)) + sigma(beta*(z-delta))]

  - z near 0:  both inner sigmas cancel -> stays at 0.5  (tie)
  - z >> delta: both -> 1                -> 1.0           (clear A win)
  - z << -delta: both -> 0              -> 0.0           (clear B win)
  - At z = +delta: ~0.75  (edge of dead zone)   => 25%-75% band

sigma_dz'(z) = 0.5*beta * [sigma'(beta*(z+delta)) + sigma'(beta*(z-delta))]
  Bimodal: peaks at z = +-delta, near-zero inside the dead zone.

We compare: standard sigma (delta=0) vs dead-zone with delta in {0.3, 0.7, 1.2}.
f(x) = log(x), K=2, w=[1/3, 2/3], N=4 firms.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize

N, K = 4, 2
rng  = np.random.default_rng(42)
B    = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
w    = np.array([1/3, 2/3])

f_v  = np.vectorize(lambda x: math.log(max(x, 1e-12)))

def make_deadzone(delta, beta=8.0):
    def win(a, b):
        z = a - b
        return 0.5 * (expit(beta*(z+delta)) + expit(beta*(z-delta)))
    def winp(a, b):
        z = a - b
        s1 = expit(beta*(z+delta)); s2 = expit(beta*(z-delta))
        return 0.5 * beta * (s1*(1-s1) + s2*(1-s2))
    return win, winp

VARIANTS = [
    ("standard\n($\\delta=0$)",    *make_deadzone(0.0)),
    ("dead-zone\n$\\delta=0.3$",   *make_deadzone(0.3)),
    ("dead-zone\n$\\delta=0.7$",   *make_deadzone(0.7)),
    ("dead-zone\n$\\delta=1.2$",   *make_deadzone(1.2)),
]

# ---------------------------------------------------------------------------
# Show what sigma_dz looks like
# ---------------------------------------------------------------------------
z_vals = np.linspace(-3, 3, 300)
fig0, ax0 = plt.subplots(figsize=(6, 3.5))
for name, wfn, _ in VARIANTS:
    ax0.plot(z_vals, [wfn(z, 0) for z in z_vals], lw=2, label=name.replace("\n"," "))
ax0.axhline(0.75, ls=":", color="gray", lw=1)
ax0.axhline(0.25, ls=":", color="gray", lw=1)
ax0.axhline(0.5,  ls="--", color="black", lw=0.8)
ax0.set_xlabel("$f_i - f_j$  (quality gap)"); ax0.set_ylabel("win probability")
ax0.set_title("Dead-zone $\\sigma$ shapes  ($\\beta=8$)", fontweight="bold")
ax0.legend(fontsize=8); plt.tight_layout()
plt.savefig("plot_deadzone_sigma_shape.pdf", format="pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Best-response Nash finder
# ---------------------------------------------------------------------------

def firm_obj(x_i, i, X, win_fn):
    th_i = f_v(x_i); th = f_v(X)
    return sum(w[k] * win_fn(th_i[k], th[ip, k])
               for ip in range(N) if ip != i for k in range(K))

def best_response(i, X, win_fn):
    Bi = B[i]
    x0s = [X[i].copy(), np.array([Bi*w[0], Bi*w[1]])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -firm_obj(x, i, X, win_fn), x0,
                       method="SLSQP",
                       bounds=[(0, Bi)]*K,
                       constraints=[{"type": "ineq", "fun": lambda x: Bi - x.sum()}],
                       options={"ftol": 1e-8, "maxiter": 200})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x, 0, None)
    return best_x

def find_nash(win_fn, max_iter=50, tol=1e-5):
    X = np.column_stack([B*w[0], B*w[1]])
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(N): X[i] = best_response(i, X, win_fn)
        if np.max(np.abs(X - X_old)) < tol: break
    return X

def sw(X): return float(np.dot(w, np.max(f_v(X), axis=0)))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

X_prop = np.column_stack([B*w[0], B*w[1]])
sw_prop = sw(X_prop)
print(f"B={np.round(B,2)}, w={w}, proportional SW={sw_prop:.4f}\n")
print(f"{'Variant':<22}  {'shares (dim0)':<34}  {'SW':>7}")
print("-"*68)

results = []
for name, wfn, _ in VARIANTS:
    X = find_nash(wfn)
    shares = X[:,0] / B
    results.append({"name": name, "X": X, "sw": sw(X), "shares": shares})
    print(f"  {name.replace(chr(10),' '):<20}  "
          f"{[round(s,3) for s in shares]}  SW={sw(X):.4f}")

# ---------------------------------------------------------------------------
# Plot: K=2 scatter (one panel per variant)
# ---------------------------------------------------------------------------

colors = plt.cm.plasma(np.linspace(0.1, 0.85, N))
t = np.linspace(0, B.max(), 200)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle(
    f"Nash allocations under dead-zone $\\sigma$  |  $K=2$, "
    f"$w=[1/3,\\,2/3]$, $N={N}$ firms,  $\\beta=8$\n"
    "(circles = Nash, stars = proportional benchmark; dashed = proportional ray)",
    fontsize=11, fontweight="bold"
)

for ax, res in zip(axes, results):
    X = res["X"]
    ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], "-", color="gray", lw=0.5, alpha=0.2)
    for i in range(N):
        ax.scatter(X[i,0], X[i,1], color=colors[i], s=90, marker="o", zorder=5)
        ax.scatter(X_prop[i,0], X_prop[i,1], color=colors[i], s=200,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.4)
        ax.annotate(f"B={B[i]:.1f}", (X[i,0], X[i,1]),
                    textcoords="offset points", xytext=(4,3), fontsize=7, color=colors[i])
    ax.set_xlabel("$x_{i,1}$", fontsize=9)
    if ax is axes[0]: ax.set_ylabel("$x_{i,2}$", fontsize=9)
    ax.set_title(f"{res['name']}\nSW={res['sw']:.4f}  (prop={sw_prop:.4f})",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

axes[0].scatter([], [], color="gray", s=90,  marker="o", label="Nash")
axes[0].scatter([], [], color="gray", s=200, marker="*", edgecolors="k",
                linewidths=0.4, label="Proportional")
axes[0].legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_deadzone_k2.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("\nSaved: plot_deadzone_sigma_shape.pdf, plot_deadzone_k2.pdf")
