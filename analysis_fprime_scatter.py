"""
1x4 scatter plot: for each alpha in {0.25, 0.5, 0.6, 0.75},
axes are f'(x_{i,1}) vs f'(x_{i,2}), one dot per firm, at the
optimal budget from the alpha sweep.

Optimal budgets (from analysis_alpha_curve.py / analysis_sigma_extreme.py):
  alpha < ~0.65: B = [eps, eps, B_total - 2*eps]  (two tiny, one huge)
  alpha >= ~0.65: B = [t, t-delta, eps]            (two large, one tiny)
We re-derive them here cleanly.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
ALPHAS = [0.25, 0.50, 0.60, 0.75]


def make_power(alpha):
    f  = lambda x: max(x, 0.0) ** alpha
    fp = lambda x: alpha * max(x, 1e-12) ** (alpha - 1)
    return f, fp


def sigma_prime(z):
    s = expit(z)
    return s * (1.0 - s)


def compute_Sigma(X, f_fn):
    fx = np.array([[f_fn(X[i, k]) for k in range(K)] for i in range(N)])
    Sigma = np.zeros((N, K))
    for i in range(N):
        for ip in range(N):
            if ip != i:
                diff = fx[i] - fx[ip]
                Sigma[i] += np.array([sigma_prime(d) for d in diff])
    return Sigma


def max_log_ror(Sigma):
    pairs = [(0, 1), (0, 2), (1, 2)]
    vals = []
    for i, j in pairs:
        ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
        rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
        vals.append(abs(np.log(max(ri / (rj + 1e-12), 1e-12))))
    return max(vals)


def run(B, alpha):
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B.copy(), W.copy(), W.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)
    Sigma = compute_Sigma(X, f_fn)
    return X, Sigma


def best_budget(alpha, n_eps=20):
    """Return (B_opt, X_nash, Sigma) at max|log R| extreme structure."""
    best_val, best_B = -np.inf, None

    # Structure A: B = [eps, eps, B_total - 2*eps]
    for eps in np.logspace(-2, -0.3, n_eps):
        if 2 * eps >= B_TOTAL - 0.05:
            continue
        B = np.array([eps, eps, B_TOTAL - 2 * eps])
        _, Sigma = run(B, alpha)
        v = max_log_ror(Sigma)
        if v > best_val:
            best_val, best_B = v, B.copy()

    # Structure B: B0=t, B1=t-0.1, B2=eps; sweep t
    delta = 0.1
    for t in np.linspace(0.5, (B_TOTAL - 0.05 - delta) / 2, n_eps):
        B = np.array([t, t - delta, B_TOTAL - 2 * t + delta - 0.05])
        if np.any(B <= 0) or B[0] <= B[1] or B[1] <= B[2]:
            continue
        _, Sigma = run(B, alpha)
        v = max_log_ror(Sigma)
        if v > best_val:
            best_val, best_B = v, B.copy()

    X_best, Sigma_best = run(best_B, alpha)
    return best_B, X_best, Sigma_best


# -----------------------------------------------------------------------
# Collect data
# -----------------------------------------------------------------------

firm_colors = ["steelblue", "darkorange", "mediumseagreen"]
firm_labels = ["firm 0", "firm 1", "firm 2"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(r"$f'(x_{i,1})$ vs $f'(x_{i,2})$ at max-$|\log R|$ budget: $f(x)=x^\alpha$, $N=3$, $K=2$",
             fontsize=12, fontweight="bold")

for ax, alpha in zip(axes, ALPHAS):
    f_fn, fp_fn = make_power(alpha)
    B_opt, X_nash, Sigma = best_budget(alpha)

    fp_vals = np.array([[fp_fn(X_nash[i, k]) for k in range(K)] for i in range(N)])
    # fp_vals[i, 0] = f'(x_{i,1}),  fp_vals[i, 1] = f'(x_{i,2})

    print(f"alpha={alpha}  B_opt={B_opt.round(3)}")
    for i in range(N):
        print(f"  firm {i}: x={X_nash[i].round(4)}  f'(x)={fp_vals[i].round(4)}")

    for i in range(N):
        ax.scatter(fp_vals[i, 0], fp_vals[i, 1],
                   color=firm_colors[i], s=120, zorder=5,
                   label=f"firm {i}  (B={B_opt[i]:.2f})")

    # Diagonal reference (equal marginals across categories)
    all_fp = fp_vals.flatten()
    lo, hi = all_fp.min() * 0.8, all_fp.max() * 1.2
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, alpha=0.6)

    ax.set_xlabel(r"$f'(x_{i,1})$  (cat 1, $w=0.7$)", fontsize=10)
    ax.set_ylabel(r"$f'(x_{i,2})$  (cat 2, $w=0.3$)", fontsize=10)
    ax.set_title(rf"$\alpha={alpha}$", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")

plt.tight_layout()
plt.savefig("plot_fprime_scatter.pdf", bbox_inches="tight")
plt.savefig("plot_fprime_scatter.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_fprime_scatter.pdf / .png")
