"""
Extension of analysis_sigma_ratio_alpha.py.

Enforce B_0 > B_1 > B_2 and sweep the budget space more thoroughly.
Also test limiting/extreme configurations to see if they maximize |log R|.

Questions:
  1. With B_0>B_1>B_2, what is the optimal interior split per alpha?
  2. Does pushing to extremes (e.g. B_0→B_total, B_2→0) keep improving things?
  3. Is the "two similar + one outlier" pattern preserved? Which pair is similar?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from competition_simulation import CompetitionSimulation

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
ALPHAS = [0.25, 0.50, 0.75]
COLORS = ["steelblue", "darkorange", "mediumseagreen"]
EPS = 0.05   # minimum budget per firm


def make_power(alpha):
    f = lambda x: max(x, 0.0) ** alpha
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


def ratio_of_ratios(Sigma, i, j):
    ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
    rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
    return ri / (rj + 1e-12)


def all_rors(Sigma):
    pairs = [(0, 1), (0, 2), (1, 2)]
    return {p: ratio_of_ratios(Sigma, *p) for p in pairs}


def max_abs_log_ror(Sigma):
    """Max |log R| over all firm pairs."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    vals = [abs(np.log(max(ratio_of_ratios(Sigma, *p), 1e-12))) for p in pairs]
    return max(vals)


def run_nash_and_sigma(B, alpha):
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B.copy(), W.copy(), W.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
    Sigma = compute_Sigma(X_nash, f_fn)
    return X_nash, Sigma


# -----------------------------------------------------------------------
# 1. Dense 2D grid over (B0, B1) with B_0 > B_1 > B_2 = B_total - B0 - B1
#    Objective: max |log R| over all pairs
# -----------------------------------------------------------------------

print("Grid search: B_0 > B_1 > B_2, B_total=9")
print("=" * 60)

GRID_N = 60  # grid resolution per axis

grid_results = {}

for alpha in ALPHAS:
    best_val, best_B = -np.inf, None
    vals = np.full((GRID_N, GRID_N), np.nan)

    b0_grid = np.linspace(EPS, B_TOTAL - 2 * EPS, GRID_N)
    b1_grid = np.linspace(EPS, B_TOTAL - 2 * EPS, GRID_N)

    for ii, b0 in enumerate(b0_grid):
        for jj, b1 in enumerate(b1_grid):
            b2 = B_TOTAL - b0 - b1
            # Enforce B_0 > B_1 > B_2 > 0
            if not (b0 > b1 > b2 > EPS):
                continue
            B = np.array([b0, b1, b2])
            _, Sigma = run_nash_and_sigma(B, alpha)
            v = max_abs_log_ror(Sigma)
            vals[ii, jj] = v
            if v > best_val:
                best_val, best_B = v, B.copy()

    grid_results[alpha] = {"best_val": best_val, "best_B": best_B, "vals": vals,
                           "b0_grid": b0_grid, "b1_grid": b1_grid}
    _, Sigma_best = run_nash_and_sigma(best_B, alpha)
    rors = all_rors(Sigma_best)
    print(f"\nalpha={alpha}:")
    print(f"  Best B = {best_B.round(3)}  (total={best_B.sum():.2f})")
    print(f"  max|log R| = {best_val:.5f}")
    print(f"  RoRs: R(0,1)={rors[(0,1)]:.5f}  R(0,2)={rors[(0,2)]:.5f}  R(1,2)={rors[(1,2)]:.5f}")
    # Which pair drives the max?
    winning_pair = max([(0,1),(0,2),(1,2)], key=lambda p: abs(np.log(max(rors[p], 1e-12))))
    print(f"  Winning pair: {winning_pair}  |log R|={abs(np.log(max(rors[winning_pair],1e-12))):.5f}")


# -----------------------------------------------------------------------
# 2. Limit analysis: push budgets toward extreme corners
#    Parameterize as B = [B_total*(1-s-t), B_total*s, B_total*t]
#    with s > t > 0 and (1-s-t) > s > t  (ordering constraint)
#    Push s→0 (B1,B2→0) or push (1-s-t)→1-2eps (B0 dominates)
# -----------------------------------------------------------------------

print("\n\nLimit sweep: push B toward extremes")
print("=" * 60)

# Parameterize: B0 = B_total - eps1 - eps2, B1=eps1, B2=eps2 with eps1>eps2
# Sweep eps2 from very small to eps1, eps1 from very small up
# Let's do: fix eps2=EPS, sweep eps1 from eps2+EPS to B_total/2

limit_results = {}

for alpha in ALPHAS:
    eps2 = EPS
    eps1_vals = np.linspace(eps2 + EPS, B_TOTAL / 2, 60)
    ror_vals = []
    B_vals = []
    for eps1 in eps1_vals:
        B = np.array([B_TOTAL - eps1 - eps2, eps1, eps2])
        if B[0] <= B[1]:  # ordering check
            ror_vals.append(np.nan)
            B_vals.append(B)
            continue
        _, Sigma = run_nash_and_sigma(B, alpha)
        ror_vals.append(max_abs_log_ror(Sigma))
        B_vals.append(B)
    limit_results[alpha] = {"eps1_vals": eps1_vals, "ror_vals": np.array(ror_vals)}

    valid = np.isfinite(ror_vals)
    best_idx = np.nanargmax(ror_vals)
    print(f"  alpha={alpha}: max at eps1={eps1_vals[best_idx]:.3f} "
          f"=> B={B_vals[best_idx].round(3)}, |log R|={ror_vals[best_idx]:.5f}")
    print(f"    At most extreme (eps1→{eps1_vals[0]:.2f}): |log R|={ror_vals[0]:.5f}")
    print(f"    At eps1=B/2 (equal B1=B2): |log R|={ror_vals[-1]:.5f}")


# -----------------------------------------------------------------------
# 3. Symmetric limit: B1 = B2 = eps (two small firms), sweep eps→0
#    (No ordering constraint here; B0 dominates)
# -----------------------------------------------------------------------

print("\n\nSymmetric limit: B1=B2=eps, B0=B_total-2*eps, sweep eps->0")
print("=" * 60)

eps_vals = np.logspace(-2, np.log10(B_TOTAL / 3 - 0.01), 50)
sym_results = {}

for alpha in ALPHAS:
    ror_eps = []
    for eps in eps_vals:
        B = np.array([B_TOTAL - 2 * eps, eps, eps])
        # B_0 > B_1 = B_2 (ties ok here)
        _, Sigma = run_nash_and_sigma(B, alpha)
        ror_eps.append(max_abs_log_ror(Sigma))
    sym_results[alpha] = np.array(ror_eps)
    print(f"  alpha={alpha}: |log R| at eps=0.01: {ror_eps[0]:.5f}  "
          f"at eps=B/3: {ror_eps[-1]:.5f}  monotone? {all(np.diff(ror_eps)<=0.001)}")


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(r"Extreme budget analysis: $N=3$, $K=2$, $f(x)=x^\alpha$", fontsize=13)

# Row 1: heatmap of max|log R| in (B0, B1) space for each alpha
for col, alpha in enumerate(ALPHAS):
    ax = axes[0, col]
    r = grid_results[alpha]
    im = ax.imshow(r["vals"].T, origin="lower", aspect="auto",
                   extent=[r["b0_grid"][0], r["b0_grid"][-1],
                           r["b1_grid"][0], r["b1_grid"][-1]],
                   cmap="viridis")
    plt.colorbar(im, ax=ax, label="|log R|")
    B_opt = r["best_B"]
    ax.scatter([B_opt[0]], [B_opt[1]], color="red", marker="*", s=120, zorder=5, label=f"opt={B_opt.round(1)}")
    ax.set_xlabel("$B_0$"); ax.set_ylabel("$B_1$")
    ax.set_title(rf"$\alpha={alpha}$: max|log R| in $(B_0,B_1)$")
    ax.legend(fontsize=8)

# Row 2: limit sweeps
# Left: asymmetric limit (eps1 sweep, B0 dominates)
ax = axes[1, 0]
for alpha, color in zip(ALPHAS, COLORS):
    r = limit_results[alpha]
    ax.plot(r["eps1_vals"], r["ror_vals"], color=color, lw=2, label=rf"$\alpha$={alpha}")
ax.set_xlabel(r"$B_1$ (= eps1, $B_2$=0.05 fixed)")
ax.set_ylabel("|log R| max over pairs")
ax.set_title("Limit: $B_0=9-B_1-0.05$, sweep $B_1$")
ax.legend(fontsize=9)

# Middle: symmetric limit (B1=B2=eps sweep)
ax = axes[1, 1]
for alpha, color in zip(ALPHAS, COLORS):
    ax.semilogx(eps_vals, sym_results[alpha], color=color, lw=2, label=rf"$\alpha$={alpha}")
ax.set_xlabel(r"$\varepsilon$ (= $B_1 = B_2$, log scale)")
ax.set_ylabel("|log R| max over pairs")
ax.set_title(r"Symmetric: $B_0=9-2\varepsilon$, $B_1=B_2=\varepsilon$")
ax.legend(fontsize=9)

# Right: optimal B distributions across alpha (B_0>B_1>B_2 constrained)
ax = axes[1, 2]
x_pos = np.arange(N)
width = 0.25
for idx, (alpha, color) in enumerate(zip(ALPHAS, COLORS)):
    B_opt = grid_results[alpha]["best_B"]
    offset = (idx - 1) * width
    ax.bar(x_pos + offset, B_opt, width=width * 0.9, color=color,
           alpha=0.85, label=rf"$\alpha$={alpha}")
ax.set_xticks(list(x_pos)); ax.set_xticklabels([f"firm {i}" for i in range(N)])
ax.set_ylabel("Budget $B_i$")
ax.set_title("Optimal B (ordered $B_0>B_1>B_2$)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_sigma_extreme.pdf", bbox_inches="tight")
plt.savefig("plot_sigma_extreme.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_sigma_extreme.pdf / .png")
