"""
Experiment: N=3 firms, K=2 categories.
f(x) = x^alpha for alpha in {0.25, 0.5, 0.75}.

For each alpha:
  (a) Fix budget B=[5,3,1], w=[0.7,0.3]. Find Nash equilibrium.
      Compute Sigma_{i,k} = sum_{i'!=i} sigma'(f(x_{i,k}) - f(x_{i',k}))
      Report Sigma_{i,1}/Sigma_{i,2} for each firm i.
      Report ratio-of-ratios R_{i,j} = (Sigma_{i,1}/Sigma_{i,2}) / (Sigma_{j,1}/Sigma_{j,2})
      for all firm pairs (i,j).

  (b) Fix total budget B_total = 9. Optimize over B_1+B_2+B_3=B_total
      (B_i >= 0.1) to maximize R_{1,2} (or a summary of all pairs).
      Check if the optimal distribution differs across alpha values.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize, differential_evolution
from competition_simulation import CompetitionSimulation

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

N, K = 3, 2
B_FIXED = np.array([5.0, 3.0, 1.0])
W = np.array([0.7, 0.3])
B_TOTAL = B_FIXED.sum()   # = 9.0

ALPHAS = [0.25, 0.50, 0.75]
COLORS = ["steelblue", "darkorange", "mediumseagreen"]


def make_power(alpha):
    f = lambda x: max(x, 0.0) ** alpha
    fp = lambda x: alpha * max(x, 1e-12) ** (alpha - 1)
    return f, fp


def sigma_prime(z):
    s = expit(z)
    return s * (1.0 - s)


def compute_Sigma(X, f_fn, N, K):
    """Sigma[i,k] = sum_{i'!=i} sigma'(f(x_{i,k}) - f(x_{i',k}))"""
    fx = np.array([[f_fn(X[i, k]) for k in range(K)] for i in range(N)])
    Sigma = np.zeros((N, K))
    for i in range(N):
        for ip in range(N):
            if ip != i:
                diff = fx[i] - fx[ip]   # shape (K,)
                Sigma[i] += np.array([sigma_prime(d) for d in diff])
    return Sigma


def ratio_of_ratios(Sigma, i, j):
    """R_{i,j} = (Sigma[i,0]/Sigma[i,1]) / (Sigma[j,0]/Sigma[j,1])"""
    ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
    rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
    return ri / (rj + 1e-12)


# -----------------------------------------------------------------------
# Part (a): fixed budget
# -----------------------------------------------------------------------

print("=" * 65)
print("PART (a): Fixed budget B=[5,3,1], w=[0.7,0.3]")
print("=" * 65)

part_a_results = {}

for alpha, color in zip(ALPHAS, COLORS):
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B_FIXED.copy(), W.copy(), W.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)

    Sigma = compute_Sigma(X_nash, f_fn, N, K)
    sigma_ratios = Sigma[:, 0] / (Sigma[:, 1] + 1e-12)  # Sigma_{i,1}/Sigma_{i,2}, shape (N,)

    pairs = [(0, 1), (0, 2), (1, 2)]
    rors = {(i, j): ratio_of_ratios(Sigma, i, j) for i, j in pairs}

    part_a_results[alpha] = {
        "X_nash": X_nash,
        "Sigma": Sigma,
        "sigma_ratios": sigma_ratios,
        "rors": rors,
    }

    print(f"\nalpha = {alpha}")
    print(f"  Nash X:\n    firm0={X_nash[0].round(4)}  firm1={X_nash[1].round(4)}  firm2={X_nash[2].round(4)}")
    print(f"  Sigma:\n    firm0={Sigma[0].round(6)}  firm1={Sigma[1].round(6)}  firm2={Sigma[2].round(6)}")
    print(f"  Sigma_{'{i,1}'}/Sigma_{'{i,2}'}:  " +
          "  ".join(f"firm{i}={sigma_ratios[i]:.4f}" for i in range(N)))
    for (i, j), ror in rors.items():
        print(f"  R_{{firm{i},firm{j}}} = {ror:.6f}")

# Summary table: how RoR changes with alpha
print("\n" + "=" * 65)
print("Ratio-of-ratios summary across alpha")
print("=" * 65)
print(f"{'alpha':>8}  {'R(0,1)':>10}  {'R(0,2)':>10}  {'R(1,2)':>10}")
for alpha in ALPHAS:
    r = part_a_results[alpha]["rors"]
    print(f"{alpha:>8.2f}  {r[(0,1)]:>10.6f}  {r[(0,2)]:>10.6f}  {r[(1,2)]:>10.6f}")

# -----------------------------------------------------------------------
# Part (b): optimize budget distribution for fixed total = 9
# -----------------------------------------------------------------------

print("\n" + "=" * 65)
print("PART (b): Optimize B (B_total=9) to maximize |log R_{0,1}|")
print("=" * 65)


def neg_abs_log_ror(b_free, alpha, pair=(0, 1)):
    """
    b_free: [B_0, B_1]; B_2 = B_TOTAL - B_0 - B_1
    Returns -|log R_{pair}| (to minimize = maximize |log R|).
    """
    B0, B1 = b_free
    B2 = B_TOTAL - B0 - B1
    if B0 <= 0 or B1 <= 0 or B2 <= 0:
        return 0.0  # infeasible, return neutral
    B = np.array([B0, B1, B2])
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B, W.copy(), W.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
    Sigma = compute_Sigma(X_nash, f_fn, N, K)
    i, j = pair
    ror = ratio_of_ratios(Sigma, i, j)
    if ror <= 0:
        return 0.0
    return -abs(np.log(ror))


part_b_results = {}

for alpha in ALPHAS:
    print(f"\nalpha = {alpha}")
    # Grid search first, then refine
    best_val = np.inf
    best_b = None
    grid = np.linspace(0.5, B_TOTAL - 1.0, 20)
    for b0 in grid:
        for b1 in grid:
            if b0 + b1 >= B_TOTAL - 0.5:
                continue
            v = neg_abs_log_ror([b0, b1], alpha)
            if v < best_val:
                best_val, best_b = v, [b0, b1]

    # Refine with SLSQP
    def constraint_fn(b):
        return B_TOTAL - b[0] - b[1] - 0.5  # B2 >= 0.5

    res = minimize(
        lambda b: neg_abs_log_ror(b, alpha),
        best_b,
        method="SLSQP",
        bounds=[(0.5, B_TOTAL - 1.0), (0.5, B_TOTAL - 1.0)],
        constraints=[{"type": "ineq", "fun": constraint_fn}],
        options={"ftol": 1e-8, "maxiter": 500},
    )

    B0_opt, B1_opt = res.x
    B2_opt = B_TOTAL - B0_opt - B1_opt
    B_opt = np.array([B0_opt, B1_opt, B2_opt])

    # Recompute everything at optimal budget
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B_opt, W.copy(), W.copy(), f_fn, fp_fn)
    X_nash_opt, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
    Sigma_opt = compute_Sigma(X_nash_opt, f_fn, N, K)
    ror_opt = ratio_of_ratios(Sigma_opt, 0, 1)

    part_b_results[alpha] = {
        "B_opt": B_opt,
        "X_nash_opt": X_nash_opt,
        "Sigma_opt": Sigma_opt,
        "ror_opt": ror_opt,
    }

    print(f"  Optimal B:   {B_opt.round(4)}  (total={B_opt.sum():.2f})")
    print(f"  Nash X at opt B:\n    firm0={X_nash_opt[0].round(4)}  firm1={X_nash_opt[1].round(4)}  firm2={X_nash_opt[2].round(4)}")
    print(f"  Sigma at opt B:\n    firm0={Sigma_opt[0].round(6)}  firm1={Sigma_opt[1].round(6)}")
    print(f"  R_{{0,1}} at opt B = {ror_opt:.6f}  (|log|={abs(np.log(ror_opt)):.6f})")
    print(f"  vs fixed-budget R_{{0,1}} = {part_a_results[alpha]['rors'][(0,1)]:.6f}")

# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(r"$\Sigma$ ratios and ratio-of-ratios vs $\alpha$: $f(x)=x^\alpha$, $N=3$, $K=2$",
             fontsize=12, fontweight="bold")

# Panel 1: Sigma_{i,1}/Sigma_{i,2} for each firm, each alpha
ax = axes[0]
for i, marker in enumerate(["o", "s", "^"]):
    vals = [part_a_results[a]["sigma_ratios"][i] for a in ALPHAS]
    ax.plot(ALPHAS, vals, marker=marker, lw=2, label=f"firm {i} (B={int(B_FIXED[i])})")
ax.axhline(W[0] / W[1], ls="--", color="gray", lw=1, label=f"$w_1/w_2$={W[0]/W[1]:.2f}")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\Sigma_{i,1}/\Sigma_{i,2}$")
ax.set_title(r"Sigma ratio per firm")
ax.legend(fontsize=9)

# Panel 2: Ratio of ratios R_{i,j} for all pairs
ax = axes[1]
pair_labels = ["R(0,1)", "R(0,2)", "R(1,2)"]
pair_keys = [(0, 1), (0, 2), (1, 2)]
for pk, lbl, marker in zip(pair_keys, pair_labels, ["o", "s", "^"]):
    vals = [part_a_results[a]["rors"][pk] for a in ALPHAS]
    ax.plot(ALPHAS, vals, marker=marker, lw=2, label=lbl)
ax.axhline(1.0, ls="--", color="gray", lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("Ratio of ratios")
ax.set_title("Ratio-of-ratios R(i,j) vs alpha")
ax.legend(fontsize=9)

# Panel 3: Optimal budget distributions (part b)
ax = axes[2]
x_pos = np.arange(N)
width = 0.25
for idx, (alpha, color) in enumerate(zip(ALPHAS, COLORS)):
    B_opt = part_b_results[alpha]["B_opt"]
    offset = (idx - 1) * width
    ax.bar(x_pos + offset, B_opt, width=width * 0.9, color=color,
           alpha=0.8, label=rf"$\alpha$={alpha}")
# Also show fixed budget
for xi, b in enumerate(B_FIXED):
    ax.scatter(xi, b, marker="*", s=80, color="black", zorder=5)
ax.set_xticks(list(x_pos)); ax.set_xticklabels([f"firm {i}" for i in range(N)])
ax.set_ylabel("Budget $B_i$")
ax.set_title("Optimal budget (part b)\n★ = fixed budget for part (a)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_sigma_ratio_alpha.pdf", bbox_inches="tight")
plt.savefig("plot_sigma_ratio_alpha.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_sigma_ratio_alpha.pdf / .png")

# -----------------------------------------------------------------------
# Sweep: log R_{0,1} vs log(alpha) to check functional form
# -----------------------------------------------------------------------

alpha_grid = np.linspace(0.1, 0.95, 30)
log_ror_grid = []

print("\nSweeping alpha for functional form of log R_{0,1} ...")
for alpha in alpha_grid:
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B_FIXED.copy(), W.copy(), W.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
    Sigma = compute_Sigma(X_nash, f_fn, N, K)
    ror = ratio_of_ratios(Sigma, 0, 1)
    log_ror_grid.append(np.log(ror) if ror > 0 else np.nan)

log_ror_grid = np.array(log_ror_grid)

# Fit log R vs alpha
valid = np.isfinite(log_ror_grid)
coeffs = np.polyfit(alpha_grid[valid], log_ror_grid[valid], 1)
print(f"  Linear fit: log R_{{0,1}} ≈ {coeffs[0]:.4f} * alpha + {coeffs[1]:.4f}")

# Fit log R vs log(alpha) (power law check)
log_alpha = np.log(alpha_grid[valid])
coeffs_log = np.polyfit(log_alpha, log_ror_grid[valid], 1)
print(f"  Log-log fit: log R_{{0,1}} ≈ {coeffs_log[0]:.4f} * log(alpha) + {coeffs_log[1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(r"Functional form of $\log R_{0,1}$ vs $\alpha$", fontsize=12)

ax = axes[0]
ax.plot(alpha_grid[valid], log_ror_grid[valid], "o", ms=4, color="steelblue")
ax.plot(alpha_grid[valid], np.polyval(coeffs, alpha_grid[valid]), "--", color="firebrick",
        label=f"linear: {coeffs[0]:.3f}α + {coeffs[1]:.3f}")
ax.axhline(0, ls=":", color="gray", lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\log R_{0,1}$")
ax.set_title(r"$\log R_{0,1}$ vs $\alpha$")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(log_alpha, log_ror_grid[valid], "o", ms=4, color="darkorange")
ax.plot(log_alpha, np.polyval(coeffs_log, log_alpha), "--", color="firebrick",
        label=f"log-log: {coeffs_log[0]:.3f}·log(α) + {coeffs_log[1]:.3f}")
ax.axhline(0, ls=":", color="gray", lw=1)
ax.set_xlabel(r"$\log\alpha$"); ax.set_ylabel(r"$\log R_{0,1}$")
ax.set_title(r"Log-log: $\log R$ vs $\log\alpha$")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_sigma_ratio_loglog.pdf", bbox_inches="tight")
plt.savefig("plot_sigma_ratio_loglog.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: plot_sigma_ratio_loglog.pdf / .png")
