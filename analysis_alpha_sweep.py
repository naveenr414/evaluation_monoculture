"""
Sweep alpha in f(x) = x^alpha. For each alpha, compute:
  1. Violation rate: fraction of (i,j,k) pairs where B_i > B_j but x_{i,k} < x_{j,k}
  2. Corner rate: fraction of firms that concentrate >=95% of budget in one dim
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from competition_simulation import CompetitionSimulation

N, K = 4, 5
N_CONFIGS = 30
RNG = np.random.default_rng(42)

alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

violation_rates = []
corner_rates    = []

for alpha in alphas:
    f_fn  = lambda x, a=alpha: max(x, 1e-12)**a
    fp_fn = lambda x, a=alpha: a * max(x, 1e-12)**(a-1)

    total_pairs, violations = 0, 0
    total_firms, corners    = 0, 0

    for cfg in range(N_CONFIGS):
        B = np.sort(RNG.uniform(0.5, 5.0, size=N))[::-1]
        w = np.sort(RNG.dirichlet(np.ones(K)))[::-1]
        sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
        X, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)

        for i in range(N):
            for j in range(i+1, N):
                for k in range(K):
                    total_pairs += 1
                    if X[i,k] < X[j,k] - 1e-4:
                        violations += 1

        for i in range(N):
            total_firms += 1
            top_share = X[i].max() / (B[i] + 1e-12)
            if top_share >= 0.95:
                corners += 1

    violation_rates.append(violations / total_pairs)
    corner_rates.append(corners / total_firms)
    print(f"alpha={alpha:.2f}  viol={violation_rates[-1]:.3f}  corner={corner_rates[-1]:.3f}", flush=True)

violation_rates = np.array(violation_rates)
corner_rates    = np.array(corner_rates)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(r"Effect of $\alpha$ in $f(x)=x^\alpha$ on dominance and specialisation",
             fontsize=12, fontweight="bold")

# Plot 1: violation rate
axes[0].plot(alphas, violation_rates, "o-", color="firebrick", lw=2, ms=6)
axes[0].set_xlabel(r"$\alpha$", fontsize=11)
axes[0].set_ylabel("Violation rate", fontsize=10)
axes[0].set_title("Dominance violation rate\n(fraction of pairs where $B_i>B_j$ but $x_{i,k}<x_{j,k}$)",
                  fontsize=9, fontweight="bold")
axes[0].set_ylim(0, None)

# Plot 2: corner rate
axes[1].plot(alphas, corner_rates, "s-", color="steelblue", lw=2, ms=6)
axes[1].set_xlabel(r"$\alpha$", fontsize=11)
axes[1].set_ylabel("Corner rate", fontsize=10)
axes[1].set_title("Corner rate\n(fraction of firms with ≥95% budget in one dim)",
                  fontsize=9, fontweight="bold")
axes[1].set_ylim(0, 1.05)

# Plot 3: both overlaid (normalised)
ax3 = axes[2]
ax3.plot(alphas, violation_rates / (violation_rates.max() + 1e-9),
         "o-", color="firebrick", lw=2, ms=6, label="Violation rate (norm)")
ax3.plot(alphas, corner_rates,
         "s-", color="steelblue", lw=2, ms=6, label="Corner rate")
ax3.set_xlabel(r"$\alpha$", fontsize=11)
ax3.set_title("Overlay (violation normalised to [0,1])\nare they correlated?",
              fontsize=9, fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("plot_alpha_sweep.png", dpi=130)
plt.close()
print("\nSaved: plot_alpha_sweep.png")
