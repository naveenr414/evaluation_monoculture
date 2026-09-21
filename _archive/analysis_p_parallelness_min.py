"""
Same as analysis_p_parallelness.py but using the MINIMUM pairwise cosine
similarity across firms (worst-case pair) instead of the average, per
budget config and p. This asks: is there ANY pair of firms whose theta
vectors are not parallel, for a given p?
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
EPS = 1e-6

BUDGETS = {
    "Moderate hetero":      [5.0, 3.0, 1.0],
    "Extreme hetero":       [7.5, 1.2, 0.3],
    "Structure A (2 tiny)": [0.05, 0.05, B_TOTAL - 0.1],
    "Structure B (2 big)":  [4.4, 4.3, 0.3],
}

P_VALUES = np.concatenate([np.linspace(0.02, 0.9, 30), [0.95, 0.98]])


def make_power(p):
    f = lambda x: np.clip(x, EPS, None) ** p
    fp = lambda x: p * np.clip(x, EPS, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def min_pairwise_cos(theta):
    sims = [cos_sim(theta[i], theta[j]) for i, j in combinations(range(N), 2)]
    return min(sims)


overall_min = []
per_config = {name: [] for name in BUDGETS}

for p in P_VALUES:
    f, fp = make_power(p)
    cos_vals = []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        c = min_pairwise_cos(theta)
        cos_vals.append(c)
        per_config[name].append(c)
    overall_min.append(min(cos_vals))

overall_min = np.array(overall_min)

fig, ax = plt.subplots(figsize=(7, 5))
for name in BUDGETS:
    ax.plot(P_VALUES, per_config[name], marker="o", ms=3, lw=1, alpha=0.5, label=name)
ax.plot(P_VALUES, overall_min, marker="o", ms=5, lw=2.5, color="black",
        label="worst case across all configs")

ax.set_xlabel(r"$p$  (in $f(x) = x^p$)", fontsize=12)
ax.set_ylabel(r"min pairwise cosine similarity of $\theta_i = f(x_i)$ vectors", fontsize=11)
ax.set_title(r"Worst-case output ($\theta$) parallelness across firms vs. $p$", fontsize=12, fontweight="bold")
ax.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.6)
ax.axhline(0.0, ls=":", color="red", lw=1, alpha=0.5)
ax.legend(fontsize=8, loc="lower left")
ax.set_ylim(top=1.02)

plt.tight_layout()
plt.savefig("plot_p_parallelness_min.pdf", bbox_inches="tight")
plt.savefig("plot_p_parallelness_min.png", dpi=130, bbox_inches="tight")
plt.close()

print("Saved plot_p_parallelness_min.pdf / .png")
print("\np, worst-case (min) cosine similarity across all configs:")
for p, c in zip(P_VALUES, overall_min):
    print(f"  p={p:.3f}  min_cos_sim={c:.5f}")
