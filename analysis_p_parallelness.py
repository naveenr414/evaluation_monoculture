"""
Plot: p (in f(x)=x^p) vs average pairwise "parallelness" (cosine similarity)
between firms' theta vectors, averaged across several heterogeneous-budget
configurations and all firm pairs.
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


def avg_pairwise_cos(theta):
    sims = [cos_sim(theta[i], theta[j]) for i, j in combinations(range(N), 2)]
    return np.mean(sims)


avg_parallelness = []
per_config = {name: [] for name in BUDGETS}

for p in P_VALUES:
    f, fp = make_power(p)
    cos_vals = []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        c = avg_pairwise_cos(theta)
        cos_vals.append(c)
        per_config[name].append(c)
    avg_parallelness.append(np.mean(cos_vals))

avg_parallelness = np.array(avg_parallelness)

fig, ax = plt.subplots(figsize=(7, 5))
for name in BUDGETS:
    ax.plot(P_VALUES, per_config[name], marker="o", ms=3, lw=1, alpha=0.5, label=name)
ax.plot(P_VALUES, avg_parallelness, marker="o", ms=5, lw=2.5, color="black",
        label="average across configs")

ax.set_xlabel(r"$p$  (in $f(x) = x^p$)", fontsize=12)
ax.set_ylabel(r"avg pairwise cosine similarity of $\theta_i = f(x_i)$ vectors", fontsize=11)
ax.set_title(r"Output ($\theta$) parallelness across firms vs. $p$", fontsize=12, fontweight="bold")
ax.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.6)
ax.legend(fontsize=8, loc="lower left")
ax.set_ylim(top=1.02)

plt.tight_layout()
plt.savefig("plot_p_parallelness.pdf", bbox_inches="tight")
plt.savefig("plot_p_parallelness.png", dpi=130, bbox_inches="tight")
plt.close()

print("Saved plot_p_parallelness.pdf / .png")
print("\np, avg cosine similarity:")
for p, c in zip(P_VALUES, avg_parallelness):
    print(f"  p={p:.3f}  cos_sim={c:.5f}")
