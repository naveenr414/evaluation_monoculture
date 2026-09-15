"""
f(x) = x^p, sweep p in (0, 1]. As p -> 0, x^p behaves like an affine
transform of log(x)  (since (x^p - 1)/p -> log(x)), so we expect p->0 to
approach the log(x) behavior (homogeneous theta-DIFFERENCES, heterogeneous
normalized theta), and p->1 (linear f(x)=x) to be the opposite extreme.
sqrt(x) is the midpoint p=0.5.

We track two separate homogeneity metrics across firms, for each p:
  (1) spread of theta_1 - theta_2 across firms  (the log-invariant quantity)
  (2) spread of normalized theta (theta/||theta||) across firms
      -- measured as 1 - min pairwise cosine similarity (0 = identical shape)

Lower value = more homogeneous under that metric.
"""

import numpy as np
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

P_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def make_power(p):
    f = lambda x: np.clip(x, EPS, None) ** p
    fp = lambda x: p * np.clip(x, EPS, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def min_pairwise_cos(theta):
    sims = []
    for i in range(N):
        for j in range(i + 1, N):
            sims.append(cos_sim(theta[i], theta[j]))
    return min(sims)


results = {name: {"diff_spread": [], "norm_het": []} for name in BUDGETS}

for name, B in BUDGETS.items():
    for p in P_VALUES:
        f, fp = make_power(p)
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        diff = theta[:, 0] - theta[:, 1]
        diff_spread = diff.max() - diff.min()
        norm_het = 1.0 - min_pairwise_cos(theta)   # 0 = perfectly homogeneous shape
        results[name]["diff_spread"].append(diff_spread)
        results[name]["norm_het"].append(norm_het)

print(f"{'p':>6}", end="")
for name in BUDGETS:
    print(f" | {name+' diffSpread':>26} {name+' normHet':>20}", end="")
print()

for k, p in enumerate(P_VALUES):
    print(f"{p:6.2f}", end="")
    for name in BUDGETS:
        ds = results[name]["diff_spread"][k]
        nh = results[name]["norm_het"][k]
        print(f" | {ds:26.5f} {nh:20.5f}", end="")
    print()

print("\nAveraged across budget configs:")
print(f"{'p':>6} {'avg diff_spread':>18} {'avg norm_het':>16}")
for k, p in enumerate(P_VALUES):
    avg_ds = np.mean([results[name]["diff_spread"][k] for name in BUDGETS])
    avg_nh = np.mean([results[name]["norm_het"][k] for name in BUDGETS])
    print(f"{p:6.2f} {avg_ds:18.5f} {avg_nh:16.5f}")
