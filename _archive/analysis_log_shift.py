"""
Check whether Nash equilibrium allocations satisfy x_{i,k} = B_i * w_k
for four log-variant production functions:
  1. log(x+1)
  2. log(x+2)
  3. log(2x+1)
  4. 2*log(x+1)

Diagnostic: if x_{i,k} = B_i * w_k, then x_{i,k} / w_k should be constant = B_i across k.
We plot x_{i,k} / w_k vs w_k for each firm to visualize any deviation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from competition_simulation import CompetitionSimulation

RNG = np.random.default_rng(42)
N, K = 3, 5
N_SEEDS = 6

FUNCTIONS = [
    ("log(x+1)",   lambda x: math.log(max(x, 0) + 1),      lambda x: 1.0 / (max(x, 0) + 1)),
    ("log(x+2)",   lambda x: math.log(max(x, 0) + 2),      lambda x: 1.0 / (max(x, 0) + 2)),
    ("log(2x+1)",  lambda x: math.log(2 * max(x, 0) + 1),  lambda x: 2.0 / (2 * max(x, 0) + 1)),
    ("2*log(x+1)", lambda x: 2 * math.log(max(x, 0) + 1),  lambda x: 2.0 / (max(x, 0) + 1)),
]

def sample_instance(seed):
    rng = np.random.default_rng(seed * 17 + 3)
    B = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
    w = rng.dirichlet(np.ones(K))
    return B, w

fig, axes = plt.subplots(N_SEEDS, len(FUNCTIONS), figsize=(16, 3 * N_SEEDS), squeeze=False)

for seed_idx in range(N_SEEDS):
    B, w = sample_instance(seed_idx)
    for col, (label, f_fn, fp_fn) in enumerate(FUNCTIONS):
        ax = axes[seed_idx, col]
        sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
        X, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)

        for i in range(N):
            # ratio x_{i,k} / w_k — should be B_i if proportional
            ratio = X[i] / (w + 1e-12)
            ax.scatter(w, ratio, label=f"Firm {i} (B={B[i]:.2f})", zorder=3)
            ax.axhline(B[i], ls="--", lw=0.8, alpha=0.5)

        ax.set_xlabel("$w_k$", fontsize=8)
        ax.set_ylabel("$x_{i,k}/w_k$", fontsize=8)
        if seed_idx == 0:
            ax.set_title(label, fontsize=10, fontweight="bold")
        if col == 0 and seed_idx == 0:
            ax.legend(fontsize=7)

fig.suptitle(
    "Nash equilibrium: $x_{i,k}/w_k$ vs $w_k$\n"
    "(flat line at $B_i$ = pure proportional; slope = shift away from $B_i w_k$)",
    fontsize=11
)
plt.tight_layout()
plt.savefig("plot_log_shift.png", dpi=130)
plt.close()
print("Saved: plot_log_shift.png")

# ---- Also print numerical summary ----
print("\n=== Numerical check: is x_{i,k} proportional to B_i * w_k? ===")
seed = 0
B, w = sample_instance(seed)
print(f"B = {np.round(B, 3)},  w = {np.round(w, 3)}\n")

for label, f_fn, fp_fn in FUNCTIONS:
    sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)
    print(f"--- {label} ---")
    for i in range(N):
        ratio = X[i] / (w + 1e-12)
        expected = B[i]
        deviation = ratio - expected
        print(f"  Firm {i} (B={B[i]:.3f}):  x/w = {np.round(ratio, 3)}  "
              f"(expected {expected:.3f} if proportional, deviation = {np.round(deviation, 3)})")
    # Check: is x_{i,k} - B_i*w_k a constant (shift) across k?
    print("  Shift x_{i,k} - B_i*w_k per firm:")
    for i in range(N):
        shift = X[i] - B[i] * w
        print(f"    Firm {i}: {np.round(shift, 4)}")
    print()
