"""
For f(x) = 1-e^{-x}, investigate:
a) Do firms invest more in large or small w_k dims (relative to proportional)?
b) Are top or bottom budget firms more uniform across dims?
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from competition_simulation import CompetitionSimulation

N, K = 5, 6
B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
w = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
w = w / w.sum()

f_fn  = lambda x: 1 - math.exp(-max(x, 0))
fp_fn = lambda x: math.exp(-max(x, 0))

sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
X, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)

print(f"w (sorted desc) = {np.round(w, 3)}")
print()

# a) Investment relative to proportional: x_{i,k} / (B_i * w_k)
# If > 1: over-invest relative to proportional (tilts toward this dim)
# If < 1: under-invest
print("=== (a) x_{i,k} / (B_i * w_k) — ratio to proportional ===")
print(f"{'':>10}", end="")
for k in range(K):
    print(f"  w={w[k]:.2f}", end="")
print()
for i in range(N):
    ratio = X[i] / (B[i] * w)
    print(f"Firm {i+1} B={B[i]:.0f}:  {np.round(ratio, 3)}")

print()

# b) Uniformity: std of x_{i,k}/B_i across k (lower = more uniform)
print("=== (b) Budget share std across dims (lower = more uniform) ===")
for i in range(N):
    share = X[i] / B[i]
    print(f"Firm {i+1} (B={B[i]:.0f}):  shares={np.round(share,3)}  std={np.std(share):.4f}")

print()

# Plot: investment share vs w_k for each firm
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("$f(x) = 1-e^{-x}$: investment patterns", fontsize=12, fontweight="bold")

colors = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "firebrick"]

# Panel a: x_{i,k}/(B_i w_k) vs w_k
ax = axes[0]
for i in range(N):
    ratio = X[i] / (B[i] * w)
    ax.plot(w, ratio, "o-", color=colors[i], lw=2, ms=6, label=f"Firm {i+1} (B={int(B[i])})")
ax.axhline(1.0, ls="--", color="gray", lw=1, label="Proportional (=1)")
ax.set_xlabel("$w_k$", fontsize=11)
ax.set_ylabel("$x_{i,k} / (B_i w_k)$", fontsize=11)
ax.set_title("(a) Investment relative to proportional\n>1 = over-invest, <1 = under-invest", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

# Panel b: budget share x_{i,k}/B_i vs w_k
ax = axes[1]
for i in range(N):
    share = X[i] / B[i]
    ax.plot(w, share, "o-", color=colors[i], lw=2, ms=6, label=f"Firm {i+1} (B={int(B[i])})")
ax.plot(w, w, "k--", lw=1.5, label="Proportional ($w_k$)")
ax.set_xlabel("$w_k$", fontsize=11)
ax.set_ylabel("$x_{i,k} / B_i$  (budget share)", fontsize=11)
ax.set_title("(b) Budget share per dim\n(how uniform across dims?)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_saturating.png", dpi=130)
plt.close()
print("Saved: plot_saturating.png")
