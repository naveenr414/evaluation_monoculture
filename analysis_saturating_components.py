"""
For f(x) = 1-e^{-x}, at the Nash equilibrium allocation X*,
plot the two KKT components for each firm across dims:
  1. w_k * f'(x_{i,k})  =  w_k * exp(-x_{i,k})
  2. Sigma_{i,k}        =  sum_{i'!=i} sigma'(f(x_{i,k}) - f(x_{i',k}))
  3. Product (= marginal return, should be equal across active dims at Nash)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from competition_simulation import CompetitionSimulation

N, K = 5, 6
B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
w = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
w = w / w.sum()

f_fn  = lambda x: 1 - math.exp(-max(x, 0))
fp_fn = lambda x: math.exp(-max(x, 0))

def sigma_prime(z):
    s = expit(z)
    return s * (1 - s)

# Evaluate at proportional point x_{i,k} = B_i * w_k (not Nash)
X = np.array([[B[i] * w[k] for k in range(K)] for i in range(N)])

# Compute components at proportional point
colors = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "firebrick"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(r"KKT components at proportional point $x_{i,k}=B_i w_k$: $f(x)=1-e^{-x}$"
             "\n(flat = same as log solution; deviation = drive away from proportional)",
             fontsize=11, fontweight="bold")

titles = [r"(1) $w_k \cdot f'(x_{i,k})$", r"(2) $\Sigma_{i,k}$", r"(3) Product (marginal return)"]

for col in range(3):
    ax = axes[col]
    for i in range(N):
        gvals   = np.array([w[k] * fp_fn(X[i, k]) for k in range(K)])
        sigvals = np.array([
            sum(sigma_prime(f_fn(X[i,k]) - f_fn(X[ip,k])) for ip in range(N) if ip != i)
            for k in range(K)
        ])
        product = gvals * sigvals

        if col == 0:
            y = gvals
        elif col == 1:
            y = sigvals
        else:
            y = product

        ax.plot(w, y, "o-", color=colors[i], lw=2, ms=6,
                label=f"Firm {i+1} (B={int(B[i])})")

    ax.set_xlabel("$w_k$", fontsize=11)
    ax.set_title(titles[col], fontsize=10, fontweight="bold")
    if col == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_saturating_components.png", dpi=130)
plt.close()
print("Saved: plot_saturating_components.png")
