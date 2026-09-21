"""
Plot the decomposition of the marginal ratio (top dim vs bottom dim) as a
function of B_i (firm budget), for fixed a values.

For each B_i, evaluated at proportional point x_{i,k} = B_i * w_k:
  1. g-ratio:     (w_h / (B_i w_h + a)) / (w_l / (B_i w_l + a))
  2. Sigma-ratio: Sigma_{i,h} / Sigma_{i,l}
  3. Product:     total marginal ratio

Other firms' budgets fixed at B_others. Sweep B_i continuously.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

w_h = 0.25
w_l = 0.10

# Fix other firms' budgets; sweep focal firm's budget
B_others = np.array([4.0, 3.0, 2.0, 1.0])   # N-1 = 4 other firms
B_i_vals = np.linspace(0.5, 8.0, 200)

a_vals  = [0.5, 1.0, 3.0, 8.0]
colors  = ["steelblue", "darkorange", "mediumseagreen", "firebrick"]

def sigma_prime(z):
    s = expit(z)
    return s * (1 - s)

def g_ratio(Bi, a):
    return (w_h / (Bi * w_h + a)) / (w_l / (Bi * w_l + a))

def sigma_ratio(Bi, a):
    sig_h = sum(sigma_prime(np.log((Bi * w_h + a) / (Bip * w_h + a))) for Bip in B_others)
    sig_l = sum(sigma_prime(np.log((Bi * w_l + a) / (Bip * w_l + a))) for Bip in B_others)
    return sig_h / (sig_l + 1e-12)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    rf"Marginal ratio vs firm budget $B_i$: top ($w_h$={w_h}) vs bottom ($w_l$={w_l}) dim",
    fontsize=12, fontweight="bold"
)

titles = [
    r"(1) $g$-ratio: $\frac{w_h/(B_i w_h+a)}{w_l/(B_i w_l+a)}$",
    r"(2) $\Sigma$-ratio: $\frac{\Sigma_{i,h}}{\Sigma_{i,l}}$",
    r"(3) Product: total marginal ratio",
]

for col in range(3):
    ax = axes[col]
    for a, color in zip(a_vals, colors):
        gr = np.array([g_ratio(Bi, a) for Bi in B_i_vals])
        sr = np.array([sigma_ratio(Bi, a) for Bi in B_i_vals])
        if col == 0:
            y = gr
        elif col == 1:
            y = sr
        else:
            y = gr * sr
        ax.plot(B_i_vals, y, color=color, lw=2, label=f"$a$={a}")

    ax.axhline(1.0, ls="--", color="gray", lw=1)
    if col == 0:
        ax.axhline(w_h / w_l, ls=":", color="gray", lw=1, label=f"$w_h/w_l$={w_h/w_l:.1f}")
    ax.set_xlabel("$B_i$  (focal firm budget)", fontsize=11)
    ax.set_title(titles[col], fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_ratio_budget.png", dpi=130)
plt.close()
print("Saved: plot_ratio_budget.png")
