"""
Plot the decomposition of the marginal ratio (top dim vs bottom dim) into:
  1. g-ratio:     g(w_h)/g(w_l)  where g(w) = w/(B_i w + a)
  2. Sigma-ratio: Sigma_{i,h} / Sigma_{i,l}
  3. Product:     total marginal ratio

For fixed w_h, w_l, B values, as a varies from 0 to 20.
One set of lines per firm (B_i value).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

N = 5
B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

w_h = 0.25
w_l = 0.10

a_vals = np.linspace(0.0, 20.0, 200)

def sigma_prime(z):
    s = expit(z)
    return s * (1 - s)

def g_ratio(Bi, a):
    return (w_h / (Bi * w_h + a)) / (w_l / (Bi * w_l + a) + 1e-12)

def sigma_ratio(i, a):
    Bi = B[i]
    sig_h = sum(sigma_prime(np.log((Bi * w_h + a) / (B[ip] * w_h + a)))
                for ip in range(N) if ip != i)
    sig_l = sum(sigma_prime(np.log((Bi * w_l + a) / (B[ip] * w_l + a)))
                for ip in range(N) if ip != i)
    return sig_h / (sig_l + 1e-12)

colors = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "firebrick"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    rf"Marginal ratio decomposition: top dim ($w_h$={w_h}) vs bottom dim ($w_l$={w_l})",
    fontsize=12, fontweight="bold"
)

titles = [
    r"(1) $g$-ratio: $\frac{w_h/(B_i w_h+a)}{w_l/(B_i w_l+a)}$",
    r"(2) $\Sigma$-ratio: $\frac{\Sigma_{i,h}}{\Sigma_{i,l}}$",
    r"(3) Product: total marginal ratio",
]

for col in range(3):
    ax = axes[col]
    for i in range(N):
        gr  = np.array([g_ratio(B[i], a) for a in a_vals])
        sr  = np.array([sigma_ratio(i, a) for a in a_vals])
        if col == 0:
            y = gr
        elif col == 1:
            y = sr
        else:
            y = gr * sr
        ax.plot(a_vals, y, color=colors[i], lw=2, label=f"Firm {i+1} (B={int(B[i])})")

    ax.axhline(1.0, ls="--", color="gray", lw=1)
    if col == 0:
        ax.axhline(w_h / w_l, ls=":", color="gray", lw=1, label=f"$w_h/w_l$ = {w_h/w_l:.1f}")
    ax.set_xlabel("$a$", fontsize=11)
    ax.set_title(titles[col], fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_ratio_decomp.png", dpi=130)
plt.close()
print("Saved: plot_ratio_decomp.png")
