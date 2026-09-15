"""
Plot how Sigma_{i,k} varies as a function of w_k for four production functions.
Sigma_{i,k} = sum_{i'!=i} sigma'(f(B_i w_k) - f(B_{i'} w_k))

Fixed firm i=0 (richest), fixed B values, sweep w_k.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def sigma_prime(z):
    s = expit(z)
    return s * (1 - s)

B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
N = len(B)
focal = 0   # firm 0, B=5

w_vals = np.linspace(0.02, 0.6, 200)

FUNCTIONS = [
    ("$\\log(x+1)$",   lambda x: math.log(max(x,0)+1),       lambda x: 1/(max(x,0)+1)),
    ("$\\log(x+100)$", lambda x: math.log(max(x,0)+100),     lambda x: 1/(max(x,0)+100)),
    ("$x^{0.5}$",      lambda x: max(x,0)**0.5,               lambda x: 0.5*max(x,1e-12)**(-0.5)),
    ("$x^{0.99}$",     lambda x: max(x,1e-12)**0.99,          lambda x: 0.99*max(x,1e-12)**(-0.01)),
    ("$1-e^{-x}$",     lambda x: 1-math.exp(-max(x,0)),       lambda x: math.exp(-max(x,0))),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
fig.suptitle(
    r"$\Sigma_{i,k}$ (blue, left axis) and $w_k f'(B_i w_k)$ (orange, right axis) vs $w_k$"
    "\n(flat lines = water-filling holds exactly)",
    fontsize=11, fontweight="bold"
)

for col, (label, f_fn, fp_fn) in enumerate(FUNCTIONS):
    ax = axes[col]
    sigma_vals, g_vals = [], []
    for w in w_vals:
        sig = sum(sigma_prime(f_fn(B[focal]*w) - f_fn(B[ip]*w))
                  for ip in range(N) if ip != focal)
        sigma_vals.append(sig)
        g_vals.append(w * fp_fn(B[focal] * w))
    sigma_vals = np.array(sigma_vals)
    g_vals     = np.array(g_vals)

    # Normalize both to value at first w_k so they're on the same scale
    sigma_norm = sigma_vals / sigma_vals[0]
    g_norm     = g_vals     / g_vals[0]

    ax.plot(w_vals, sigma_norm, color="steelblue", lw=2, label=r"$\Sigma_{i,k}$ (norm)")
    ax.plot(w_vals, g_norm,     color="darkorange", lw=2, ls="--", label=r"$w_k f'(B_i w_k)$ (norm)")
    ax.axhline(1.0, ls=":", color="gray", lw=1)

    sig_range = sigma_vals.max() - sigma_vals.min()
    g_range   = g_vals.max()     - g_vals.min()
    ax.set_title(f"{label}\n$\\Sigma$ range={sig_range:.4f},  $g$ range={g_range:.4f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("$w_k$", fontsize=10)
    ax.set_ylabel("Normalized value (=1 at $w_k$=min)" if col == 0 else "", fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_sigma_variation.png", dpi=130)
plt.close()
print("Saved: plot_sigma_variation.png")
