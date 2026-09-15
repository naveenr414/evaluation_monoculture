"""
Show that Sigma_{i,k} is nearly constant across w_k while w_k*f'(B_i*w_k)
and the product both vary substantially. Three functions chosen to show this clearly.
Evaluated at proportional point x_{i,k} = B_i * w_k.
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
focal = 0  # richest firm, B=5

w_vals = np.linspace(0.02, 0.6, 200)

FUNCTIONS = [
    ("$\\log(x+100)$", lambda x: math.log(max(x,0)+100),  lambda x: 1/(max(x,0)+100)),
    ("$\\sqrt{x}$",    lambda x: max(x,0)**0.5,            lambda x: 0.5*max(x,1e-12)**(-0.5)),
    ("$1-e^{-x}$",     lambda x: 1-math.exp(-max(x,0)),   lambda x: math.exp(-max(x,0))),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
fig.suptitle(
    r"$\Sigma_{i,k}$ vs $w_k f'(B_i w_k)$ vs product — all normalized to 1 at $w_k^{\min}$"
    "\n"
    r"(shows $\Sigma$ is nearly flat while $g$ and the product vary substantially)",
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
    product    = sigma_vals * g_vals

    # Normalize all three to value at min w_k
    sigma_norm  = sigma_vals / sigma_vals[0]
    g_norm      = g_vals     / g_vals[0]
    product_norm = product   / product[0]

    ax.plot(w_vals, sigma_norm,   color="steelblue",  lw=2,   label=r"$\Sigma_{i,k}$")
    ax.plot(w_vals, g_norm,       color="darkorange",  lw=2, ls="--", label=r"$w_k f'(B_i w_k)$")
    ax.plot(w_vals, product_norm, color="firebrick",   lw=2, ls=":",  label=r"Product")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)

    sig_pct = 100*(sigma_vals.max()/sigma_vals.min() - 1)
    g_pct   = 100*(g_vals.max()/g_vals.min() - 1)
    ax.set_title(f"{label}\n"
                 r"$\Sigma$ varies " + f"{sig_pct:.1f}%,  " + r"$g$ varies " + f"{g_pct:.0f}%",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("$w_k$", fontsize=11)
    if col == 0:
        ax.set_ylabel("Normalized value (= 1 at $w_k^{\\min}$)", fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot_sigma_variation.pdf", format="pdf")
plt.close()
print("Saved: plot_sigma_variation.pdf")
