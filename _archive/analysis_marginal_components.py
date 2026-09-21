"""
Empirically compute the two components of the marginal return at the
proportional point x_{i,k} = B_i * w_k, as b increases in f(x) = log(x+b).

For each b, at x_{i,k} = B_i * w_k:
  - f'(x_{i,k}) = 1 / (B_i * w_k + b)
  - Sigma_{i,k} = sum_{i'!=i} sigma'(log(B_i*w_k + b) - log(B_{i'}*w_k + b))
  - Marginal = w_k * f'(x_{i,k}) * Sigma_{i,k}

We track how each component varies across k (high vs low w_k dim),
for the richest and poorest firm.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

N, K = 5, 6
B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
w = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
w = w / w.sum()

b_vals = np.linspace(0.01, 20.0, 80)

def sigma_prime(z):
    s = expit(z)
    return s * (1 - s)

# For each b, compute f', Sigma, and marginal at proportional point
# Focus on: firm 0 (richest, B=5) and firm 4 (poorest, B=1)
# And dims: k=0 (top, w=0.25) and k=5 (bottom, w=0.10)

results = {
    "fp_rich_top":    [], "fp_rich_bot":    [],
    "fp_poor_top":    [], "fp_poor_bot":    [],
    "sig_rich_top":   [], "sig_rich_bot":   [],
    "sig_poor_top":   [], "sig_poor_bot":   [],
    "marg_rich_top":  [], "marg_rich_bot":  [],
    "marg_poor_top":  [], "marg_poor_bot":  [],
    "sig_variation_rich": [],   # std of Sigma_{i,k} across k for firm 0
    "sig_variation_poor": [],
    "fp_variation_rich":  [],   # std of f'(x_{i,k})*w_k across k for firm 0
    "fp_variation_poor":  [],
}

for b in b_vals:
    for firm_idx, firm_label in [(0, "rich"), (4, "poor")]:
        Bi = B[firm_idx]
        fp_vals = []
        sig_vals = []
        marg_vals = []
        for k in range(K):
            x_ik = Bi * w[k]
            fp = 1.0 / (x_ik + b)
            # Sigma_{i,k}
            sig = 0.0
            for ip in range(N):
                if ip != firm_idx:
                    x_ipk = B[ip] * w[k]
                    diff = math.log(x_ik + b) - math.log(x_ipk + b)
                    sig += sigma_prime(diff)
            marg = w[k] * fp * sig
            fp_vals.append(fp)
            sig_vals.append(sig)
            marg_vals.append(marg)

        fp_arr   = np.array(fp_vals)
        sig_arr  = np.array(sig_vals)
        marg_arr = np.array(marg_vals)

        results[f"fp_{firm_label}_top"].append(fp_arr[0])
        results[f"fp_{firm_label}_bot"].append(fp_arr[-1])
        results[f"sig_{firm_label}_top"].append(sig_arr[0])
        results[f"sig_{firm_label}_bot"].append(sig_arr[-1])
        results[f"marg_{firm_label}_top"].append(marg_arr[0])
        results[f"marg_{firm_label}_bot"].append(marg_arr[-1])
        results[f"sig_variation_{firm_label}"].append(np.std(sig_arr))
        results[f"fp_variation_{firm_label}"].append(np.std(fp_arr * w))

results = {k: np.array(v) for k, v in results.items()}

# ---------------------------------------------------------------------------
# Figure: 3 rows x 2 cols
#   Row 1: f'(x_{i,k}) for top vs bot dim  (rich | poor)
#   Row 2: Sigma_{i,k} for top vs bot dim  (rich | poor)
#   Row 3: Marginal = w_k f' Sigma          (rich | poor)
# Plus a 4th row: variation (std across k) of Sigma and f'*w_k
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 14),
                         sharex=True, gridspec_kw={"hspace": 0.45, "wspace": 0.3})

firm_labels = ["Richest firm (B=5)", "Poorest firm (B=1)"]
firm_keys   = ["rich", "poor"]
dim_colors  = {"top": "steelblue", "bot": "firebrick"}

for col, (flabel, fkey) in enumerate(zip(firm_labels, firm_keys)):
    # Row 0: f'
    ax = axes[0, col]
    ax.plot(b_vals, results[f"fp_{fkey}_top"], color="steelblue", lw=2, label=f"top dim ($w$={w[0]:.2f})")
    ax.plot(b_vals, results[f"fp_{fkey}_bot"], color="firebrick", lw=2, label=f"bot dim ($w$={w[-1]:.2f})")
    ax.set_ylabel("$f'(x_{i,k})$", fontsize=10)
    ax.set_title(f"$f'$ — {flabel}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Row 1: Sigma
    ax = axes[1, col]
    ax.plot(b_vals, results[f"sig_{fkey}_top"], color="steelblue", lw=2, label=f"top dim")
    ax.plot(b_vals, results[f"sig_{fkey}_bot"], color="firebrick", lw=2, label=f"bot dim")
    ax.axhline((N-1)/4, ls="--", color="gray", lw=1, label=f"$(N-1)/4 = {(N-1)/4:.2f}$")
    ax.set_ylabel(r"$\Sigma_{i,k}$", fontsize=10)
    ax.set_title(rf"$\Sigma_{{i,k}}$ — {flabel}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Row 2: Marginal
    ax = axes[2, col]
    ax.plot(b_vals, results[f"marg_{fkey}_top"], color="steelblue", lw=2, label=f"top dim")
    ax.plot(b_vals, results[f"marg_{fkey}_bot"], color="firebrick", lw=2, label=f"bot dim")
    ax.set_ylabel(r"$w_k \cdot f' \cdot \Sigma_{i,k}$", fontsize=10)
    ax.set_title(f"Marginal — {flabel}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Row 3: variation across k
    ax = axes[3, col]
    ax.plot(b_vals, results[f"sig_variation_{fkey}"],  color="darkorange",     lw=2, label=r"std of $\Sigma_{i,k}$ across $k$")
    ax.plot(b_vals, results[f"fp_variation_{fkey}"],   color="mediumseagreen", lw=2, label=r"std of $w_k f'(x_{i,k})$ across $k$")
    ax.set_ylabel("Std across dims", fontsize=10)
    ax.set_title(f"Variation across $k$ — {flabel}", fontsize=10, fontweight="bold")
    ax.set_xlabel("b", fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle(
    r"Components of marginal return $w_k f'(x_{i,k}) \Sigma_{i,k}$ at proportional point $x_{i,k} = B_i w_k$",
    fontsize=12, fontweight="bold"
)

plt.savefig("plot_marginal_components.png", dpi=130)
plt.close()
print("Saved: plot_marginal_components.png")
