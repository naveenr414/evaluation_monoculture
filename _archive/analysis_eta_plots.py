"""
Two-panel figure exploring η_f(x) = -x f''(x)/f'(x) as the driver of heterogeneity.

Plot A: Line plot of heterogeneity (cosine sim, tilt) vs η_f value.
         Power laws give constant η_f = 1−α, so sweep α to get a clean curve.
         Saturating functions overlaid as separate markers (mean η_f at equilibrium).

Plot B: KDE subplots — allocation share x_{i,k}/B_i for high vs low budget firms,
         for four power-law values of η_f (constant across x → cleanest comparison).
         Each subplot = one η_f level; two KDEs = richest vs poorest firm.
"""

import math, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from competition_simulation import CompetitionSimulation

RNG = np.random.default_rng(42)
N, K = 4, 6
N_SEEDS = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_instance(seed):
    rng = np.random.default_rng(seed * 17 + 3)
    B = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
    w = rng.dirichlet(np.ones(K))
    return B, w

def pairwise_cosine(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    C = Xn @ Xn.T
    n = X.shape[0]
    vals = [C[i, j] for i in range(n) for j in range(i+1, n)]
    return np.mean(vals)

def mean_tilt_asymmetry(X, w, B):
    """mean(KL(share_small||w)) - mean(KL(share_large||w)) across seeds."""
    shares = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    kls = []
    for i in range(len(X)):
        p = np.maximum(shares[i], 1e-12)
        q = np.maximum(w, 1e-12)
        kls.append(np.sum(p * np.log(p / q)))
    return kls  # ordered large→small budget

def run_nash(f_fn, fp_fn, B, w):
    sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
    return X

def eta_f_power(alpha, x):
    return 1.0 - alpha   # constant for power laws

def eta_f_exp(k, x):
    return k * x          # = kx for 1-exp(-kx)

# ---------------------------------------------------------------------------
# Plot A data: sweep α for power laws
# ---------------------------------------------------------------------------

alphas = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
eta_power = 1.0 - alphas   # constant η_f for each α

cos_power, tilt_asym_power = [], []

print("Sweeping power laws ...")
for alpha in alphas:
    f_fn  = lambda x, a=alpha: max(x, 0.0)**a
    fp_fn = lambda x, a=alpha: a * max(x, 1e-12)**(a-1)

    seed_cos, seed_tilts = [], []
    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        X = run_nash(f_fn, fp_fn, B, w)
        seed_cos.append(pairwise_cosine(X))
        kls = mean_tilt_asymmetry(X, w, B)
        seed_tilts.append(kls[-1] - kls[0])   # small − large

    cos_power.append(np.mean(seed_cos))
    tilt_asym_power.append(np.mean(seed_tilts))
    print(f"  α={alpha:.2f}  η={1-alpha:.2f}  cos={cos_power[-1]:.4f}  asym={tilt_asym_power[-1]:.4f}")

# Saturating functions: compute mean η_f at equilibrium investment level
sat_specs = [
    ("$1-e^{-0.5x}$", lambda x: 1-math.exp(-0.5*max(x,0)), lambda x: 0.5*math.exp(-0.5*max(x,0)), 0.5),
    ("$1-e^{-1.0x}$", lambda x: 1-math.exp(-1.0*max(x,0)), lambda x: 1.0*math.exp(-1.0*max(x,0)), 1.0),
]

sat_results = []
print("\nSaturating functions ...")
for label, f_fn, fp_fn, k_val in sat_specs:
    seed_cos, seed_tilts, seed_etas = [], [], []
    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        X = run_nash(f_fn, fp_fn, B, w)
        seed_cos.append(pairwise_cosine(X))
        kls = mean_tilt_asymmetry(X, w, B)
        seed_tilts.append(kls[-1] - kls[0])
        # mean η_f = k * x at equilibrium investments
        mean_x = X[X > 1e-4].mean()
        seed_etas.append(k_val * mean_x)
    sat_results.append({
        "label": label,
        "cos": np.mean(seed_cos),
        "asym": np.mean(seed_tilts),
        "eta": np.mean(seed_etas),
    })
    print(f"  {label}  η≈{sat_results[-1]['eta']:.2f}  cos={sat_results[-1]['cos']:.4f}  asym={sat_results[-1]['asym']:.4f}")

# ---------------------------------------------------------------------------
# Plot B data: allocation shares for high vs low budget, four η_f levels
# ---------------------------------------------------------------------------

b_alphas  = [0.05, 0.3, 0.6, 0.9]   # η_f = 0.95, 0.70, 0.40, 0.10
b_labels  = [rf"$\eta_f = {1-a:.2f}$  ($\alpha={a}$)" for a in b_alphas]

# Collect shares for firm 0 (largest) and firm N-1 (smallest) across all seeds
shares_high = {a: [] for a in b_alphas}
shares_low  = {a: [] for a in b_alphas}

print("\nGathering allocation shares for Plot B ...")
for alpha in b_alphas:
    f_fn  = lambda x, a=alpha: max(x, 0.0)**a
    fp_fn = lambda x, a=alpha: a * max(x, 1e-12)**(a-1)
    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        X = run_nash(f_fn, fp_fn, B, w)
        # normalise by budget to get shares; also normalise by w so 1.0 = proportional
        share0 = X[0]  / B[0]   # richest firm
        shareN = X[-1] / B[-1]  # poorest firm
        # express as ratio to w_k (so 1 = log-case proportional)
        shares_high[alpha].extend((share0 / (w + 1e-12)).tolist())
        shares_low[alpha].extend((shareN / (w + 1e-12)).tolist())

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.93, bottom=0.08)

# ── Plot A (top row, spanning all 4 columns) ────────────────────────────────
ax_cos  = fig.add_subplot(gs[0, :2])
ax_asym = fig.add_subplot(gs[0, 2:])

# cosine vs η_f (power laws)
ax_cos.plot(eta_power, cos_power, "o-", color="steelblue", lw=2, ms=6, label="Power law $x^\\alpha$")
for r in sat_results:
    ax_cos.scatter(r["eta"], r["cos"], marker="D", s=80, zorder=5,
                   label=r["label"], color="darkorange" if "0.5" in r["label"] else "firebrick")
ax_cos.axvline(1.0, ls="--", color="gray", lw=1, label="log ($\\eta_f=1$)")
ax_cos.set_xlabel(r"$\eta_f$  (elasticity of marginal returns)", fontsize=11)
ax_cos.set_ylabel("Mean pairwise cosine similarity", fontsize=10)
ax_cos.set_title("(A1)  Homogeneity vs $\\eta_f$", fontsize=11, fontweight="bold")
ax_cos.legend(fontsize=8)
ax_cos.set_xlim(0, 1.15)
ax_cos.set_ylim(0.75, 1.005)
ax_cos.invert_xaxis()   # η→0 = near-linear on left, η→1 = log on right

# tilt asymmetry vs η_f
ax_asym.plot(eta_power, tilt_asym_power, "s-", color="mediumseagreen", lw=2, ms=6,
             label="Power law $x^\\alpha$")
for r in sat_results:
    ax_asym.scatter(r["eta"], r["asym"], marker="D", s=80, zorder=5,
                    label=r["label"], color="darkorange" if "0.5" in r["label"] else "firebrick")
ax_asym.axhline(0.0, ls="--", color="gray", lw=1)
ax_asym.axvline(1.0, ls="--", color="gray", lw=1)
ax_asym.set_xlabel(r"$\eta_f$  (elasticity of marginal returns)", fontsize=11)
ax_asym.set_ylabel("KL(small||w) − KL(large||w)", fontsize=10)
ax_asym.set_title("(A2)  Budget Stratification vs $\\eta_f$", fontsize=11, fontweight="bold")
ax_asym.legend(fontsize=8)
ax_asym.set_xlim(0, 1.15)
ax_asym.invert_xaxis()

# ── Plot B (bottom row: one KDE subplot per η_f level) ──────────────────────
kde_colors = {"high": "steelblue", "low": "firebrick"}

for col, (alpha, blabel) in enumerate(zip(b_alphas, b_labels)):
    ax = fig.add_subplot(gs[1, col])
    h_vals = np.array(shares_high[alpha])
    l_vals = np.array(shares_low[alpha])

    # clip to sensible range for KDE
    clip = (0.0, 6.0)
    h_vals = h_vals[(h_vals > clip[0]) & (h_vals < clip[1])]
    l_vals = l_vals[(l_vals > clip[0]) & (l_vals < clip[1])]

    xg = np.linspace(0.0, 5.5, 300)
    kde_h = gaussian_kde(h_vals, bw_method=0.3)
    kde_l = gaussian_kde(l_vals, bw_method=0.3)

    ax.fill_between(xg, kde_h(xg), alpha=0.25, color="steelblue")
    ax.fill_between(xg, kde_l(xg), alpha=0.25, color="firebrick")
    ax.plot(xg, kde_h(xg), color="steelblue", lw=2, label="High budget")
    ax.plot(xg, kde_l(xg), color="firebrick", lw=2, label="Low budget")
    ax.axvline(1.0, ls="--", color="gray", lw=1.2, label="Proportional ($w_k$)")

    ax.set_title(blabel, fontsize=10, fontweight="bold")
    ax.set_xlabel(r"$x_{i,k} / (B_i \cdot w_k)$" + "\n(1 = proportional to $w_k$)", fontsize=8)
    ax.set_ylabel("Density" if col == 0 else "", fontsize=9)
    ax.set_xlim(0, 5)
    if col == 0:
        ax.legend(fontsize=8)

fig.suptitle(
    r"$\eta_f(x) = -x f''(x)/f'(x)$ as the driver of heterogeneity and budget stratification",
    fontsize=12, fontweight="bold"
)

plt.savefig("plot_eta_heterogeneity.png", dpi=140)
plt.close()
print("\nSaved: plot_eta_heterogeneity.png")
