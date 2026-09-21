"""
Sweep over production functions f and measure:
  1. Price of Anarchy: SW_opt / SW_Nash
  2. Allocation heterogeneity: pairwise cosine similarity of x_i vectors
  3. Budget-stratified concentration: where do high vs low budget firms invest?
  4. Omega_f(x) = x f'(x) profile across the investment range

For each f, average results over multiple (B, w=v) draws.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from competition_simulation import CompetitionSimulation

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Production function families
# ---------------------------------------------------------------------------

def make_log():
    # log(1+x): same Omega_f behaviour (Omega -> const as x->inf), avoids negative outputs.
    # PoA is analytically 1 when v=w (Lemma 2 in draft); skip numerical social_opt.
    return (
        lambda x: math.log(1.0 + max(x, 0.0)),
        lambda x: 1.0 / (1.0 + max(x, 0.0)),
        r"$\log(1{+}x)$",
    )

def make_power(alpha):
    return (
        lambda x: max(x, 0.0) ** alpha,
        lambda x: alpha * max(x, 1e-12) ** (alpha - 1),
        rf"$x^{{{alpha}}}$",
    )

def make_sqrt():
    return make_power(0.5)

def make_exp_sat(k=1.0):
    # f(x) = 1 - exp(-kx),  concave, saturating
    return (
        lambda x: 1.0 - math.exp(-k * max(x, 0.0)),
        lambda x: k * math.exp(-k * max(x, 0.0)),
        rf"$1-e^{{-{k}x}}$",
    )

def make_logistic(L=1.0, k=1.0, x0=2.0):
    # f(x) = L / (1 + exp(-k(x-x0))), concave for x > x0
    return (
        lambda x: L / (1.0 + math.exp(-k * (x - x0))),
        lambda x: L * k * math.exp(-k * (x - x0)) / (1.0 + math.exp(-k * (x - x0))) ** 2,
        rf"$\sigma({k}(x-{x0}))$",
    )

F_SPECS = [
    make_log(),
    make_power(0.25),
    make_power(0.5),
    make_power(0.75),
    make_power(0.9),
    make_exp_sat(k=0.5),
    make_exp_sat(k=1.0),
    make_logistic(L=1.0, k=1.0, x0=2.0),
]

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

N, K = 4, 5
N_SEEDS = 8   # budget/weight draws per f

def sample_instance(seed):
    rng = np.random.default_rng(seed)
    # Budgets: one large, rest smaller — spread ~3x
    B = rng.uniform(1.0, 4.0, size=N)
    B = np.sort(B)[::-1]          # firm 0 = largest budget
    w = rng.dirichlet(np.ones(K))  # v = w
    return B, w

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_cosine(X):
    """Mean pairwise cosine similarity across all (i,j) pairs."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sims = Xn @ Xn.T
    n = len(X)
    vals = [sims[i, j] for i in range(n) for j in range(i+1, n)]
    return np.mean(vals)

def concentration_profile(X, w):
    """
    For each firm, compute the correlation between its allocation share
    x_{i,k}/B_i and the weight w_k.  Perfect log-case gives corr=1 for all firms.
    Returns array of shape (N,).
    """
    shares = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    corrs = [np.corrcoef(shares[i], w)[0, 1] for i in range(len(X))]
    return np.array(corrs)

def omega_profile(f, f_prime, x_vals):
    """Omega_f(x) = x * f'(x) evaluated at x_vals."""
    return np.array([x * f_prime(x) for x in x_vals])

def budget_tilt(X, w, B):
    """
    Tilt: for each firm, compute the KL divergence of its allocation share
    from w (how much it deviates from the log-case proportional allocation).
    Returns array of shape (N,).
    """
    shares = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    kls = []
    for i in range(len(X)):
        p = np.maximum(shares[i], 1e-12)
        q = np.maximum(w, 1e-12)
        kl = np.sum(p * np.log(p / q))
        kls.append(kl)
    return np.array(kls)

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

results = []

for f_fn, fp_fn, f_label in F_SPECS:
    print(f"\n{'='*55}\nf = {f_label}")
    seed_results = []

    for seed in range(N_SEEDS):
        B, w = sample_instance(seed * 100)
        v = w.copy()

        sim = CompetitionSimulation(N, K, B, w, v, f_fn, fp_fn)

        # Nash
        X_nash, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)
        sw_nash = sim.social_welfare(X_nash)

        # Social optimum — skip for log(1+x) (analytically PoA=1 when v=w)
        if f_label == r"$\log(1{+}x)$":
            sw_opt = sw_nash
        else:
            X_opt, sw_opt = sim.social_optimum(n_restarts=10)

        poa = sw_opt / max(sw_nash, 1e-12)
        cos_sim = pairwise_cosine(X_nash)
        conc = concentration_profile(X_nash, w)
        tilt = budget_tilt(X_nash, w, B)

        seed_results.append({
            "poa": poa,
            "cos_sim": cos_sim,
            "conc_by_budget": list(zip(B, conc)),
            "tilt_by_budget": list(zip(B, tilt)),
            "X_nash": X_nash,
            "B": B,
            "w": w,
        })
        print(f"  seed={seed}  PoA={poa:.4f}  cos={cos_sim:.4f}  "
              f"tilt(large)={tilt[0]:.4f}  tilt(small)={tilt[-1]:.4f}")

    # Aggregate
    poas = [r["poa"] for r in seed_results]
    coss = [r["cos_sim"] for r in seed_results]
    # tilt vs budget rank (firm 0=largest ... firm N-1=smallest)
    tilts = np.array([[r["tilt_by_budget"][i][1] for i in range(N)]
                      for r in seed_results])  # (n_seeds, N)

    results.append({
        "label": f_label,
        "f": f_fn,
        "fp": fp_fn,
        "poa_mean": np.mean(poas),
        "poa_std": np.std(poas),
        "cos_mean": np.mean(coss),
        "cos_std": np.std(coss),
        "tilt_mean": tilts.mean(axis=0),   # shape (N,) — largest to smallest budget
        "tilt_std": tilts.std(axis=0),
        "seed_results": seed_results,
    })

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print(f"\n{'='*75}")
print(f"{'f':>25}  {'PoA':>8}  {'±':>6}  {'cos':>8}  {'±':>6}  "
      f"{'tilt[0]':>8}  {'tilt[-1]':>9}")
print(f"{'-'*75}")
for r in results:
    print(f"{r['label']:>25}  {r['poa_mean']:>8.4f}  {r['poa_std']:>6.4f}  "
          f"{r['cos_mean']:>8.4f}  {r['cos_std']:>6.4f}  "
          f"{r['tilt_mean'][0]:>8.4f}  {r['tilt_mean'][-1]:>9.4f}")

# ---------------------------------------------------------------------------
# Omega_f profile plot
# ---------------------------------------------------------------------------

x_vals = np.linspace(0.05, 5.0, 300)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for ax, (f_fn, fp_fn, f_label) in zip(axes, F_SPECS):
    om = omega_profile(f_fn, fp_fn, x_vals)
    ax.plot(x_vals, om, lw=2)
    ax.axhline(om.mean(), ls="--", color="gray", lw=1, label=r"mean $\Omega$")
    ax.set_title(f_label, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Omega_f(x) = x f'(x)$")
    ax.legend(fontsize=8)

plt.suptitle(r"$\Omega_f(x) = xf'(x)$ profiles", fontsize=13)
plt.tight_layout()
plt.savefig("plot_omega_profiles.png", dpi=130)
plt.close()
print("\nSaved: plot_omega_profiles.png")

# ---------------------------------------------------------------------------
# Summary plots: PoA, cosine similarity, tilt by budget rank
# ---------------------------------------------------------------------------

labels = [r["label"] for r in results]
poa_means = [r["poa_mean"] for r in results]
poa_stds  = [r["poa_std"]  for r in results]
cos_means = [r["cos_mean"] for r in results]
cos_stds  = [r["cos_std"]  for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# PoA
ax = axes[0]
xs = range(len(results))
ax.bar(xs, poa_means, yerr=poa_stds, capsize=4, color="steelblue", alpha=0.8)
ax.axhline(1.0, ls="--", color="black", lw=1)
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Price of Anarchy (SW_opt / SW_Nash)")
ax.set_title("Price of Anarchy by f")

# Cosine similarity
ax = axes[1]
ax.bar(xs, cos_means, yerr=cos_stds, capsize=4, color="darkorange", alpha=0.8)
ax.axhline(1.0, ls="--", color="black", lw=1)
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean pairwise cosine similarity")
ax.set_title("Allocation Heterogeneity by f")

# Tilt by budget rank
ax = axes[2]
budget_labels = [f"firm {i+1}\n(rank {i+1})" for i in range(N)]
x_pos = np.arange(N)
width = 0.8 / len(results)
for idx, r in enumerate(results):
    offset = (idx - len(results)/2 + 0.5) * width
    ax.bar(x_pos + offset, r["tilt_mean"], width=width*0.9,
           yerr=r["tilt_std"], capsize=2, label=r["label"], alpha=0.8)
ax.set_xticks(list(x_pos)); ax.set_xticklabels(budget_labels, fontsize=9)
ax.set_ylabel(r"KL(share $\|$ $w$) — deviation from proportional")
ax.set_title("Budget Tilt: who deviates from w?")
ax.legend(fontsize=7, ncol=2)

plt.suptitle("f-sweep: Nash vs Optimal, Heterogeneity, Budget Tilt", fontsize=12)
plt.tight_layout()
plt.savefig("plot_f_sweep_summary.png", dpi=130)
plt.close()
print("Saved: plot_f_sweep_summary.png")
