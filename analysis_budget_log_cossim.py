"""
Sweep log(B_max/B_min) exponentially and plot worst-case pairwise cosine
similarity of theta (output vectors) vs. that log-budget-ratio.

Setup: N=3, K=2, f(x) = x^0.5 (as in analysis_worst_case_structure.py).
Worst case structure: one large-budget firm, two tiny-budget firms
(the config that analysis_worst_case_structure.py found drives minimal
cos_sim). We push the large budget exponentially larger relative to a
fixed small budget and track how cos_sim(large, small) behaves.

Total budget is NOT held fixed here -- B_total = B_large + 2*B_small
grows as B_large grows, since we want log(B_large/B_small) to range widely
without B_small vanishing to numerical dust. A second panel repeats the
sweep holding B_total fixed (redistributing away from the small firms)
so the two tiny firms shrink together as the big one grows.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from competition_simulation import CompetitionSimulation

K = 2
W = np.array([0.7, 0.3])
EPS_FLOOR = 1e-9
ALPHAS = [0.1, 0.5, 0.9]
COLORS = ["steelblue", "darkorange", "firebrick"]


def make_power(p):
    f = lambda x: np.clip(x, EPS_FLOOR, None) ** p
    fp = lambda x: p * np.clip(x, EPS_FLOOR, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def run(B, p):
    f, fp = make_power(p)
    N = len(B)
    sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=400, tol=1e-11, verbose=False)
    theta = sim.theta(X)
    return theta


# -----------------------------------------------------------------------
# Sweep 1: B_small fixed, B_large grows exponentially (B_total grows)
# -----------------------------------------------------------------------
B_SMALL_FIXED = 0.05
log_ratios = np.linspace(0.1, 9.0, 30)   # log(B_large / B_small)

results_growing = {p: [] for p in ALPHAS}
for p in ALPHAS:
    for lr in log_ratios:
        B_large = B_SMALL_FIXED * np.exp(lr)
        B = [B_large, B_SMALL_FIXED, B_SMALL_FIXED]
        theta = run(B, p)
        cs = cos_sim(theta[0], theta[1])   # large vs. one small firm
        cs_small_small = cos_sim(theta[1], theta[2])
        results_growing[p].append((cs, cs_small_small))

# -----------------------------------------------------------------------
# Sweep 2: B_total fixed = 9.0, redistribute so B_large grows and the two
# small firms shrink together (worst-case structure from
# analysis_worst_case_structure.py: two-tiny-one-huge).
# -----------------------------------------------------------------------
B_TOTAL = 9.0
# B_large = B_total - 2*eps; log ratio = log((B_total-2eps)/eps)
eps_vals = np.logspace(np.log10(B_TOTAL / 2 - 1e-6), -5, 40)  # eps: large -> tiny
results_fixed = {p: [] for p in ALPHAS}
log_ratios_fixed = {p: [] for p in ALPHAS}
for p in ALPHAS:
    for eps in eps_vals:
        B_large = B_TOTAL - 2 * eps
        if B_large <= eps:
            continue
        B = [B_large, eps, eps]
        theta = run(B, p)
        cs = cos_sim(theta[0], theta[1])
        log_ratios_fixed[p].append(np.log(B_large / eps))
        results_fixed[p].append(cs)


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for p, color in zip(ALPHAS, COLORS):
    cs_large_small = [r[0] for r in results_growing[p]]
    ax.plot(log_ratios, cs_large_small, color=color, lw=2, marker="o", ms=3,
             label=rf"$\alpha$={p}")
ax.set_xlabel(r"$\log(B_{\mathrm{large}}/B_{\mathrm{small}})$  ($B_{\mathrm{small}}$=0.05 fixed)")
ax.set_ylabel(r"cos_sim($\theta_{\mathrm{large}}, \theta_{\mathrm{small}}$)")
ax.set_title("Worst-case cos_sim vs. log budget ratio\n(B_small fixed, B_total grows)")
ax.legend()
ax.set_ylim(-1.05, 1.05)
ax.axhline(0, ls="--", color="gray", lw=0.8)

ax = axes[1]
for p, color in zip(ALPHAS, COLORS):
    ax.plot(log_ratios_fixed[p], results_fixed[p], color=color, lw=2, marker="o", ms=3,
             label=rf"$\alpha$={p}")
ax.set_xlabel(r"$\log(B_{\mathrm{large}}/B_{\mathrm{small}})$  ($B_{\mathrm{total}}$=9 fixed)")
ax.set_ylabel(r"cos_sim($\theta_{\mathrm{large}}, \theta_{\mathrm{small}}$)")
ax.set_title("Worst-case cos_sim vs. log budget ratio\n(B_total fixed, two-tiny-one-huge)")
ax.legend()
ax.set_ylim(-1.05, 1.05)
ax.axhline(0, ls="--", color="gray", lw=0.8)

fig.suptitle(r"$N=3$, $K=2$, $f(x)=x^\alpha$: does large log-budget gap force worst-case cos_sim $\to -1$?",
             fontsize=12)
plt.tight_layout()
plt.savefig("plot_budget_log_cossim.pdf", bbox_inches="tight")
plt.savefig("plot_budget_log_cossim.png", dpi=130, bbox_inches="tight")
plt.close()

print("Saved: plot_budget_log_cossim.pdf / .png")
print()
for p in ALPHAS:
    cs_vals = [r[0] for r in results_growing[p]]
    print(f"alpha={p}: growing-budget sweep cos_sim range [{min(cs_vals):.4f}, {max(cs_vals):.4f}], "
          f"final (log ratio={log_ratios[-1]:.2f}) = {cs_vals[-1]:.4f}")
    fv = results_fixed[p]
    print(f"         fixed-total sweep cos_sim range [{min(fv):.4f}, {max(fv):.4f}], "
          f"final (log ratio={log_ratios_fixed[p][-1]:.2f}) = {fv[-1]:.4f}")
