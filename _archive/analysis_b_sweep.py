"""
Investigate how b affects Nash equilibrium allocations in f(x) = log(x+b).

N=5 firms, B=[5,4,3,2,1]. Fix w, sweep b.

Plot A: Investment ratio x_{top}/x_{bot} (top vs bottom w_k dim) averaged across firms, vs b.
Plot B: Investment ratio for highest-budget firm vs lowest-budget firm, vs b.
Plot C: Five subplots (one per firm) — absolute investment in top w_k dim, vs b.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from competition_simulation import CompetitionSimulation

N, K = 5, 6
B = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

# Fix w with top weight ~1/4, rest spread across remaining dims
w = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
w = w / w.sum()   # normalize (already sums to 1 but be safe)
print(f"w = {np.round(w, 3)}")

b_vals = np.concatenate([np.linspace(0.01, 0.5, 8), np.linspace(0.5, 5.0, 20), np.linspace(5.0, 20.0, 10)])
b_vals = np.unique(np.round(b_vals, 4))

# Storage
ratio_top_bot  = []           # avg across firms: x_{i,top} / x_{i,bot}
ratio_rich     = []           # firm 0 (B=5): x_{top}/x_{bot}
ratio_poor     = []           # firm 4 (B=1): x_{top}/x_{bot}
abs_top        = [[] for _ in range(N)]   # absolute x_{i,top} per firm

print(f"Sweeping {len(b_vals)} b values ...", flush=True)
for idx, b in enumerate(b_vals):
    print(f"  b={b:.3f} ({idx+1}/{len(b_vals)})", flush=True)
    f_fn  = lambda x, b=b: math.log(max(x, 0) + b)
    fp_fn = lambda x, b=b: 1.0 / (max(x, 0) + b)

    sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=150, tol=1e-6, verbose=False)

    top_k = 0    # highest w_k (index 0 since w is sorted descending)
    bot_k = K-1  # lowest  w_k

    shares = []
    for i in range(N):
        x_top = X[i, top_k]
        s = x_top / B[i]   # share of budget in top dim
        shares.append(s)
        abs_top[i].append(x_top)

    ratio_top_bot.append(np.mean(shares))
    ratio_rich.append(shares[0])    # firm 0, B=5
    ratio_poor.append(shares[-1])   # firm 4, B=1

ratio_top_bot = np.array(ratio_top_bot)
ratio_rich    = np.array(ratio_rich)
ratio_poor    = np.array(ratio_poor)
abs_top       = [np.array(a) for a in abs_top]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1, hspace=0.45,
                      left=0.08, right=0.97, top=0.93, bottom=0.07)

# ── Plot A: avg investment ratio top/bot vs b ────────────────────────────────
ax_a = fig.add_subplot(gs[0])
ax_a.plot(b_vals, ratio_top_bot, "o-", color="steelblue", lw=2, ms=4)
ax_a.set_xlabel("b  (shift in log(x+b))", fontsize=11)
ax_a.set_ylabel(r"Avg $x_{i,top} / B_i$", fontsize=11)
ax_a.set_title("(A)  Avg budget share in top $w_k$ dim", fontsize=11, fontweight="bold")
ax_a.axvline(0, ls="--", color="gray", lw=1)

# ── Plot B: ratio for richest vs poorest firm ────────────────────────────────
ax_b = fig.add_subplot(gs[1])
ax_b.plot(b_vals, ratio_rich, "o-", color="steelblue", lw=2, ms=4, label=f"Richest firm (B=5)")
ax_b.plot(b_vals, ratio_poor, "s-", color="firebrick", lw=2, ms=4, label=f"Poorest firm (B=1)")
ax_b.set_xlabel("b", fontsize=11)
ax_b.set_ylabel(r"$x_{i,top} / B_i$  (budget share in top dim)", fontsize=11)
ax_b.set_title("(B)  Budget share in top dim: richest vs poorest firm", fontsize=11, fontweight="bold")
ax_b.legend(fontsize=10)

# ── Plot C: all firms on one plot ───────────────────────────────────────────
colors = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "firebrick"]
ax_c = fig.add_subplot(gs[2])
for i in range(N):
    ax_c.plot(b_vals, abs_top[i], "o-", color=colors[i], lw=2, ms=3, label=f"Firm {i+1} (B={int(B[i])})")
    ax_c.axhline(B[i] * w[0], ls="--", color=colors[i], lw=0.8, alpha=0.5)
ax_c.set_xlabel("b", fontsize=11)
ax_c.set_ylabel("$x_{i,top}$  (absolute investment in top dim)", fontsize=10)
ax_c.set_title("(C)  Absolute investment in top $w_k$ dim — all firms", fontsize=11, fontweight="bold")
ax_c.legend(fontsize=9)

fig.suptitle(
    r"Effect of shift $b$ in $f(x)=\log(x+b)$ on Nash equilibrium allocations",
    fontsize=13, fontweight="bold"
)

plt.savefig("plot_b_sweep.png", dpi=130)
plt.close()
print("\nSaved: plot_b_sweep.png")
