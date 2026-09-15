"""
Follow-up: is the interior optimum for alpha=0.75 real or numerical artifact?

Tests:
  1. Diagonal sweep: B0=B1=t, B2=9-2t, t in [0.1, 4.4]
     -- isolates the "two equal large + one tiny" structure
     -- if interior optimum is real, |log R(1,2)| should have a peak strictly inside
  2. Near-diagonal: B0=t+delta, B1=t, B2=9-2t-delta for small delta=0.1
     -- confirm ordering B0>B1>B2 is maintained and result is similar
  3. alpha=0.6 full grid search (same as before) to confirm transition
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
EPS = 0.05


def make_power(alpha):
    f = lambda x: max(x, 0.0) ** alpha
    fp = lambda x: alpha * max(x, 1e-12) ** (alpha - 1)
    return f, fp


def sigma_prime(z):
    s = expit(z)
    return s * (1.0 - s)


def compute_Sigma(X, f_fn):
    fx = np.array([[f_fn(X[i, k]) for k in range(K)] for i in range(N)])
    Sigma = np.zeros((N, K))
    for i in range(N):
        for ip in range(N):
            if ip != i:
                diff = fx[i] - fx[ip]
                Sigma[i] += np.array([sigma_prime(d) for d in diff])
    return Sigma


def ror(Sigma, i, j):
    ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
    rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
    return ri / (rj + 1e-12)


def run(B, alpha):
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B.copy(), W.copy(), W.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)
    Sigma = compute_Sigma(X, f_fn)
    return X, Sigma


# -----------------------------------------------------------------------
# 1. Diagonal sweep: B0=B1=t, B2=9-2t
# -----------------------------------------------------------------------

print("=" * 60)
print("Diagonal sweep: B0=B1=t, B2=9-2t")
print("=" * 60)

t_vals = np.linspace(0.1, 4.4, 80)
ALPHAS_MAIN = [0.25, 0.50, 0.60, 0.75]
COLORS = ["steelblue", "darkorange", "purple", "mediumseagreen"]

diag_results = {}
for alpha in ALPHAS_MAIN:
    log_r12 = []
    log_r01 = []
    log_r02 = []
    for t in t_vals:
        B2 = B_TOTAL - 2 * t
        if B2 <= 0:
            log_r12.append(np.nan); log_r01.append(np.nan); log_r02.append(np.nan)
            continue
        B = np.array([t, t, B2])
        _, Sigma = run(B, alpha)
        log_r12.append(np.log(max(ror(Sigma, 1, 2), 1e-12)))
        log_r01.append(np.log(max(ror(Sigma, 0, 1), 1e-12)))
        log_r02.append(np.log(max(ror(Sigma, 0, 2), 1e-12)))
    diag_results[alpha] = {
        "log_r12": np.array(log_r12),
        "log_r01": np.array(log_r01),
        "log_r02": np.array(log_r02),
    }
    valid = np.isfinite(log_r12)
    idx_max = np.nanargmax(np.abs(log_r12))
    print(f"  alpha={alpha}: peak |log R(1,2)| = {abs(log_r12[idx_max]):.5f} "
          f"at t={t_vals[idx_max]:.3f}  (B=[{t_vals[idx_max]:.2f},{t_vals[idx_max]:.2f},{B_TOTAL-2*t_vals[idx_max]:.2f}])")
    # Is the peak at the boundary (t->0.1) or interior?
    if idx_max == 0 or idx_max == 1:
        print(f"    => BOUNDARY: monotone decreasing, extreme is optimal")
    elif idx_max >= len(t_vals[valid]) - 3:
        print(f"    => BOUNDARY: monotone increasing, push t->4.4 is optimal")
    else:
        print(f"    => INTERIOR optimum at t={t_vals[idx_max]:.3f}")

# -----------------------------------------------------------------------
# 2. Near-diagonal (B0=t+0.1, B1=t, B2=9-2t-0.1) to preserve B0>B1>B2
# -----------------------------------------------------------------------

print("\nNear-diagonal (delta=0.1): B0=t+0.1, B1=t, B2=9-2t-0.1")
print("=" * 60)

delta = 0.1
near_diag = {}
for alpha in ALPHAS_MAIN:
    log_r12 = []
    for t in t_vals:
        B0 = t + delta
        B1 = t
        B2 = B_TOTAL - B0 - B1
        if B2 <= EPS or B0 <= B1 or B1 <= B2:
            log_r12.append(np.nan)
            continue
        B = np.array([B0, B1, B2])
        _, Sigma = run(B, alpha)
        log_r12.append(np.log(max(ror(Sigma, 1, 2), 1e-12)))
    near_diag[alpha] = np.array(log_r12)
    idx_max = np.nanargmax(np.abs(near_diag[alpha]))
    print(f"  alpha={alpha}: peak |log R(1,2)| at t={t_vals[idx_max]:.3f}  val={abs(near_diag[alpha][idx_max]):.5f}")

# -----------------------------------------------------------------------
# 3. Full grid search for alpha=0.6
# -----------------------------------------------------------------------

print("\nFull grid search: alpha=0.6, B0>B1>B2")
print("=" * 60)

GRID_N = 60
alpha = 0.6
best_val, best_B = -np.inf, None
vals_grid = np.full((GRID_N, GRID_N), np.nan)
b0_grid = np.linspace(EPS, B_TOTAL - 2 * EPS, GRID_N)
b1_grid = np.linspace(EPS, B_TOTAL - 2 * EPS, GRID_N)

for ii, b0 in enumerate(b0_grid):
    for jj, b1 in enumerate(b1_grid):
        b2 = B_TOTAL - b0 - b1
        if not (b0 > b1 > b2 > EPS):
            continue
        B = np.array([b0, b1, b2])
        _, Sigma = run(B, alpha)
        pairs = [(0,1),(0,2),(1,2)]
        v = max(abs(np.log(max(ror(Sigma,*p), 1e-12))) for p in pairs)
        vals_grid[ii, jj] = v
        if v > best_val:
            best_val, best_B = v, B.copy()

_, Sigma_best = run(best_B, alpha)
rors06 = {p: ror(Sigma_best, *p) for p in [(0,1),(0,2),(1,2)]}
print(f"  alpha=0.6: best B={best_B.round(3)}, max|log R|={best_val:.5f}")
print(f"  RoRs: R(0,1)={rors06[(0,1)]:.5f}  R(0,2)={rors06[(0,2)]:.5f}  R(1,2)={rors06[(1,2)]:.5f}")
winning = max([(0,1),(0,2),(1,2)], key=lambda p: abs(np.log(max(rors06[p],1e-12))))
print(f"  Winning pair: {winning}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(r"Interior vs boundary: $N=3$, $K=2$, $B_0=B_1=t$, $B_2=9-2t$",
             fontsize=12, fontweight="bold")

# Panel 1: log R(1,2) along diagonal for all alphas
ax = axes[0]
for alpha, color in zip(ALPHAS_MAIN, COLORS):
    y = diag_results[alpha]["log_r12"]
    ax.plot(t_vals, y, color=color, lw=2, label=rf"$\alpha$={alpha}")
ax.axhline(0, ls=":", color="gray", lw=1)
ax.set_xlabel("$t$ = $B_0 = B_1$")
ax.set_ylabel(r"$\log R(1,2)$")
ax.set_title(r"$\log R(1,2)$ along diagonal $B_0=B_1=t$")
ax.legend(fontsize=9)

# Panel 2: |log R(1,2)| to make peak clearer
ax = axes[1]
for alpha, color in zip(ALPHAS_MAIN, COLORS):
    y = np.abs(diag_results[alpha]["log_r12"])
    ax.plot(t_vals, y, color=color, lw=2, label=rf"$\alpha$={alpha}")
ax.set_xlabel("$t$ = $B_0 = B_1$")
ax.set_ylabel(r"$|\log R(1,2)|$")
ax.set_title(r"$|\log R(1,2)|$: interior peak?")
ax.legend(fontsize=9)

# Panel 3: heatmap for alpha=0.6
ax = axes[2]
im = ax.imshow(vals_grid.T, origin="lower", aspect="auto",
               extent=[b0_grid[0], b0_grid[-1], b1_grid[0], b1_grid[-1]],
               cmap="viridis")
plt.colorbar(im, ax=ax, label="|log R| max over pairs")
ax.scatter([best_B[0]], [best_B[1]], color="red", marker="*", s=120, zorder=5,
           label=f"opt={best_B.round(1)}")
ax.set_xlabel("$B_0$"); ax.set_ylabel("$B_1$")
ax.set_title(r"$\alpha=0.6$: heatmap (B_0>B_1>B_2)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_sigma_extreme2.pdf", bbox_inches="tight")
plt.savefig("plot_sigma_extreme2.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_sigma_extreme2.pdf / .png")
