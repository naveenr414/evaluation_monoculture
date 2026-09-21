"""
Sweep alpha densely, find max |log R| at each alpha (using the known optimal
structure: two firms at extremes, one tiny), then fit curves.

Optimal structure from previous analysis:
  - alpha < ~0.6: B = [eps, eps, B_total - 2*eps]  (two tiny + one huge)
  - alpha >= ~0.6: B = [t, t-delta, eps]  (two large + one tiny), push t->max

We use both structures and take the max to be safe.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import curve_fit
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


def run(B, alpha):
    f_fn, fp_fn = make_power(alpha)
    sim = CompetitionSimulation(N, K, B.copy(), W.copy(), W.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)
    Sigma = compute_Sigma(X, f_fn)
    return Sigma


def max_log_ror(Sigma):
    pairs = [(0, 1), (0, 2), (1, 2)]
    vals = []
    for i, j in pairs:
        ri = Sigma[i, 0] / (Sigma[i, 1] + 1e-12)
        rj = Sigma[j, 0] / (Sigma[j, 1] + 1e-12)
        r = ri / (rj + 1e-12)
        vals.append(abs(np.log(max(r, 1e-12))))
    return max(vals)


# -----------------------------------------------------------------------
# Two extreme structures to evaluate at each alpha
# -----------------------------------------------------------------------

def eval_alpha(alpha, n_eps=8):
    """
    Try both extreme structures, return the best max|log R|.
    Structure A: B = [eps, eps, B_total - 2*eps]  (two tiny)
    Structure B: B = [t, t-delta, eps]              (two large, sweep t)
    """
    best = -np.inf

    # Structure A: sweep eps
    for eps in np.logspace(-2, -0.5, n_eps):
        if 2 * eps >= B_TOTAL - EPS:
            continue
        B = np.array([eps, eps, B_TOTAL - 2 * eps])
        Sigma = run(B, alpha)
        v = max_log_ror(Sigma)
        best = max(best, v)

    # Structure B: B0=t, B1=t-delta, B2=eps; sweep t
    delta = 0.1
    for t in np.linspace(0.5, (B_TOTAL - EPS - delta) / 2, n_eps):
        B = np.array([t, t - delta, B_TOTAL - 2 * t + delta - EPS])
        if np.any(B <= 0) or B[0] <= B[1] or B[1] <= B[2]:
            continue
        Sigma = run(B, alpha)
        v = max_log_ror(Sigma)
        best = max(best, v)

    return best


# Dense alpha sweep
alpha_vals = np.linspace(0.05, 0.97, 50)
print("Sweeping alpha ...")
max_rors = []
for alpha in alpha_vals:
    v = eval_alpha(alpha, n_eps=12)
    max_rors.append(v)
    print(f"  alpha={alpha:.3f}  max|log R|={v:.5f}")

max_rors = np.array(max_rors)

# -----------------------------------------------------------------------
# Curve fitting
# -----------------------------------------------------------------------

print("\nFitting curves ...")

# Candidate functional forms
def power_law(a, c, beta):
    return c * a ** beta

def exp_law(a, c, lam):
    return c * np.exp(lam * a)

def log_law(a, c, d):
    return c * np.log(a) + d   # log fit (note: negative for a<1 if c>0)

def power_plus_const(a, c, beta, d):
    return c * (a ** beta) + d

def rational(a, c, beta):
    # c * alpha^beta / (1 - alpha)
    return c * a ** beta / (1 - a + 1e-12)

# Filter out any bad values
valid = np.isfinite(max_rors) & (max_rors > 0)
a = alpha_vals[valid]
y = max_rors[valid]

fits = {}

try:
    p, _ = curve_fit(power_law, a, y, p0=[1.0, 2.0], maxfev=5000)
    y_pred = power_law(a, *p)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    fits["power: c·α^β"] = {"params": p, "r2": r2, "fn": power_law,
                             "label": rf"$c\alpha^\beta$: c={p[0]:.3f}, β={p[1]:.3f}"}
    print(f"  power_law:         c={p[0]:.4f}, beta={p[1]:.4f}, R²={r2:.5f}")
except Exception as e:
    print(f"  power_law failed: {e}")

try:
    p, _ = curve_fit(exp_law, a, y, p0=[0.1, 3.0], maxfev=5000)
    y_pred = exp_law(a, *p)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    fits["exp: c·e^{λα}"] = {"params": p, "r2": r2, "fn": exp_law,
                              "label": rf"$ce^{{\lambda\alpha}}$: c={p[0]:.3f}, λ={p[1]:.3f}"}
    print(f"  exp_law:           c={p[0]:.4f}, lam={p[1]:.4f}, R²={r2:.5f}")
except Exception as e:
    print(f"  exp_law failed: {e}")

try:
    p, _ = curve_fit(log_law, a, y, p0=[-0.5, 0.0], maxfev=5000)
    y_pred = log_law(a, *p)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    fits["log: c·log(α)+d"] = {"params": p, "r2": r2, "fn": log_law,
                                "label": rf"$c\log\alpha+d$: c={p[0]:.3f}, d={p[1]:.3f}"}
    print(f"  log_law:           c={p[0]:.4f}, d={p[1]:.4f}, R²={r2:.5f}")
except Exception as e:
    print(f"  log_law failed: {e}")

try:
    p, _ = curve_fit(rational, a, y, p0=[0.1, 2.0], maxfev=5000)
    y_pred = rational(a, *p)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    fits["rational: c·α^β/(1-α)"] = {"params": p, "r2": r2, "fn": rational,
                                       "label": rf"$c\alpha^\beta/(1-\alpha)$: c={p[0]:.3f}, β={p[1]:.3f}"}
    print(f"  rational:          c={p[0]:.4f}, beta={p[1]:.4f}, R²={r2:.5f}")
except Exception as e:
    print(f"  rational failed: {e}")

# Also fit in log-log space to check power law quality
log_a = np.log(a)
log_y = np.log(y)
coeffs = np.polyfit(log_a, log_y, 1)
beta_loglog = coeffs[0]
c_loglog = np.exp(coeffs[1])
y_loglog = c_loglog * a ** beta_loglog
r2_loglog = 1 - np.sum((y - y_loglog)**2) / np.sum((y - np.mean(y))**2)
print(f"  log-log linear:    c={c_loglog:.4f}, beta={beta_loglog:.4f}, R²={r2_loglog:.5f}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"max$|\log R|$ as a function of $\alpha$: $f(x)=x^\alpha$, $N=3$, $K=2$",
             fontsize=12, fontweight="bold")

a_plot = np.linspace(0.05, 0.97, 300)

# Panel 1: linear scale
ax = axes[0]
ax.scatter(alpha_vals[valid], y, color="black", s=20, zorder=5, label="data")
best_fit = max(fits.items(), key=lambda kv: kv[1]["r2"])
colors_fit = ["firebrick", "steelblue", "darkorange", "mediumseagreen"]
for (name, fd), col in zip(fits.items(), colors_fit):
    try:
        y_curve = fd["fn"](a_plot, *fd["params"])
        ax.plot(a_plot, y_curve, color=col, lw=1.8, ls="--",
                label=f"{fd['label']}  $R^2$={fd['r2']:.3f}")
    except Exception:
        pass
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"max$|\log R|$")
ax.set_title("Linear scale — all fits")
ax.legend(fontsize=8)

# Panel 2: log-log scale
ax = axes[1]
ax.loglog(alpha_vals[valid], y, "o", color="black", ms=4, label="data")
ax.loglog(a_plot, c_loglog * a_plot ** beta_loglog, "--", color="firebrick", lw=2,
          label=rf"power law: {c_loglog:.3f}·$\alpha^{{{beta_loglog:.3f}}}$  $R^2$={r2_loglog:.3f}")
if "rational: c·α^β/(1-α)" in fits:
    fd = fits["rational: c·α^β/(1-α)"]
    try:
        y_r = fd["fn"](a_plot, *fd["params"])
        ax.loglog(a_plot[y_r > 0], y_r[y_r > 0], "--", color="steelblue", lw=2,
                  label=f"{fd['label']}  $R^2$={fd['r2']:.3f}")
    except Exception:
        pass
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"max$|\log R|$")
ax.set_title("Log-log scale")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_alpha_curve.pdf", bbox_inches="tight")
plt.savefig("plot_alpha_curve.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_alpha_curve.pdf / .png")
