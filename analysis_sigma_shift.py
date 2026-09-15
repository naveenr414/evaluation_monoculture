"""
Test whether proportional allocation (x_{i,k} = B_i * w_k) remains a Nash
equilibrium for f(x) = log(1+x) when the win-rate function sigma is shifted
so that equal-quality outputs yield win probability c ≠ 0.5.

Shifted sigma: sigma_c(z) = sigma(z + logit(c))
  => sigma_c(0) = c   (when both firms have equal quality, firm i wins with prob c)

We sweep c in {0.1, 0.25, 0.5, 0.75, 0.9} and check:
  1. KL divergence of Nash allocation from proportional (should ≈ 0)
  2. Max marginal deviation at proportional point (KKT residual, should ≈ 0)
  3. Whether best-response dynamics converge to proportional
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from competition_simulation import CompetitionSimulation

# ---------------------------------------------------------------------------
# Shifted sigma: sigma_c(z) = sigma(z + logit(c)), sigma_c'(z) = sigma'(z + logit(c))
# ---------------------------------------------------------------------------

def make_shifted_sigma(c: float):
    """Returns (sigma_c, sigma_c') pair."""
    offset = logit(c)
    def s(z):
        v = expit(z + offset)
        return v
    def sp(z):
        v = expit(z + offset)
        return v * (1 - v)
    return s, sp

# ---------------------------------------------------------------------------
# Modified CompetitionSimulation that accepts a custom sigma
# ---------------------------------------------------------------------------

from scipy.optimize import minimize

class CompSimSigma(CompetitionSimulation):
    """Extend CompetitionSimulation to use a custom win-rate function sigma."""
    def __init__(self, N, K, B, w, v, f, f_prime, sigma, sigma_prime):
        super().__init__(N, K, B, w, v, f, f_prime)
        self._sigma     = np.vectorize(sigma)
        self._sigma_p   = np.vectorize(sigma_prime)
        self._sigma_s   = sigma
        self._sigma_ps  = sigma_prime

    def firm_objective(self, x_i, i, X):
        th_i = self.f(x_i)
        th   = self.theta(X)
        total = 0.0
        for ip in range(self.N):
            if ip != i:
                total += float(np.sum(self.w * self._sigma(th_i - th[ip])))
        return total

    def all_objectives(self, X):
        th = self.theta(X)
        objs = np.zeros(self.N)
        for i in range(self.N):
            for ip in range(self.N):
                if ip != i:
                    objs[i] += float(np.sum(self.w * self._sigma(th[i] - th[ip])))
        return objs

    def kkt_marginals(self, X):
        th = self.theta(X)
        fp = self._f_prime_vals(X)
        marginals = np.zeros((self.N, self.K))
        for i in range(self.N):
            for ip in range(self.N):
                if ip != i:
                    diff = th[i] - th[ip]
                    sp   = np.array([self._sigma_ps(float(d)) for d in diff])
                    marginals[i] += self.w * sp * fp[i]
        return marginals

# ---------------------------------------------------------------------------
# Production function: log(1 + x)
# ---------------------------------------------------------------------------

f_log    = lambda x: math.log(1.0 + max(x, 0.0))
fp_log   = lambda x: 1.0 / (1.0 + max(x, 0.0))

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

RNG     = np.random.default_rng(7)
N, K    = 4, 6
N_SEEDS = 10
C_VALS  = [0.1, 0.25, 0.5, 0.75, 0.9]

def sample_instance(seed):
    rng = np.random.default_rng(seed * 13 + 5)
    B = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
    w = rng.dirichlet(np.ones(K))
    return B, w

def proportional_alloc(B, w):
    return np.outer(B, w)

def kl_from_proportional(X, w):
    """KL(share_i || w) averaged over firms."""
    shares = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    kls = []
    for i in range(len(X)):
        p = np.maximum(shares[i], 1e-12)
        q = np.maximum(w,        1e-12)
        kls.append(float(np.sum(p * np.log(p / q))))
    return np.mean(kls)

def marginal_range(sim, X):
    """Max - Min of KKT marginals across active dimensions, averaged over firms."""
    marg = sim.kkt_marginals(X)
    ranges = []
    for i in range(sim.N):
        active = X[i] > 1e-8
        if active.sum() > 1:
            ranges.append(marg[i][active].max() - marg[i][active].min())
    return np.mean(ranges) if ranges else 0.0

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

print(f"{'c':>6}  {'KL_nash':>10}  {'KL_prop':>10}  {'marg_res_nash':>14}  {'marg_res_prop':>14}")
print("-" * 60)

summary = {}

for c in C_VALS:
    sig, sigp = make_shifted_sigma(c)
    kl_nash_list, kl_prop_list = [], []
    mr_nash_list, mr_prop_list = [], []

    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        sim = CompSimSigma(N, K, B, w, w.copy(), f_log, fp_log, sig, sigp)

        # Nash equilibrium
        X_nash, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)
        kl_nash_list.append(kl_from_proportional(X_nash, w))
        mr_nash_list.append(marginal_range(sim, X_nash))

        # Proportional point
        X_prop = proportional_alloc(B, w)
        kl_prop_list.append(kl_from_proportional(X_prop, w))   # should be ~0
        mr_prop_list.append(marginal_range(sim, X_prop))        # should be ~0 if Nash

    summary[c] = {
        "kl_nash": np.mean(kl_nash_list),
        "kl_prop": np.mean(kl_prop_list),
        "mr_nash": np.mean(mr_nash_list),
        "mr_prop": np.mean(mr_prop_list),
    }
    print(f"{c:>6.2f}  {np.mean(kl_nash_list):>10.2e}  {np.mean(kl_prop_list):>10.2e}"
          f"  {np.mean(mr_nash_list):>14.2e}  {np.mean(mr_prop_list):>14.2e}")

# ---------------------------------------------------------------------------
# Analytical check: marginals at proportional point
# ---------------------------------------------------------------------------

print("\n--- Analytical verification at proportional point ---")
print("For log(1+x) and sigma_c, the KKT marginal at x_{i,k}=B_i*w_k equals:")
print("  (1/B_i) * sum_{i'!=i} sigma_c'(log((1+B_i*w_k)/(1+B_i'*w_k)))")
print("The key question: is this independent of k? (it won't be exactly because of the")
print("log(1+x) vs log(x) difference — the +1 offset breaks exact proportionality.)\n")

# Use log(x) (no +1) to get exact result
f_logx   = lambda x: math.log(max(x, 1e-12))
fp_logx  = lambda x: 1.0 / max(x, 1e-12)

print("With f(x) = log(x) (no +1 shift):")
print(f"{'c':>6}  {'marg_res_prop (log x)':>22}")
for c in C_VALS:
    sig, sigp = make_shifted_sigma(c)
    mr_list = []
    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        # Avoid x=0: use uniform lower bound
        w_safe = np.maximum(w, 0.01); w_safe /= w_safe.sum()
        sim = CompSimSigma(N, K, B, w_safe, w_safe.copy(), f_logx, fp_logx, sig, sigp)
        X_prop = proportional_alloc(B, w_safe)
        mr_list.append(marginal_range(sim, X_prop))
    print(f"{c:>6.2f}  {np.mean(mr_list):>22.2e}")

# ---------------------------------------------------------------------------
# Plot: Nash allocation vs proportional for each c
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, len(C_VALS), figsize=(4*len(C_VALS), 4), sharey=True)
fig.suptitle(
    r"Nash allocation shares vs $w_k$ for $f(x)=\log(1{+}x)$, varying $\sigma_c$ center"
    "\n(dots should lie on y=x line if proportional allocation is Nash)",
    fontsize=11
)

for ax, c in zip(axes, C_VALS):
    sig, sigp = make_shifted_sigma(c)
    all_w, all_share = [], []
    seed = 0
    B, w = sample_instance(seed)
    sim = CompSimSigma(N, K, B, w, w.copy(), f_log, fp_log, sig, sigp)
    X_nash, _ = sim.find_nash_br(max_iter=200, tol=1e-7, verbose=False)
    shares = X_nash / X_nash.sum(axis=1, keepdims=True)
    for i in range(N):
        ax.scatter(w, shares[i], s=30, alpha=0.7, label=f"firm {i+1} (B={B[i]:.1f})")
    lims = [0, max(w.max(), shares.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="y=x (proportional)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"$w_k$ (evaluator weight)", fontsize=9)
    ax.set_ylabel(r"$x_{i,k}/B_i$ (share)", fontsize=9) if ax == axes[0] else None
    ax.set_title(f"$\\sigma_c$, c={c}\n(KL_Nash={summary[c]['kl_nash']:.1e})", fontsize=9)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("plot_sigma_shift.pdf", bbox_inches="tight")
plt.savefig("plot_sigma_shift.png", dpi=130)
plt.close()
print("\nSaved: plot_sigma_shift.png / .pdf")
