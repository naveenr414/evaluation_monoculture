"""
Test whether proportional allocation holds under scale-shifted sigma:
  sigma_beta(z) = sigma(beta * z)

With f(x) = log(x), the KKT argument is beta*(log(B_i) - log(B_{i'})),
still independent of k — so proportional should remain Nash for any beta.

With f(x) = log(1+x) it should again break.

We also test combining scale + center shift: sigma(beta*z + logit(c)).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from analysis_sigma_shift import CompSimSigma, proportional_alloc, marginal_range

BETA_VALS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_SEEDS   = 1
N, K = 3, 4

def sample_instance(seed):
    rng = np.random.default_rng(seed * 13 + 5)
    B = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
    w = rng.dirichlet(np.ones(K))
    return B, w

f_logx  = lambda x: math.log(max(x, 1e-12))
fp_logx = lambda x: 1.0 / max(x, 1e-12)

f_log1x  = lambda x: math.log(1.0 + max(x, 0.0))
fp_log1x = lambda x: 1.0 / (1.0 + max(x, 0.0))

def make_scaled_sigma(beta, c=0.5):
    offset = logit(c)
    sig  = lambda z: float(expit(beta * z + offset))
    sigp = lambda z: beta * float(expit(beta * z + offset) * (1 - expit(beta * z + offset)))
    return sig, sigp

# ---------------------------------------------------------------------------
# Sweep beta, f = log(x)
# ---------------------------------------------------------------------------

print("=== f(x) = log(x), sigma_beta(z) = sigma(beta*z) ===")
print(f"{'beta':>8}  {'marg_res_prop':>15}")
for beta in BETA_VALS:
    sig, sigp = make_scaled_sigma(beta)
    mr_list = []
    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        w_safe = np.maximum(w, 0.01); w_safe /= w_safe.sum()
        sim = CompSimSigma(N, K, B, w_safe, w_safe.copy(), f_logx, fp_logx, sig, sigp)
        X_prop = proportional_alloc(B, w_safe)
        mr_list.append(marginal_range(sim, X_prop))
    print(f"{beta:>8.1f}  {np.mean(mr_list):>15.2e}")

# ---------------------------------------------------------------------------
# Sweep beta, f = log(1+x)
# ---------------------------------------------------------------------------

print("\n=== f(x) = log(x), sigma(beta*z + logit(c)) — combined grid ===")
C_VALS    = [0.25, 0.5, 0.75]
BETA_GRID = [0.5, 1.0, 5.0]
print(f"{'beta':>6}  {'c':>6}  {'marg_res_prop':>15}")
B, w = sample_instance(0)
w_safe = np.maximum(w, 0.01); w_safe /= w_safe.sum()
for beta in BETA_GRID:
    for c in C_VALS:
        sig, sigp = make_scaled_sigma(beta, c)
        sim = CompSimSigma(N, K, B, w_safe, w_safe.copy(), f_logx, fp_logx, sig, sigp)
        X_prop = proportional_alloc(B, w_safe)
        print(f"{beta:>6.1f}  {c:>6.2f}  {marginal_range(sim, X_prop):>15.2e}")
