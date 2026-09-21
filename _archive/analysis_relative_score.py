"""
Quick check: does proportional allocation survive relative score variants?
  (a) absolute baseline:  sigma(f_i - f_j)
  (b) diff / arith mean:  sigma((f_i-f_j) / mean(|f_i|,|f_j|))
  (c) diff / opponent:    sigma((f_i-f_j) / |f_j|)

f(x) = log(x). 1 seed, N=3, K=4.

Prediction: (a) holds (k-independent argument), (b)(c) break (denominator has log(w_k)).
"""

import math
import numpy as np
from scipy.special import expit
from analysis_sigma_shift import CompSimSigma, proportional_alloc, marginal_range

N, K = 3, 4

def sample_instance(seed):
    rng = np.random.default_rng(seed * 13 + 5)
    B = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
    w = rng.dirichlet(np.ones(K))
    return B, w
f_logx  = lambda x: math.log(max(x, 1e-12))
fp_logx = lambda x: 1.0 / max(x, 1e-12)
EPS = 1e-8

class CompSimRelative(CompSimSigma):
    def __init__(self, N, K, B, w, v, f, fp, win_fn, win_fp):
        super().__init__(N, K, B, w, v, f, fp, lambda z: 0., lambda z: 0.)
        self._wfn = win_fn
        self._wfp = win_fp

    def firm_objective(self, x_i, i, X):
        th_i, th = self.f(x_i), self.theta(X)
        return sum(self.w[k] * self._wfn(th_i[k], th[ip, k])
                   for ip in range(self.N) if ip != i for k in range(self.K))

    def kkt_marginals(self, X):
        th, fp = self.theta(X), self._f_prime_vals(X)
        marg = np.zeros((self.N, self.K))
        for i in range(self.N):
            for ip in range(self.N):
                if ip != i:
                    for k in range(self.K):
                        marg[i, k] += self.w[k] * self._wfp(th[i,k], th[ip,k]) * fp[i,k]
        return marg

# --- win functions ---
def abs_win(a, b):
    v = expit(a - b); return float(v)
def abs_win_p(a, b):
    v = expit(a - b); return float(v*(1-v))

def arith_win(a, b):
    d = (abs(a)+abs(b))/2 + EPS; return float(expit((a-b)/d))
def arith_win_p(a, b):
    d = (abs(a)+abs(b))/2 + EPS; z = (a-b)/d
    sa = 1. if a>=0 else -1.
    dz = (d - (a-b)*(sa/2)) / d**2
    v = expit(z); return float(v*(1-v)*dz)

def opp_win(a, b):
    d = max(abs(b), EPS); return float(expit((a-b)/d))
def opp_win_p(a, b):
    d = max(abs(b), EPS); z = (a-b)/d
    v = expit(z); return float(v*(1-v)/d)

VARIANTS = [
    ("absolute  sigma(f_i - f_j)",          abs_win,   abs_win_p),
    ("rel/mean  sigma((f_i-f_j)/mean|f|)",  arith_win, arith_win_p),
    ("rel/opp   sigma((f_i-f_j)/|f_j|)",    opp_win,   opp_win_p),
]

B, w = sample_instance(0)
w = np.maximum(w, 0.02); w /= w.sum()

print(f"B = {np.round(B,2)},  w = {np.round(w,3)}\n")
print(f"{'Variant':<44}  {'marg_res_prop':>14}  {'kl_nash':>10}  {'SW_nash':>9}  {'SW_opt':>9}  {'PoA':>7}")
print("-"*100)

for name, wfn, wfp in VARIANTS:
    sim = CompSimRelative(N, K, B, w, w.copy(), f_logx, fp_logx, wfn, wfp)
    X_prop = proportional_alloc(B, w)
    mr = marginal_range(sim, X_prop)

    X_nash, _ = sim.find_nash_br(max_iter=100, tol=1e-6, verbose=False)
    shares = X_nash / (X_nash.sum(axis=1, keepdims=True) + 1e-12)
    kl = np.mean([float(np.sum(np.maximum(shares[i],1e-12)*np.log(
                  np.maximum(shares[i],1e-12)/np.maximum(w,1e-12)))) for i in range(N)])

    sw_nash = sim.social_welfare(X_nash)
    _, sw_opt = sim.social_optimum(n_restarts=6)
    poa = sw_opt / max(sw_nash, 1e-12)
    print(f"  {name:<42}  {mr:>14.2e}  {kl:>10.2e}  {sw_nash:>9.4f}  {sw_opt:>9.4f}  {poa:>7.4f}")
