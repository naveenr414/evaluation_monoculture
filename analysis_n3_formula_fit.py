"""
Derive & test an approximate closed form for x_{i,k} in the N=3 worst-case
structure: two starved firms (1,2) with small, roughly equal budgets, one
dominant firm (3) with the bulk of the budget.

Derivation sketch (leading-order asymptotics, sigma'(z) ~ e^{-|z|} for |z| large):

STARVED firms (1,2): each faces one weak opponent (sigma'(~0) = 1/4, roughly
constant across k) and one dominant opponent (sigma'(theta_i,k - theta_3,k)
~ e^{-theta_3,k}, small vs 1/4 once firm 3 has decisively won). To leading
order the dominant-opponent term is negligible next to the 1/4 term, so
S_{i,k} ~ (1/4) w_k -- independent of k-details of firm 3. FOC then gives a
CLEAN power law:

    x_{i,k}  ~  B_i * w_k^{1/(1-p)} / sum_k' w_k'^{1/(1-p)}      (i = 1, 2)

DOMINANT firm (3): faces two weak opponents, S_{3,k} ~ 2 w_k e^{-x_{3,k}^p}.
FOC: p x_{3,k}^{p-1} * 2 w_k * e^{-x_{3,k}^p} = lambda_3 for all k.
Taking logs:  (p-1) log(x_{3,k})  -  x_{3,k}^p  +  log(w_k)  =  const
This is transcendental (no closed form) but has clean limits:
  p -> 0:  x_{3,k} ~ w_k                      (matches starved formula! -> homogeneity)
  p -> 1:  x_{3,k} ~ log(w_k) + const          (ADDITIVE in log(w_k), much flatter
                                                 than the starved firms' w_k^{1/(1-p)})
"""

import numpy as np
from scipy.optimize import brentq
from competition_simulation import CompetitionSimulation

K = 3
W = np.array([0.5, 0.3, 0.2])
EPS_FLOOR = 1e-9


def make_power(p):
    f = lambda x: np.clip(x, EPS_FLOOR, None) ** p
    fp = lambda x: p * np.clip(x, EPS_FLOOR, None) ** (p - 1)
    return f, fp


def starved_formula(B_i, p, W):
    weights = W ** (1 / (1 - p))
    return B_i * weights / weights.sum()


def dominant_formula(B3, p, W, n_weak=2):
    # solve (p-1)ln(x) - x^p + ln(w_k) = C for each k, with sum_k x_k = B3
    # g(x) = (p-1)ln(x) - x^p + ln(w) - C is monotonically decreasing in x
    def x_of_C(C, w):
        g = lambda x: (p - 1) * np.log(x) - x ** p + np.log(w) - C
        lo, hi = 1e-12, 1e4
        while g(lo) < 0:
            lo /= 10
        while g(hi) > 0:
            hi *= 10
        return brentq(g, lo, hi)

    def total(C):
        return sum(x_of_C(C, w) for w in W) - B3

    C_lo, C_hi = -100.0, 100.0
    while total(C_lo) < 0:
        C_lo -= 50
    while total(C_hi) > 0:
        C_hi += 50
    C = brentq(total, C_lo, C_hi)
    return np.array([x_of_C(C, w) for w in W])


for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    f, fp = make_power(p)
    B = np.array([0.02, 0.02, 9.0 - 0.04])
    sim = CompetitionSimulation(3, K, B, W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=500, tol=1e-12, verbose=False)

    pred1 = starved_formula(B[0], p, W)
    pred2 = starved_formula(B[1], p, W)
    pred3 = dominant_formula(B[2], p, W)

    print(f"\np={p}")
    print(f"  firm0 actual={np.round(X[0],6)}   predicted={np.round(pred1,6)}")
    print(f"  firm1 actual={np.round(X[1],6)}   predicted={np.round(pred2,6)}")
    print(f"  firm2 actual={np.round(X[2],4)}     predicted={np.round(pred3,4)}")
