"""
Characterize the worst-case allocation structure for f(x)=x^p that produces
maximal non-parallelism in theta across firms.
"""

import numpy as np
from itertools import combinations
from competition_simulation import CompetitionSimulation
from scipy.special import expit

K = 2
W = np.array([0.7, 0.3])
EPS_FLOOR = 1e-9


def make_power(p):
    f = lambda x: np.clip(x, EPS_FLOOR, None) ** p
    fp = lambda x: p * np.clip(x, EPS_FLOOR, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def compute_S(theta, N, K, W):
    S = np.zeros((N, K))
    for i in range(N):
        for ip in range(N):
            if ip != i:
                diff = theta[i] - theta[ip]
                S[i] += W * expit(diff) * (1 - expit(diff))
    return S


def report(p, B, label):
    f, fp = make_power(p)
    N = len(B)
    sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=400, tol=1e-11, verbose=False)
    theta = sim.theta(X)
    S = compute_S(theta, N, K, W)
    print(f"\n=== {label}  p={p}  B={np.round(B,6)} ===")
    print(f"X (allocations):\n{np.round(X,6)}")
    print(f"X shares (x/B_i):\n{np.round(X/np.array(B)[:,None],6)}")
    print(f"theta:\n{np.round(theta,6)}")
    print(f"S (competitive pressure per firm/category):\n{np.round(S,6)}")
    print(f"S ratio per firm (S_1/S_2): {np.round(S[:,0]/S[:,1],4)}")
    worst = min(cos_sim(theta[i], theta[j]) for i,j in combinations(range(N),2))
    print(f"worst pairwise cos_sim: {worst:.6f}")


report(0.5, [1.7e-5, 1.7e-5, 9.0 - 3.4e-5], "Extreme two-tiny-one-huge (p=0.5)")
report(0.5, [8.357, 0.0162, 9.9811], "Random-search worst config (p=0.5)")
report(0.9, [1e-5, 1e-5, 9 - 2e-5], "Extreme two-tiny-one-huge (p=0.9)")
report(0.1, [1e-5, 1e-5, 9 - 2e-5], "Extreme two-tiny-one-huge (p=0.1)")
