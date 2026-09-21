"""
How close does the actual Nash-equilibrium cos_sim get to the theoretical
floor 1/sqrt(K), in practice, for finite budgets?

Two questions:
  1. With extreme w (near one-hot), does cos_sim -> 1/sqrt(K) even at small p?
  2. With moderate w, does cos_sim -> 1/sqrt(K) only as p -> 1 (as theory predicts),
     and how large a budget does that require?
"""

import numpy as np
from competition_simulation import CompetitionSimulation

EPS_FLOOR = 1e-9


def make_power(p):
    f = lambda x: np.clip(x, EPS_FLOOR, None) ** p
    fp = lambda x: p * np.clip(x, EPS_FLOOR, None) ** (p - 1)
    return f, fp


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def theory_floor(w, p, K):
    q = 1.0 / (1.0 - p)
    s = w ** q
    s = s / s.sum()
    u = s ** p
    return u.sum() / (np.sqrt(K) * np.linalg.norm(u))


def run(B, w, p, K):
    f, fp = make_power(p)
    N = len(B)
    sim = CompetitionSimulation(N, K, np.array(B), w, w, f, fp)
    X, _ = sim.find_nash_br(max_iter=600, tol=1e-12, verbose=False)
    theta = sim.theta(X)
    return theta


print("=" * 78)
print(f"1/sqrt(2) = {1/np.sqrt(2):.5f},   1/sqrt(3) = {1/np.sqrt(3):.5f},   1/sqrt(4) = {1/np.sqrt(4):.5f}")
print("=" * 78)

# -----------------------------------------------------------------------
# K=2: sweep p, for moderate w vs extreme w, at a big budget gap
# -----------------------------------------------------------------------
K = 2
B_SMALL = 0.02
B_LARGE = 2000.0
P_VALS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

print("\nK=2, B_large=2000, B_small=0.02, two-tiny-one-huge")
print(f"{'p':>6} | {'w=(0.7,0.3) sim':>16} {'theory':>8} | {'w=(0.99,0.01) sim':>18} {'theory':>8}")
for p in P_VALS:
    w_mod = np.array([0.7, 0.3])
    w_ext = np.array([0.99, 0.01])

    B = [B_LARGE, B_SMALL, B_SMALL]
    th_mod = run(B, w_mod, p, K)
    cs_mod = cos_sim(th_mod[0], th_mod[1])
    th_ext = run(B, w_ext, p, K)
    cs_ext = cos_sim(th_ext[0], th_ext[1])

    print(f"{p:6.2f} | {cs_mod:16.5f} {theory_floor(w_mod,p,K):8.5f} | "
          f"{cs_ext:18.5f} {theory_floor(w_ext,p,K):8.5f}")

# -----------------------------------------------------------------------
# K=2: for a fixed p, how does the achieved floor depend on B_large
# (does more budget-gap actually help close the gap to 1/sqrt(2))?
# -----------------------------------------------------------------------
print("\nK=2, w=(0.7,0.3), p=0.5: does bigger B_large close the gap to floor?")
w = np.array([0.7, 0.3])
p = 0.5
for B_LARGE in [1, 10, 100, 1000, 10000, 100000]:
    B = [B_LARGE, B_SMALL, B_SMALL]
    th = run(B, w, p, K)
    cs = cos_sim(th[0], th[1])
    print(f"  B_large={B_LARGE:>8}: cos_sim={cs:.5f}  (theory floor={theory_floor(w,p,K):.5f})")

# -----------------------------------------------------------------------
# K=3, K=4: extreme w, does it hit 1/sqrt(K)?
# -----------------------------------------------------------------------
print("\nHigher K, extreme w (all weight ~1 on one dim), p=0.5:")
for K in [2, 3, 4, 5]:
    w = np.zeros(K)
    w[0] = 0.999
    w[1:] = 0.001 / (K - 1)
    p = 0.5
    B = [2000.0] + [0.02] * (K - 1)
    th = run(B, w, p, K)
    # worst pair: large firm vs each small firm (they're identical by symmetry)
    cs = cos_sim(th[0], th[1])
    print(f"  K={K}: cos_sim={cs:.5f}   1/sqrt(K)={1/np.sqrt(K):.5f}   theory={theory_floor(w,p,K):.5f}")
