"""
Rate of "domination reversal": for a pair of firms (i,j) with B_i > B_j,
in how many categories k does the SMALLER-budget firm j actually have
higher output theta_{j,k} > theta_{i,k} (i.e. still wins/dominates that
category despite less budget)?

Rate = (# of (pair, category) instances where lower-budget firm wins)
       / (# of (pair, category) instances total)
"""

import numpy as np
from itertools import combinations
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0
EPS = 1e-6

BUDGETS = {
    "Moderate hetero":      [5.0, 3.0, 1.0],
    "Extreme hetero":       [7.5, 1.2, 0.3],
    "Structure A (2 tiny)": [0.05, 0.05, B_TOTAL - 0.1],
    "Structure B (2 big)":  [4.4, 4.3, 0.3],
}


def make_power(p):
    f = lambda x: np.clip(x, EPS, None) ** p
    fp = lambda x: p * np.clip(x, EPS, None) ** (p - 1)
    return f, fp


def make_log():
    f = lambda x: np.log(np.clip(x, EPS, None))
    fp = lambda x: 1.0 / np.clip(x, EPS, None)
    return f, fp


def domination_stats(f, fp):
    wins, total = 0, 0
    detail = []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        for i, j in combinations(range(N), 2):
            hi, lo = (i, j) if B[i] > B[j] else (j, i)
            if B[hi] == B[lo]:
                continue
            for k in range(K):
                total += 1
                if theta[lo, k] > theta[hi, k]:
                    wins += 1
                    detail.append((name, lo, hi, k, B[lo], B[hi],
                                    theta[lo, k], theta[hi, k]))
    return wins, total, detail


rows = []
for label, (f, fp) in [
    ("x^p, small p (p=0.05)", make_power(0.05)),
    ("sqrt(x)  (p=0.5)",      make_power(0.5)),
    ("log(x)",                make_log()),
    ("linear  (p=1.0)",       make_power(1.0)),
    ("x^p, large p (p=0.95)", make_power(0.95)),
]:
    wins, total, detail = domination_stats(f, fp)
    rate = wins / total
    rows.append((label, wins, total, rate, detail))

print(f"{'f(x)':<24} {'wins':>6} {'total':>6} {'rate':>8}")
print("-" * 48)
for label, wins, total, rate, detail in rows:
    print(f"{label:<24} {wins:6d} {total:6d} {rate:8.3f}")

print("\nDetail of upset wins (lower-budget firm dominates a category):")
for label, wins, total, rate, detail in rows:
    if detail:
        print(f"\n{label}:")
        for name, lo, hi, k, B_lo, B_hi, th_lo, th_hi in detail:
            print(f"  {name}: firm{lo}(B={B_lo:.2f}) beats firm{hi}(B={B_hi:.2f}) "
                  f"in cat{k}  (theta {th_lo:.3f} > {th_hi:.3f})")
