"""
Claim 2 (clarified): is theta = f(x) homogenization across firms caused by
f itself being a step function (e.g. f(x) = 1[x > c]), as opposed to f
being log(x), log(1+x), sqrt(x)?

A literal indicator function isn't strictly increasing/concave (f'=0 a.e.,
breaks the KKT/best-response machinery), so we approximate a "step" f with
a steep logistic in x:

    f_step(x; beta, c) = sigmoid(beta * (x - c))

As beta -> infinity this -> 1[x > c]. We sweep beta from mild (~1, close to
a smooth s-curve) to extreme (~200, essentially a hard step), and compare
theta-homogenization (cosine similarity of theta_i = f(x_i) vectors across
firms) against log(x), log(1+x), sqrt(x) on the same budget configurations
used throughout this project.
"""

import numpy as np
from itertools import combinations
from scipy.special import expit
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


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def min_pairwise_cos(M):
    return min(cos_sim(M[i], M[j]) for i, j in combinations(range(N), 2))


def avg_pairwise_cos(M):
    return np.mean([cos_sim(M[i], M[j]) for i, j in combinations(range(N), 2)])


def make_step(beta, c=1.5):
    f = lambda x: float(expit(beta * (x - c)))
    fp = lambda x: float(beta * expit(beta * (x - c)) * (1 - expit(beta * (x - c))))
    return f, fp


FIXED_FUNCS = {
    "log(x)":     (lambda x: np.log(np.clip(x, EPS, None)), lambda x: 1.0 / np.clip(x, EPS, None)),
    "log(1+x)":   (lambda x: np.log1p(np.clip(x, 0.0, None)), lambda x: 1.0 / (1.0 + np.clip(x, 0.0, None))),
    "sqrt(x)":    (lambda x: np.sqrt(np.clip(x, 0.0, None)), lambda x: 0.5 / np.sqrt(np.clip(x, 1e-12, None))),
}

BETAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def eval_fn(f, fp):
    mins, avgs = [], []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=400, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        mins.append(min_pairwise_cos(theta))
        avgs.append(avg_pairwise_cos(theta))
    return np.mean(mins), np.mean(avgs)


print(f"{'f(x)':<24} {'avg min_cos':>14} {'avg avg_cos':>14}")
print("-" * 54)

for label, (f, fp) in FIXED_FUNCS.items():
    mn, av = eval_fn(f, fp)
    print(f"{label:<24} {mn:14.6f} {av:14.6f}")

print()
for beta in BETAS:
    f, fp = make_step(beta)
    mn, av = eval_fn(f, fp)
    print(f"{'step (beta='+str(beta)+')':<24} {mn:14.6f} {av:14.6f}")

print("\nDone.")
