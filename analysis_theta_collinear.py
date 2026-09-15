"""
Consistent metric across all f: do the raw output vectors theta_i = f(x_i)
(i=1..N firms) lie on a common LINE in R^K (not necessarily through the
origin)?  This is an affine-collinearity test, robust to sign/additive
shifts (unlike raw cosine similarity, which breaks when theta has mixed
signs, e.g. log(x)<0).

Test: center theta by its mean across firms, then look at the top-2
singular values of the centered (N,K) matrix. If sigma_2/sigma_1 ~ 0,
all points lie (numerically) exactly on one line.
"""

import numpy as np
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


def collinearity_ratio(theta):
    """sigma_2 / sigma_1 of centered theta; 0 = exactly collinear."""
    centered = theta - theta.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    if s[0] < 1e-12:
        return 0.0
    return float(s[1] / s[0]) if len(s) > 1 else 0.0


def make_step(beta, c=0.0):
    f = lambda x: float(expit(beta * (x - c)))
    fp = lambda x: float(beta * expit(beta * (x - c)) * (1 - expit(beta * (x - c))))
    return f, fp


FUNCS = {
    "log(x)":     (lambda x: np.log(np.clip(x, EPS, None)), lambda x: 1.0 / np.clip(x, EPS, None)),
    "log(1+x)":   (lambda x: np.log1p(np.clip(x, 0.0, None)), lambda x: 1.0 / (1.0 + np.clip(x, 0.0, None))),
    "sqrt(x)":    (lambda x: np.sqrt(np.clip(x, 0.0, None)), lambda x: 0.5 / np.sqrt(np.clip(x, 1e-12, None))),
    "x^0.8":      (lambda x: np.clip(x, EPS, None) ** 0.8, lambda x: 0.8 * np.clip(x, EPS, None) ** -0.2),
}


def eval_fn(f, fp):
    ratios = []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=400, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        ratios.append(collinearity_ratio(theta))
    return ratios


print(f"{'f(x)':<22} " + " | ".join(f"{n:<20}" for n in BUDGETS))
print("-" * (22 + 23 * len(BUDGETS)))

for label, (f, fp) in FUNCS.items():
    ratios = eval_fn(f, fp)
    print(f"{label:<22} " + " | ".join(f"{r:<20.3e}" for r in ratios))

for beta in [1.0, 10.0, 100.0, 1000.0]:
    f, fp = make_step(beta, c=0.0)
    ratios = eval_fn(f, fp)
    print(f"{'step c=0, beta='+str(beta):<22} " + " | ".join(f"{r:<20.3e}" for r in ratios))

print()
for beta in [1.0, 10.0, 100.0, 1000.0]:
    f, fp = make_step(beta, c=1.5)
    ratios = eval_fn(f, fp)
    print(f"{'step c=1.5, beta='+str(beta):<22} " + " | ".join(f"{r:<20.3e}" for r in ratios))
