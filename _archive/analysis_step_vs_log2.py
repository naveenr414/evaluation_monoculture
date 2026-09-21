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

def make_step(beta, c=1.5):
    f = lambda x: float(expit(beta * (x - c)))
    fp = lambda x: float(beta * expit(beta * (x - c)) * (1 - expit(beta * (x - c))))
    return f, fp

FIXED_FUNCS = {
    "log(x)":     (lambda x: np.log(np.clip(x, EPS, None)), lambda x: 1.0 / np.clip(x, EPS, None)),
    "log(1+x)":   (lambda x: np.log1p(np.clip(x, 0.0, None)), lambda x: 1.0 / (1.0 + np.clip(x, 0.0, None))),
    "sqrt(x)":    (lambda x: np.sqrt(np.clip(x, 0.0, None)), lambda x: 0.5 / np.sqrt(np.clip(x, 1e-12, None))),
}

def eval_fn(f, fp, verbose=False):
    diffs, shape_hets = [], []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=400, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        diff = theta[:, 0] - theta[:, 1]
        diffs.append(diff.max() - diff.min())
        shape_hets.append(1.0 - min_pairwise_cos(theta))
        if verbose:
            print(f"    [{name}] X=\n{np.round(X,4)}\n    theta=\n{np.round(theta,4)}")
    return np.mean(diffs), np.mean(shape_hets)

print(f"{'f(x)':<24} {'theta1-theta2 spread':>22} {'shape_het(1-min_cos)':>22}")
print("-"*70)
for label, (f, fp) in FIXED_FUNCS.items():
    ds, sh = eval_fn(f, fp)
    print(f"{label:<24} {ds:22.6f} {sh:22.6f}")

print()
for beta in [1.0, 3.0, 10.0, 30.0, 100.0]:
    f, fp = make_step(beta)
    ds, sh = eval_fn(f, fp)
    print(f"{'step (beta='+str(beta)+')':<24} {ds:22.6f} {sh:22.6f}")

print("\n--- Detail: step beta=100 allocations ---")
f, fp = make_step(100.0)
eval_fn(f, fp, verbose=True)
