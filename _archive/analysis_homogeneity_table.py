import numpy as np
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


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def min_pairwise_cos(theta):
    sims = []
    for i in range(N):
        for j in range(i + 1, N):
            sims.append(cos_sim(theta[i], theta[j]))
    return min(sims)


def eval_fn(f, fp):
    diff_spreads, norm_hets = [], []
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=300, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        diff = theta[:, 0] - theta[:, 1]
        diff_spreads.append(diff.max() - diff.min())
        norm_hets.append(1.0 - min_pairwise_cos(theta))
    return np.mean(diff_spreads), np.mean(norm_hets)


rows = []
for label, (f, fp) in [
    ("x^p, small p (p=0.05)", make_power(0.05)),
    ("sqrt(x)  (p=0.5)",      make_power(0.5)),
    ("log(x)",                make_log()),
    ("x^p, large p (p=0.95)", make_power(0.95)),
]:
    ds, nh = eval_fn(f, fp)
    rows.append((label, ds, nh))

print(f"{'f(x)':<24} {'avg diff_spread':>18}  (theta1-theta2 homogeneity, 0=perfect)")
print(f"{'':<24} {'avg norm_het':>18}  (normalized-shape homogeneity, 0=perfect)")
print("-" * 70)
for label, ds, nh in rows:
    print(f"{label:<24} {ds:18.5f}  {nh:18.5f}")
