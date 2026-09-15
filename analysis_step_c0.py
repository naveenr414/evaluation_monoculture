import numpy as np
from scipy.special import expit
from competition_simulation import CompetitionSimulation

N, K = 3, 2
W = np.array([0.7, 0.3])
B_TOTAL = 9.0

BUDGETS = {
    "Moderate hetero":      [5.0, 3.0, 1.0],
    "Extreme hetero":       [7.5, 1.2, 0.3],
    "Structure A (2 tiny)": [0.05, 0.05, B_TOTAL - 0.1],
}

def make_step(beta, c=0.0):
    f = lambda x: float(expit(beta * (x - c)))
    fp = lambda x: float(beta * expit(beta * (x - c)) * (1 - expit(beta * (x - c))))
    return f, fp

for beta in [10.0, 100.0, 1000.0]:
    print(f"\n=== step, c=0, beta={beta} ===")
    f, fp = make_step(beta, c=0.0)
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=400, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        print(f"  [{name}] X=\n{np.round(X,4)}\n    theta=\n{np.round(theta,4)}")
