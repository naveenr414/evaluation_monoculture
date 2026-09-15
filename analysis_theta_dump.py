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
    centered = theta - theta.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    if s[0] < 1e-12:
        return 0.0
    return float(s[1]/s[0]) if len(s) > 1 else 0.0

def make_step(beta, c=0.0):
    f = lambda x: float(expit(beta*(x-c)))
    fp = lambda x: float(beta*expit(beta*(x-c))*(1-expit(beta*(x-c))))
    return f, fp

f_log = lambda x: np.log(np.clip(x, EPS, None))
fp_log = lambda x: 1.0/np.clip(x, EPS, None)

f_sqrt = lambda x: np.sqrt(np.clip(x,0.0,None))
fp_sqrt = lambda x: 0.5/np.sqrt(np.clip(x,1e-12,None))

def dump(f, fp, label):
    print(f"\n########## f = {label} ##########")
    for name, B in BUDGETS.items():
        sim = CompetitionSimulation(N, K, np.array(B), W, W, f, fp)
        X, _ = sim.find_nash_br(max_iter=400, tol=1e-9, verbose=False)
        theta = sim.theta(X)
        r = collinearity_ratio(theta)
        print(f"-- {name}: B={B}, W={W}")
        print(f"   X=\n{np.round(X,5)}")
        print(f"   theta=\n{np.round(theta,5)}")
        print(f"   collinearity ratio sigma2/sigma1 = {r:.3e}")

dump(f_log, fp_log, "log(x)")
dump(f_sqrt, fp_sqrt, "sqrt(x)")
f,fp = make_step(1000.0, c=0.0)
dump(f, fp, "step c=0 beta=1000")
f,fp = make_step(1000.0, c=1.5)
dump(f, fp, "step c=1.5 beta=1000")
