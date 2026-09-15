import numpy as np
from competition_simulation import CompetitionSimulation
from scipy.special import expit

K = 3
W = np.array([0.5, 0.3, 0.2])
EPS_FLOOR = 1e-9


def make_power(p):
    f = lambda x: np.clip(x, EPS_FLOOR, None) ** p
    fp = lambda x: p * np.clip(x, EPS_FLOOR, None) ** (p - 1)
    return f, fp


def compute_S(theta, N, K, W):
    S = np.zeros((N, K))
    for i in range(N):
        for ip in range(N):
            if ip != i:
                diff = theta[i] - theta[ip]
                S[i] += W * expit(diff) * (1 - expit(diff))
    return S


def report(p, B1, B2):
    f, fp = make_power(p)
    B = np.array([B1, B2])
    sim = CompetitionSimulation(2, K, B, W, W, f, fp)
    X, _ = sim.find_nash_br(max_iter=500, tol=1e-12, verbose=False)
    theta = sim.theta(X)
    S = compute_S(theta, 2, K, W)
    print(f"\np={p}  B=({B1},{B2})")
    print(f"  X_1={np.round(X[0],6)}   shares={np.round(X[0]/B1,4)}")
    print(f"  X_2={np.round(X[1],8)}   shares={np.round(X[1]/B2,4)}")
    print(f"  theta_1={np.round(theta[0],5)}  theta_2={np.round(theta[1],8)}")
    print(f"  S_1={np.round(S[0],6)}  S_2={np.round(S[1],6)}")
    # test conjecture: x2_k proportional to (w_k * exp(-theta1_k))^(1/(1-p))
    pred = (W * np.exp(-theta[0])) ** (1/(1-p))
    pred = pred / pred.sum() * B2
    print(f"  conjectured x2 shape (∝ [w_k e^-θ1k]^1/(1-p)), scaled to B2: {np.round(pred,8)}")


for p in [0.1, 0.5, 0.9]:
    report(p, 9.0, 0.01)
