import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sig(z): return 1/(1+np.exp(-z))

def U(x_i, own_f, opp_x, opp_f, w):
    Fi = own_f(x_i)
    Fj = opp_f(opp_x)
    return -(np.sum(w * sig(Fi - Fj)) - x_i.sum()**2)

def br(x0, own_f, opp_x, opp_f, w, bnd):
    K = len(x0)
    res = minimize(U, x0, args=(own_f, opp_x, opp_f, w), bounds=[bnd]*K, method='L-BFGS-B')
    return res.x

def find_eq(K, f1, f2, w, bnd=(0,10), iters=500, ninit=8, seed=0):
    rng = np.random.default_rng(seed)
    best = None; bestval = -1e18
    for _ in range(ninit):
        x1 = rng.uniform(0, 1.0, K)
        x2 = rng.uniform(0, 1.0, K)
        for _ in range(iters):
            x1 = br(x1, f1, x2, f2, w, bnd)
            x2 = br(x2, f2, x1, f1, w, bnd)
        val = -U(x1, f1, x2, f2, w) - U(x2, f2, x1, f1, w)
        if val > bestval:
            bestval = val; best = (x1.copy(), x2.copy())
    return best

eps = 50.0  # very large -- push firm 2 toward zero effort everywhere
f1 = lambda x: np.log(np.maximum(x,0)+1)
f2 = lambda x: np.log(np.maximum(x,0)+1+eps)

Ks = list(range(2, 9))
coss = []
for K in Ks:
    # skewed weights: one dominant category (0.7), remainder split evenly among the rest
    w = np.full(K, 0.3/(K-1))
    w[0] = 0.7
    x1, x2 = find_eq(K, f1, f2, w, seed=1)
    F1 = f1(x1); F2 = f2(x2)
    n1, n2 = np.linalg.norm(F1), np.linalg.norm(F2)
    cos = float(np.dot(F1,F2)/(n1*n2)) if n1>1e-9 and n2>1e-9 else np.nan
    coss.append(cos)
    print(f"K={K}  x1={np.round(x1,4)}  x2={np.round(x2,4)}  F1={np.round(F1,4)}  cos={cos:.5f}  1/sqrt(K)={1/np.sqrt(K):.5f}")

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(Ks, coss, color="#3b6fd6", lw=1.8, marker='o', ms=6, label="measured cosine similarity")
ax.plot(Ks, [1/np.sqrt(K) for K in Ks], color="#d65f3b", lw=1.5, ls='--', marker='x', ms=6, label=r"$1/\sqrt{K}$")
ax.set_xlabel("K (number of categories)")
ax.set_ylabel("cosine similarity of F(x1), F(x2)")
ax.set_title(r"Cosine similarity floor vs K, as $\epsilon\to\infty$ (skewed weights)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("verify_1_over_sqrtK.pdf")
print("saved verify_1_over_sqrtK.pdf")
