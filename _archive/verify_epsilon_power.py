import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sig(z): return 1/(1+np.exp(-z))

def make_Us(f1, f2, cost):
    def U1(vars_, opp, w):
        a,b = vars_; c,d = opp
        return -(w[0]*sig(f1(a)-f2(c)) + w[1]*sig(f1(b)-f2(d)) - cost(a+b))
    def U2(vars_, opp, w):
        c,d = vars_; a,b = opp
        return -(w[0]*sig(f2(c)-f1(a)) + w[1]*sig(f2(d)-f1(b)) - cost(c+d))
    return U1, U2

def br(fun, x0, opp, w, bnd):
    return minimize(fun, x0, args=(opp,w), bounds=[bnd,bnd], method='L-BFGS-B').x

def find_eq(U1, U2, w, bnd, iters=300, ninit=5, seed=0):
    best = None; bestval = -1e18
    rng = np.random.default_rng(seed)
    for _ in range(ninit):
        a,b = rng.uniform(0, 1.5, 2)
        c,d = rng.uniform(0, 1.5, 2)
        for _ in range(iters):
            a,b = br(U1, [a,b], (c,d), w, bnd)
            c,d = br(U2, [c,d], (a,b), w, bnd)
        val = -U1([a,b],(c,d),w) - U2([c,d],(a,b),w)
        if val > bestval:
            bestval = val; best = (a,b,c,d)
    return best

w = [0.25, 0.75]
cost = lambda s: s**2
bnd = (0, 10)
base = 0.25
f1 = lambda x: max(x,0)**base

epsilons = np.linspace(-0.2, 0.7, 60)  # exponent ranges 0.05 .. 0.95, stays concave
coss = []
for eps in epsilons:
    p = base + eps
    f2 = lambda x, p=p: max(x,0)**p
    U1, U2 = make_Us(f1, f2, cost)
    a,b,c,d = find_eq(U1, U2, w, bnd)
    F1 = np.array([f1(a), f1(b)])
    F2 = np.array([f2(c), f2(d)])
    n1, n2 = np.linalg.norm(F1), np.linalg.norm(F2)
    cosv = float(np.dot(F1,F2)/(n1*n2)) if n1>1e-9 and n2>1e-9 else np.nan
    coss.append(cosv)
    print(f"eps={eps:+.3f} (p={p:.3f})  x1=({a:.4f},{b:.4f}) x2=({c:.4f},{d:.4f})  F1={F1}  F2={F2}  cos={cosv:.5f}")

coss = np.array(coss)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(epsilons, coss, color="#3b6fd6", lw=1.8, marker='o', ms=3)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"epsilon  ($f_2(x) = x^{0.25+\epsilon}$,  $f_1(x)=x^{0.25}$,  w=[0.25,0.75])")
ax.set_ylabel("cosine similarity of F(x1), F(x2)")
ax.set_title("Cosine similarity vs exponent gap (eps=0 => identical firms)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("verify_epsilon_power.pdf")
print("saved verify_epsilon_power.pdf")
print("cos at eps=0:", coss[np.argmin(np.abs(epsilons))])
print("min cos:", np.nanmin(coss), "at eps=", epsilons[np.nanargmin(coss)])
