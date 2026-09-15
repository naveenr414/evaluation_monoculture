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

def find_eq(U1, U2, w, bnd, iters=250, ninit=4, seed=0):
    best = None; bestval = -1e18
    rng = np.random.default_rng(seed)
    hi = bnd[1] if np.isfinite(bnd[1]) else 3.0
    for _ in range(ninit):
        a,b = rng.uniform(0, hi*0.5, 2)
        c,d = rng.uniform(0, hi*0.5, 2)
        for _ in range(iters):
            a,b = br(U1, [a,b], (c,d), w, bnd)
            c,d = br(U2, [c,d], (a,b), w, bnd)
        val = -U1([a,b],(c,d),w) - U2([c,d],(a,b),w)
        if val > bestval:
            bestval = val; best = (a,b,c,d)
    return best

def cos_curve(f1, f2, cost=lambda s: s**2, bnd=(0,10), npts=45, seed=0):
    U1, U2 = make_Us(f1, f2, cost)
    w1s = np.linspace(0.01, 0.99, npts)
    coss = []
    for w1 in w1s:
        w2 = 1-w1
        a,b,c,d = find_eq(U1, U2, [w1,w2], bnd, seed=seed)
        F1 = np.array([f1(a), f1(b)])
        F2 = np.array([f2(c), f2(d)])
        n1, n2 = np.linalg.norm(F1), np.linalg.norm(F2)
        cos = float(np.dot(F1,F2)/(n1*n2)) if n1>1e-9 and n2>1e-9 else np.nan
        coss.append(cos)
    return w1s, np.array(coss)

cases = [
    ("log(x+1) vs log(x+2)",  lambda x: np.log(max(x,0)+1), lambda x: np.log(max(x,0)+2)),
    ("log(x+1) vs log(x+5)",  lambda x: np.log(max(x,0)+1), lambda x: np.log(max(x,0)+5)),
    ("sqrt(x) vs x^0.25",     lambda x: np.sqrt(max(x,0)),   lambda x: max(x,0)**0.25),
    ("sqrt(x) vs log(x+1)",   lambda x: np.sqrt(max(x,0)),   lambda x: np.log(max(x,0)+1)),
    ("log(x+1) vs 2log(x+1)", lambda x: np.log(max(x,0)+1), lambda x: 2*np.log(max(x,0)+1)),
]

fig, axes = plt.subplots(len(cases), 1, figsize=(6, 3*len(cases)), sharex=True)
for ax, (name, f1, f2) in zip(axes, cases):
    w1s, coss = cos_curve(f1, f2)
    ax.plot(w1s, coss, color="#3b6fd6", lw=1.8)
    ax.axvline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("cosine sim")
    ax.grid(alpha=0.3)
    print(name, "min", np.nanmin(coss), "max", np.nanmax(coss), "at w1=0.5-ish:",
          coss[np.argmin(np.abs(w1s-0.5))])

axes[-1].set_xlabel("w1")
fig.tight_layout()
fig.savefig("verify_parallel.pdf")
print("saved verify_parallel.pdf")
