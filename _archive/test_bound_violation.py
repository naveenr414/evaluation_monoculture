import numpy as np
from scipy.optimize import minimize

def sig(z): return 1/(1+np.exp(-z))

def U(x_i, own_f, opp_x, opp_f, w):
    Fi = own_f(x_i)
    Fj = opp_f(opp_x)
    return -(np.sum(w * sig(Fi - Fj)) - x_i.sum()**2)

def br(x0, own_f, opp_x, opp_f, w, bnd):
    K = len(x0)
    res = minimize(U, x0, args=(own_f, opp_x, opp_f, w), bounds=[bnd]*K, method='L-BFGS-B')
    return res.x

def find_eq(K, f1, f2, w, bnd=(0,10), iters=500, ninit=10, seed=0):
    rng = np.random.default_rng(seed)
    best = None; bestval = -1e18
    for _ in range(ninit):
        x1 = rng.uniform(0, 2.0, K)
        x2 = rng.uniform(0, 2.0, K)
        for _ in range(iters):
            x1 = br(x1, f1, x2, f2, w, bnd)
            x2 = br(x2, f2, x1, f1, w, bnd)
        val = -U(x1, f1, x2, f2, w) - U(x2, f2, x1, f1, w)
        if val > bestval:
            bestval = val; best = (x1.copy(), x2.copy())
    return best

def cos(F1,F2):
    n1,n2 = np.linalg.norm(F1), np.linalg.norm(F2)
    return float(np.dot(F1,F2)/(n1*n2)) if n1>1e-9 and n2>1e-9 else np.nan

# "avoid the leader" test: firm 1 has an overwhelming technology edge (very front-loaded, near-step)
# firm 2 has a much weaker/flatter technology. See if firm 2 abandons the high-weight category
# (where it can never catch up) and concentrates entirely in the low-weight one instead.

K = 2
w = np.array([0.9, 0.1])   # category 1 heavily favored

f1 = lambda x: np.maximum(x,0)**0.02   # extremely front-loaded / near-step concave function
f2 = lambda x: np.log(np.maximum(x,0)+1)   # much weaker technology

x1, x2 = find_eq(K, f1, f2, w, seed=2)
F1, F2 = f1(x1), f2(x2)
print("Case A: f1=x^0.02 (dominant tech) vs f2=log(x+1), w=[0.9,0.1]")
print(" x1 =", x1, " x2 =", x2)
print(" F1 =", F1, " F2 =", F2)
print(" cos =", cos(F1,F2), " 1/sqrt(K) =", 1/np.sqrt(K))
print()

# push it further: even more extreme technology gap
f1b = lambda x: np.maximum(x,0)**0.001
x1,x2 = find_eq(K, f1b, f2, w, seed=3)
F1,F2 = f1b(x1), f2(x2)
print("Case B: f1=x^0.001 (near-step) vs f2=log(x+1), w=[0.9,0.1]")
print(" x1 =", x1, " x2 =", x2)
print(" F1 =", F1, " F2 =", F2)
print(" cos =", cos(F1,F2), " 1/sqrt(K) =", 1/np.sqrt(K))
