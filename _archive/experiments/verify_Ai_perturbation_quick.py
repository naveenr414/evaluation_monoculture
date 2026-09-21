import numpy as np
from scipy.optimize import minimize

def sig(z): return 1.0/(1.0+np.exp(-z))
def make_F(A): return lambda x: A*np.log(np.clip(x,0,None)+1.0)

def firm_utility(x_i, F_i, x_other, F_other, w, cost):
    th_i, th_o = F_i(x_i), F_other(x_other)
    return -(np.sum(w*sig(th_i-th_o)) - cost(np.sum(x_i)))

def best_response(F_i, x_other, F_other, w, cost, K, bnd, x0):
    res = minimize(firm_utility, x0, args=(F_i,x_other,F_other,w,cost),
                    method="L-BFGS-B", bounds=[bnd]*K)
    return res.x

def find_eq(F1, F2, w, cost, K, bnd, iters=40):
    x1 = np.array([bnd[1]/2]*K); x2 = np.array([bnd[1]/2]*K)
    for _ in range(iters):
        x1n = best_response(F1, x2, F2, w, cost, K, bnd, x1)
        x2n = best_response(F2, x1n, F1, w, cost, K, bnd, x2)
        if np.max(np.abs(x1n-x1))<1e-8 and np.max(np.abs(x2n-x2))<1e-8:
            x1,x2 = x1n,x2n
            break
        x1,x2 = x1n,x2n
    return x1,x2

def cosine(u,v):
    nu,nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.dot(u,v)/(nu*nv)) if nu>1e-9 and nv>1e-9 else np.nan

K=2
w=np.array([0.5,0.5])
cost=lambda s: 0.5*s**2
Bmax=5.0
bnd=(0.0,Bmax)
A1=np.array([1.0,1.0])
log_bmax1=np.log(Bmax+1.0)

for delta in [0.0, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0]:
    A2 = A1 + np.array([delta,0.0])
    F1,F2 = make_F(A1), make_F(A2)
    eps = delta*log_bmax1
    x1,x2 = find_eq(F1,F2,w,cost,K,bnd)
    th1,th2 = F1(x1), F2(x2)
    H = cosine(th1,th2)
    print(f"delta={delta:6.3f} eps={eps:6.3f}  H={H:.6f}  1-H={1-H:.3e}")
