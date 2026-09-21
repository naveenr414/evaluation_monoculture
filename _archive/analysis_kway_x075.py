import math, numpy as np, matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import softmax as sp_softmax

N, K = 12, 2
B    = np.linspace(5.0, 1.0, N)
w    = np.array([1/3, 2/3])
k, tau = 4, 0.5

f_fn  = lambda x: max(x, 0)**0.75
fv    = np.vectorize(f_fn)

SUBSETS = {i: list(combinations([j for j in range(N) if j!=i], k-1)) for i in range(N)}

def obj(x_i, i, X):
    X2 = X.copy(); X2[i] = x_i
    th  = fv(X2)
    total = 0.0
    for dk in range(K):
        s = 0.0
        for S in SUBSETS[i]:
            idx = [i] + list(S)
            s  += sp_softmax(th[idx, dk] / tau)[0]
        total += w[dk] * s / len(SUBSETS[i])
    return total

def best_response(i, X):
    Bi  = B[i]
    x0s = [X[i].copy(), np.array([Bi*w[0],Bi*w[1]]),
            np.array([Bi*.97,Bi*.03]), np.array([Bi*.03,Bi*.97])]
    best_x, best_v = X[i].copy(), -np.inf
    for x0 in x0s:
        res = minimize(lambda x: -obj(x,i,X), x0, method="SLSQP",
                       bounds=[(0,Bi)]*K,
                       constraints=[{"type":"ineq","fun":lambda x: Bi-x.sum()}],
                       options={"ftol":1e-9,"maxiter":200})
        if -res.fun > best_v:
            best_v, best_x = -res.fun, np.clip(res.x,0,None)
    return best_x

X = np.column_stack([B*w[0], B*w[1]])
for _ in range(50):
    X_old = X.copy()
    for i in range(N): X[i] = best_response(i, X)
    if np.max(np.abs(X-X_old)) < 1e-5: break

X_prop = np.column_stack([B*w[0], B*w[1]])
shares = X[:,0]/B
sw_val = float(np.dot(w, np.max(fv(X), axis=0)))
sw_prop = float(np.dot(w, np.max(fv(X_prop), axis=0)))
nc = int(np.sum((shares<.05)|(shares>.95)))
print(f"k={k}, tau={tau}, f=x^0.75")
print(f"SW={sw_val:.4f}  (prop={sw_prop:.4f})  corners={nc}/12")
for i in range(N):
    print(f"  firm {i+1:2d} B={B[i]:.2f}: ({X[i,0]:.3f}, {X[i,1]:.3f})  share_0={shares[i]:.3f}")

# plot
pt_colors = plt.cm.plasma(np.linspace(0.05, 0.95, N))
t = np.linspace(0, B.max(), 200)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(w[0]*t, w[1]*t, "k--", lw=1, alpha=0.4, label="proportional ray")
for bi in B:
    ax.plot([bi,0],[0,bi],"-",color="gray",lw=0.4,alpha=0.15)
for i in range(N):
    ax.scatter(X[i,0], X[i,1], color=pt_colors[i], s=80, marker="o", zorder=5)
    ax.scatter(X_prop[i,0], X_prop[i,1], color=pt_colors[i], s=180,
               marker="*", zorder=6, edgecolors="k", linewidths=0.4, alpha=0.6)
    if i < 4 or i >= N-2:
        ax.annotate(f"B={B[i]:.1f}", (X[i,0], X[i,1]),
                    textcoords="offset points", xytext=(5,3), fontsize=7, color=pt_colors[i])

ax.scatter([],[],color="gray",s=80, marker="o",label="Nash")
ax.scatter([],[],color="gray",s=180,marker="*",edgecolors="k",linewidths=0.4,alpha=0.6,label="Proportional")
ax.set_xlabel("$x_{i,1}$  (dim 0)", fontsize=11)
ax.set_ylabel("$x_{i,2}$  (dim 1)", fontsize=11)
ax.set_title(
    f"$k$-way match: $k={k}$, $\\tau={tau}$, $f(x)=x^{{0.75}}$\n"
    f"$N=12$, $w=[1/3,\\,2/3]$  |  SW={sw_val:.3f}  (prop={sw_prop:.3f})  —  {nc}/12 corners",
    fontsize=10, fontweight="bold"
)
ax.legend(fontsize=9)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("plot_kway_x075.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("Saved: plot_kway_x075.pdf")
