"""
Symmetric bidirectional battles: 3 matchup rates (p01, p02, p12) on a simplex.
When matchup (i,j) occurs, both firms i and j participate.

Firm objectives:
  U_0 = p01*sum_k w_k*sigma(f(x0k)-f(x1k)) + p02*sum_k w_k*sigma(f(x0k)-f(x2k))
  U_1 = p01*sum_k w_k*sigma(f(x1k)-f(x0k)) + p12*sum_k w_k*sigma(f(x1k)-f(x2k))
  U_2 = p02*sum_k w_k*sigma(f(x2k)-f(x0k)) + p12*sum_k w_k*sigma(f(x2k)-f(x1k))

"Each firm fights someone" excludes the 3 corners of the simplex (one firm idle).
We enforce p01, p02, p12 >= t_min.

Settings:
  (A) normal B=[5,3,1]
  (B) extreme B=[0.01,0.01,8.98]
  (C) extreme B=[4.43,4.33,0.2]

All at alpha in {0.5, 0.75}.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.special import expit
from scipy.optimize import minimize

N, K = 3, 2
W = np.array([0.7, 0.3])
T_MIN = 0.05   # minimum rate per matchup so all firms fight


def make_power(alpha):
    f  = lambda x: max(x, 0.0) ** alpha
    fp = lambda x: alpha * max(x, 1e-12) ** (alpha - 1)
    return f, fp


def sigma_prime(z):
    s = expit(z); return s * (1.0 - s)


# Row-stochastic T from global matchup rates: each firm normalizes by its fight frequency.
# Firm 0 fights with total rate p01+p02, so it weights opp 1 as p01/(p01+p02), etc.
def T_mat(p01, p02, p12):
    r0 = p01 + p02
    r1 = p01 + p12
    r2 = p02 + p12
    return np.array([
        [0.0,       p01/r0, p02/r0],
        [p01/r1,    0.0,    p12/r1],
        [p02/r2,    p12/r2, 0.0   ],
    ])


def find_nash(B, alpha, T, max_iter=30, tol=1e-5):
    f_fn, _ = make_power(alpha)
    X = np.stack([np.full(K, B[i] / K) for i in range(N)])
    for _ in range(max_iter):
        Xp = X.copy()
        for i in range(N):
            def neg(xi, i=i):
                fxi = np.array([f_fn(xi[k]) for k in range(K)])
                return -sum(T[i,j] * float(np.sum(W * expit(fxi - np.array([f_fn(X[j,k]) for k in range(K)]))))
                            for j in range(N) if j != i)
            Bi = B[i]
            res = minimize(neg, X[i], method="SLSQP", bounds=[(0,Bi)]*K,
                           constraints=[{"type":"ineq","fun":lambda x,Bi=Bi: Bi-np.sum(x)}],
                           options={"ftol":1e-8,"maxiter":200})
            if res.success:
                X[i] = np.clip(res.x, 0, None)
        if np.max(np.abs(X - Xp)) < tol:
            break
    return X


def compute_sigma_ratio_and_ror(X, alpha, T):
    f_fn, _ = make_power(alpha)
    fx = np.array([[f_fn(X[i,k]) for k in range(K)] for i in range(N)])
    Sigma = np.zeros((N, K))
    for i in range(N):
        for j in range(N):
            if j == i: continue
            diff = fx[i] - fx[j]
            Sigma[i] += T[i,j] * np.array([sigma_prime(d) for d in diff])
    # Check degeneracy
    if np.any(Sigma.sum(axis=1) < 1e-8):
        return None, None
    pairs = [(0,1),(0,2),(1,2)]
    rors = {}
    for i,j in pairs:
        ri = Sigma[i,0]/(Sigma[i,1]+1e-12)
        rj = Sigma[j,0]/(Sigma[j,1]+1e-12)
        rors[(i,j)] = ri/(rj+1e-12)
    max_log = max(abs(np.log(max(v,1e-12))) for v in rors.values())
    return rors, max_log


def eval_pt(p01, p02, p12, B, alpha):
    T = T_mat(p01, p02, p12)
    X = find_nash(B, alpha, T)
    rors, v = compute_sigma_ratio_and_ror(X, alpha, T)
    return v if v is not None else np.nan


# -----------------------------------------------------------------------
# Simplex grid (all pts with each coord >= T_MIN)
# -----------------------------------------------------------------------

GRID_N = 30

def simplex_pts(n, t_min):
    pts = []
    for i in range(n+1):
        for j in range(n+1-i):
            k = n - i - j
            p = np.array([i/n, j/n, k/n])
            if np.all(p >= t_min):
                pts.append(p)
    return np.array(pts)

T_pts = simplex_pts(GRID_N, T_MIN)
print(f"Grid: {len(T_pts)} simplex points (t_min={T_MIN})")

SETTINGS = [
    {"label": "Normal B=[5,3,1]",          "B": np.array([5.0,  3.0,  1.0 ]), "alpha": 0.5 },
    {"label": "Normal B=[5,3,1]",          "B": np.array([5.0,  3.0,  1.0 ]), "alpha": 0.75},
    {"label": "Extreme B=[.01,.01,8.98]",  "B": np.array([0.01, 0.01, 8.98]), "alpha": 0.5 },
    {"label": "Extreme B=[4.43,4.33,.2]",  "B": np.array([4.43, 4.33, 0.20]), "alpha": 0.75},
]

all_res = []

for s in SETTINGS:
    B, alpha, label = s["B"], s["alpha"], s["label"]
    print(f"\n{'='*55}\n{label},  alpha={alpha}")

    scores = np.array([eval_pt(*pt, B, alpha) for pt in T_pts])

    best_i = np.nanargmax(scores)
    p_best = T_pts[best_i]
    v_best = scores[best_i]

    v_unif = eval_pt(1/3, 1/3, 1/3, B, alpha)

    T_best = T_mat(*p_best)
    X_best = find_nash(B, alpha, T_best)
    rors, _ = compute_sigma_ratio_and_ror(X_best, alpha, T_best)

    print(f"  Best (p01={p_best[0]:.3f}, p02={p_best[1]:.3f}, p12={p_best[2]:.3f})")
    print(f"  max|log R| = {v_best:.5f}   (uniform 1/3: {v_unif:.5f})")
    for (i,j), r in rors.items():
        print(f"    R({i},{j}) = {r:.5f}")

    all_res.append({"label":label,"alpha":alpha,"B":B,
                    "T_pts":T_pts,"scores":scores,
                    "p_best":p_best,"v_best":v_best,"v_unif":v_unif})

# -----------------------------------------------------------------------
# Plot: 4 simplex heatmaps
# -----------------------------------------------------------------------

# Vertices of equilateral triangle for (p01, p02, p12)
V = np.array([[0.0,0.0],[1.0,0.0],[0.5, np.sqrt(3)/2]])  # p01=1, p02=1, p12=1

fig, axes = plt.subplots(1,4,figsize=(18,5))
fig.suptitle(r"max$|\log R|$ over battle simplex $(p_{01},p_{02},p_{12})$, bidirectional fights",
             fontsize=12, fontweight="bold")

for ax, res in zip(axes, all_res):
    pts = res["T_pts"]
    sc  = res["scores"]

    xy = pts[:,0:1]*V[0] + pts[:,1:2]*V[1] + pts[:,2:3]*V[2]
    x, y = xy[:,0], xy[:,1]

    triang = mtri.Triangulation(x, y)
    tpc = ax.tripcolor(triang, sc, shading="gouraud", cmap="viridis")
    plt.colorbar(tpc, ax=ax, label="|log R|")

    pb = res["p_best"]
    xy_b = pb[0]*V[0] + pb[1]*V[1] + pb[2]*V[2]
    ax.scatter(*xy_b, color="red", s=120, marker="*", zorder=5,
               label=f"opt ({pb[0]:.2f},{pb[1]:.2f},{pb[2]:.2f})\n={res['v_best']:.3f}")

    xy_u = (V[0]+V[1]+V[2])/3
    ax.scatter(*xy_u, color="white", s=60, marker="o", edgecolors="black", lw=1.2, zorder=5,
               label=f"uniform={res['v_unif']:.3f}")

    tri_v = np.vstack([V, V[0]])
    ax.plot(tri_v[:,0], tri_v[:,1], "k-", lw=1)
    off = 0.07
    ax.text(V[0,0]-off, V[0,1]-off, r"$p_{01}=1$"+"\n(0v1 only)", ha="center", fontsize=7)
    ax.text(V[1,0]+off, V[1,1]-off, r"$p_{02}=1$"+"\n(0v2 only)", ha="center", fontsize=7)
    ax.text(V[2,0],     V[2,1]+off, r"$p_{12}=1$"+"\n(1v2 only)", ha="center", fontsize=7)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{res['label']},  α={res['alpha']}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower center")

plt.tight_layout()
plt.savefig("plot_battle_simplex.pdf", bbox_inches="tight")
plt.savefig("plot_battle_simplex.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved: plot_battle_simplex.pdf / .png")
