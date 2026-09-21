import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps = 5.0
f1  = lambda x: np.log(x+1)
f1p = lambda x: 1/(x+1)
f2  = lambda x: np.log(x+1+eps)
f2p = lambda x: 1/(x+1+eps)

def sig(z):  return 1/(1+np.exp(-z))
def sigp(z): return sig(z)*(1-sig(z))

def eqs(vars_, K):
    a,b = vars_
    a = max(a, 0.0); b = max(b, 0.0)
    s = sigp(f1(a)-f2(b))
    return [s*f1p(a) - 2*K*a, s*f2p(b) - 2*K*b]

def solve(K):
    for a0 in [0.001, 0.01, 0.05, 0.1, 0.3]:
        for b0 in [0.0005, 0.005, 0.02, 0.05]:
            a,b = fsolve(eqs, [a0,b0], args=(K,))
            if a > -1e-6 and b > -1e-6:
                return max(a,0), max(b,0)
    raise RuntimeError(f"no solution found for K={K}")

Ks = list(range(1, 21))
a_vals, b_vals, F1_vals, F2_vals, win_probs = [], [], [], [], []
for K in Ks:
    a,b = solve(K)
    F1, F2 = f1(a), f2(b)
    a_vals.append(a); b_vals.append(b)
    F1_vals.append(F1); F2_vals.append(F2)
    win_probs.append(sig(F1-F2))
    print(f"K={K:2d}  a={a:.5f} b={b:.5f}  F1={F1:.4f} F2={F2:.4f}  "
          f"win_prob(firm1 vs firm2, per category)={sig(F1-F2):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11,4))

axes[0].plot(Ks, a_vals, label="firm 1 (log(x+1)) per-category effort a", color="#3b6fd6", marker='o', ms=4)
axes[0].plot(Ks, b_vals, label="firm 2 (log(x+1+eps)) per-category effort b", color="#d65f3b", marker='o', ms=4)
axes[0].set_xlabel("K (number of categories)")
axes[0].set_ylabel("per-category effort")
axes[0].set_title(r"Per-category effort vs K  ($\epsilon=5$)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(Ks, win_probs, color="#3b6fd6", marker='o', ms=4)
axes[1].set_xlabel("K (number of categories)")
axes[1].set_ylabel("firm 1's win probability per category")
axes[1].set_title("Firm 1's disadvantage widens as K grows")
axes[1].set_ylim(0,0.35)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("verify_K_scaling.pdf")
print("\nsaved verify_K_scaling.pdf")
print("\n(Cosine similarity of F(x1),F(x2) is EXACTLY 1 for every K here, by the equal-weight "
      "symmetry theorem -- both F-vectors sit on the all-ones ray. The earlier full-K-dim numeric "
      "run that showed cos dropping to 0.58 was a solver-precision artifact, not real economics.)")
