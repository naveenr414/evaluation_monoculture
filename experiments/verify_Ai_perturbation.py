"""
Test the conjecture:
    sup_x ||F_1(x) - F_2(x)||_inf <= eps   =>   H_{1,2} >= 1 - O(eps)

Setting: N=2 firms, K=2 categories, production functions
    F_i(x) = A_i * log(x + 1)      (elementwise, A_i in R^K_{>0})
Firm 1 is fixed at A_1 = (1, 1). Firm 2's coefficient A_2 = A_1 + delta is
perturbed by delta in [-eps, eps]^K (worst-case direction searched), so that
    sup_x ||F_1(x)-F_2(x)||_inf = eps * log(Bmax+1)
over the feasible allocation domain x in [0, Bmax].

For each eps we solve the two-firm Nash equilibrium of the win-rate game
(Bradley-Terry sigmoid), read off theta_i = F_i(x_i*), and compute
H_{1,2} = cos(theta_1, theta_2).
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_F(A):
    # F(x) elementwise: A_k * log(x_k + 1)
    return lambda x: A * np.log(np.clip(x, 0, None) + 1.0)


def firm_utility(x_i, F_i, x_other, F_other, w, cost):
    th_i = F_i(x_i)
    th_o = F_other(x_other)
    win = np.sum(w * sig(th_i - th_o))
    return -(win - cost(np.sum(x_i)))


def best_response(F_i, x_other, F_other, w, cost, K, bnd, n_restarts=6, seed=0):
    rng = np.random.default_rng(seed)
    best_x, best_val = None, np.inf
    x0s = [rng.uniform(bnd[0], bnd[1], K) for _ in range(n_restarts)]
    for x0 in x0s:
        res = minimize(
            firm_utility, x0, args=(F_i, x_other, F_other, w, cost),
            method="L-BFGS-B", bounds=[bnd] * K,
        )
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    return best_x


def find_equilibrium(F1, F2, w, cost, K, bnd, iters=200, ninit=4, seed=0):
    rng = np.random.default_rng(seed)
    best, best_val = None, -np.inf
    for trial in range(ninit):
        x1 = rng.uniform(bnd[0], bnd[1], K)
        x2 = rng.uniform(bnd[0], bnd[1], K)
        for _ in range(iters):
            x1_new = best_response(F1, x2, F2, w, cost, K, bnd, seed=seed + trial)
            x2_new = best_response(F2, x1_new, F1, w, cost, K, bnd, seed=seed + trial + 100)
            if np.max(np.abs(x1_new - x1)) < 1e-9 and np.max(np.abs(x2_new - x2)) < 1e-9:
                x1, x2 = x1_new, x2_new
                break
            x1, x2 = x1_new, x2_new
        val = -firm_utility(x1, F1, x2, F2, w, cost) - firm_utility(x2, F2, x1, F1, w, cost)
        if val > best_val:
            best_val, best = val, (x1.copy(), x2.copy())
    return best


def cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))


if __name__ == "__main__":
    K = 2
    w = np.array([0.5, 0.5])
    cost = lambda s: 0.5 * s ** 2
    Bmax = 5.0
    bnd = (0.0, Bmax)
    A1 = np.array([1.0, 1.0])

    log_bmax1 = np.log(Bmax + 1.0)

    # worst-case perturbation direction: push A2 apart in a single coordinate
    # (maximizes sup_x||F1-F2||_inf for a given delta magnitude)
    deltas = np.concatenate([[0.0], np.geomspace(1e-4, 2.0, 40)])

    eps_list, H_list = [], []
    for delta in deltas:
        A2 = A1 + np.array([delta, 0.0])  # perturb only coordinate 0
        F1 = make_F(A1)
        F2 = make_F(A2)

        eps = delta * log_bmax1  # sup_x ||F1(x)-F2(x)||_inf over x in [0,Bmax]

        x1, x2 = find_equilibrium(F1, F2, w, cost, K, bnd, seed=1)
        theta1, theta2 = F1(x1), F2(x2)
        H = cosine(theta1, theta2)

        eps_list.append(eps)
        H_list.append(H)
        print(f"delta={delta:.5f} eps={eps:.5f}  x1={np.round(x1,4)} x2={np.round(x2,4)}  "
              f"theta1={np.round(theta1,4)} theta2={np.round(theta2,4)}  H={H:.6f}  1-H={1-H:.3e}")

    eps_arr = np.array(eps_list)
    H_arr = np.array(H_list)

    # check O(eps) / O(eps^2) scaling of (1-H) for small eps
    mask = (eps_arr > 1e-3) & (eps_arr < 0.5)
    if mask.sum() > 2:
        slope, intercept = np.polyfit(np.log(eps_arr[mask]), np.log(1 - H_arr[mask] + 1e-15), 1)
        print(f"\nFit: log(1-H) ~ {slope:.3f} * log(eps) + {intercept:.3f}  "
              f"(slope~1 => 1-H = O(eps); slope~2 => O(eps^2))")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(eps_arr, H_arr, marker='o', ms=3, color="#3b6fd6")
    axes[0].set_xscale("symlog", linthresh=1e-3)
    axes[0].set_xlabel("eps = sup_x ||F1(x)-F2(x)||_inf")
    axes[0].set_ylabel("H_{1,2} = cos(theta1, theta2)")
    axes[0].set_title("Heterogeneity vs. perturbation eps")
    axes[0].grid(alpha=0.3)

    pos = eps_arr > 0
    axes[1].loglog(eps_arr[pos], 1 - H_arr[pos] + 1e-15, marker='o', ms=3, color="#d6603b")
    axes[1].set_xlabel("eps (log scale)")
    axes[1].set_ylabel("1 - H  (log scale)")
    axes[1].set_title("Scaling of 1-H with eps")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("experiments/verify_Ai_perturbation.pdf")
    print("\nsaved experiments/verify_Ai_perturbation.pdf")
