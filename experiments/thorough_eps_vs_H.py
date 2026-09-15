"""
Thorough version of eps-vs-H: sweep many perturbation DIRECTIONS of A_2
(not just one coordinate), warm-starting equilibria along eps for speed,
and report the worst-case (min H) and typical (mean H) envelope across
directions at each eps magnitude. This is what actually tests
    H_{1,2} >= 1 - O(eps)      (a WORST-CASE bound over all directions)
rather than one arbitrary perturbation path.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sig(z): return 1.0 / (1.0 + np.exp(-z))


def make_F(A):
    return lambda x: A * np.log(np.clip(x, 0, None) + 1.0)


def firm_utility(x_i, F_i, x_other, F_other, w, cost):
    th_i, th_o = F_i(x_i), F_other(x_other)
    return -(np.sum(w * sig(th_i - th_o)) - cost(np.sum(x_i)))


def best_response(F_i, x_other, F_other, w, cost, K, bnd, x0):
    res = minimize(firm_utility, x0, args=(F_i, x_other, F_other, w, cost),
                    method="L-BFGS-B", bounds=[bnd] * K)
    return res.x


def find_eq_warm(F1, F2, w, cost, K, bnd, x1_0, x2_0, iters=60, tol=1e-9):
    x1, x2 = x1_0.copy(), x2_0.copy()
    for _ in range(iters):
        x1n = best_response(F1, x2, F2, w, cost, K, bnd, x1)
        x2n = best_response(F2, x1n, F1, w, cost, K, bnd, x2)
        if np.max(np.abs(x1n - x1)) < tol and np.max(np.abs(x2n - x2)) < tol:
            x1, x2 = x1n, x2n
            break
        x1, x2 = x1n, x2n
    return x1, x2


def find_eq_multistart(F1, F2, w, cost, K, bnd, ninit=6, iters=80, seed=0):
    rng = np.random.default_rng(seed)
    best, best_val = None, -np.inf
    for t in range(ninit):
        x1_0 = rng.uniform(bnd[0], bnd[1], K)
        x2_0 = rng.uniform(bnd[0], bnd[1], K)
        x1, x2 = find_eq_warm(F1, F2, w, cost, K, bnd, x1_0, x2_0, iters=iters)
        val = -firm_utility(x1, F1, x2, F2, w, cost) - firm_utility(x2, F2, x1, F1, w, cost)
        if val > best_val:
            best_val, best = val, (x1.copy(), x2.copy())
    return best


def cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.dot(u, v) / (nu * nv)) if nu > 1e-9 and nv > 1e-9 else np.nan


if __name__ == "__main__":
    K = 2
    w = np.array([0.5, 0.5])
    cost = lambda s: 0.5 * s ** 2
    Bmax = 5.0
    bnd = (0.0, Bmax)
    A1 = np.array([1.0, 1.0])
    log_bmax1 = np.log(Bmax + 1.0)

    # base equilibrium at eps=0 (A2=A1), multi-start to be safe
    F1 = make_F(A1)
    x1_base, x2_base = find_eq_multistart(F1, F1, w, cost, K, bnd, ninit=6, seed=0)

    # direction set: 4 axis-aligned + 4 random unit directions in R^K
    rng = np.random.default_rng(42)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                  np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2)]
    for _ in range(4):
        d = rng.normal(size=K)
        directions.append(d / np.linalg.norm(d))

    eps_grid = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 24)])

    H_by_dir = {}
    for di, d in enumerate(directions):
        x1c, x2c = x1_base.copy(), x2_base.copy()
        Hs = []
        for eps in eps_grid:
            delta = (eps / log_bmax1) * d
            A2 = A1 + delta
            F2 = make_F(A2)
            x1c, x2c = find_eq_warm(F1, F2, w, cost, K, bnd, x1c, x2c, iters=80)
            th1, th2 = F1(x1c), F2(x2c)
            Hs.append(cosine(th1, th2))
        H_by_dir[di] = np.array(Hs)
        print(f"direction {di} ({np.round(d,3)}): H range = "
              f"[{np.nanmin(Hs):.4f}, {np.nanmax(Hs):.4f}] at largest eps={eps_grid[-1]:.3f}")

    H_matrix = np.stack([H_by_dir[di] for di in range(len(directions))])  # (ndir, neps)
    H_worst = np.nanmin(H_matrix, axis=0)
    H_best = np.nanmax(H_matrix, axis=0)
    H_mean = np.nanmean(H_matrix, axis=0)

    print("\neps      H_worst   H_mean    H_best")
    for e, wv, mv, bv in zip(eps_grid, H_worst, H_mean, H_best):
        print(f"{e:8.4f} {wv:9.5f} {mv:9.5f} {bv:9.5f}")

    # fit slope of 1-H_worst vs eps (small-eps regime) to check O(eps) bound
    mask = (eps_grid > 1e-3) & (eps_grid < 0.5) & (1 - H_worst > 0)
    if mask.sum() > 3:
        slope, intercept = np.polyfit(np.log(eps_grid[mask]), np.log(1 - H_worst[mask] + 1e-15), 1)
        print(f"\nWorst-case fit: log(1-H_worst) ~ {slope:.3f} * log(eps) + {intercept:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for di in range(len(directions)):
        axes[0].plot(eps_grid, H_by_dir[di], alpha=0.35, color="gray", lw=1)
    axes[0].plot(eps_grid, H_worst, color="#d6603b", lw=2.2, label="worst-case (min over dirs)")
    axes[0].plot(eps_grid, H_mean, color="#3b6fd6", lw=2.2, label="mean over dirs")
    axes[0].set_xlabel("eps = sup_x ||F1(x)-F2(x)||_inf")
    axes[0].set_ylabel("H_{1,2}")
    axes[0].set_title("H vs eps across 8 perturbation directions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    pos = eps_grid > 0
    axes[1].loglog(eps_grid[pos], 1 - H_worst[pos] + 1e-15, marker='o', ms=4,
                    color="#d6603b", label="worst-case 1-H")
    axes[1].loglog(eps_grid[pos], 1 - H_mean[pos] + 1e-15, marker='o', ms=4,
                    color="#3b6fd6", label="mean 1-H")
    axes[1].set_xlabel("eps (log)")
    axes[1].set_ylabel("1 - H (log)")
    axes[1].set_title("Scaling of 1-H with eps")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("experiments/thorough_eps_vs_H.png", dpi=150)
    print("\nsaved experiments/thorough_eps_vs_H.png")
