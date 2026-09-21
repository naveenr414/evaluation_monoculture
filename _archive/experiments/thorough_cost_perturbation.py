"""
Thorough version of the cost-perturbation fragility test.

Start from a near-H=1 equilibrium with A1 != A2 (production functions
genuinely different, but outputs collinear at equilibrium). Perturb the
cost function c0(s)=s^2 by c_eps(s) = s^2 + eps * g(s) for MANY random
smooth, bounded directions g drawn from a random-Fourier basis (sup-norm
normalized so ||c_eps - c0||_inf = eps). For each direction, estimate the
directional derivative d H / d eps at eps=0 via central finite differences,
using equilibrium continuation (warm start) for speed and consistency.

If H=1 (A1 != A2) were "typically preserved" under cost perturbations, most
random directions would give dH/deps ~ 0. The lemma predicts the opposite:
generically dH/deps != 0 (H moves away immediately, to first order), and
only a measure-zero / lower-dimensional set of directions preserve it (or
hit it again at isolated points).
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sig(z): return 1.0 / (1.0 + np.exp(-z))


def make_F(A):
    return lambda x: A @ np.log(np.clip(x, 0, None) + 1.0)


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


def find_eq_multistart(F1, F2, w, cost, K, bnd, ninit=8, iters=100, seed=0):
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


def random_fourier_direction(rng, domain, n_modes=5, max_freq=6.0):
    freqs = rng.uniform(0.5, max_freq, n_modes)
    phases = rng.uniform(0, 2 * np.pi, n_modes)
    coeffs = rng.normal(size=n_modes)
    coeffs /= np.linalg.norm(coeffs)

    def g(s):
        s = np.atleast_1d(s)
        val = sum(c * np.sin(f * s + p) for c, f, p in zip(coeffs, freqs, phases))
        return val

    sup_g = np.max(np.abs(g(domain)))
    return lambda s, g=g, sup_g=sup_g: g(np.atleast_1d(s)) / sup_g


if __name__ == "__main__":
    A1 = np.array([[1.0, 0.2], [0.3, 0.8]])
    A2 = np.array([[0.0025, 1.4142], [0.9497, 0.0082]])
    w = np.array([0.4, 0.6])
    F1, F2 = make_F(A1), make_F(A2)

    K = 2
    bnd = (0.0, 2.0)
    c0 = lambda s: s ** 2
    domain = np.linspace(0, 2.0, 400)

    x1_base, x2_base = find_eq_multistart(F1, F2, w, c0, K, bnd, ninit=8, seed=0)
    th1_base, th2_base = F1(x1_base), F2(x2_base)
    H_base = cosine(th1_base, th2_base)
    print(f"base equilibrium under c0=s^2:  x1={np.round(x1_base,4)} x2={np.round(x2_base,4)}  H={H_base:.6f}")

    n_dirs = 40
    eps_fd = 0.01  # small step for central-difference directional derivative
    rng = np.random.default_rng(7)

    slopes = []
    for d in range(n_dirs):
        g = random_fourier_direction(rng, domain, n_modes=5, max_freq=6.0)

        c_plus = lambda s, g=g: s ** 2 + eps_fd * g(s)[0] if np.isscalar(s) else s ** 2 + eps_fd * g(s)
        c_minus = lambda s, g=g: s ** 2 - eps_fd * g(s)[0] if np.isscalar(s) else s ** 2 - eps_fd * g(s)

        x1p, x2p = find_eq_warm(F1, F2, w, c_plus, K, bnd, x1_base, x2_base, iters=100)
        x1m, x2m = find_eq_warm(F1, F2, w, c_minus, K, bnd, x1_base, x2_base, iters=100)

        Hp = cosine(F1(x1p), F2(x2p))
        Hm = cosine(F1(x1m), F2(x2m))

        slope = (Hp - Hm) / (2 * eps_fd)
        slopes.append(slope)
        print(f"dir {d:2d}: H(+eps)={Hp:.6f}  H(-eps)={Hm:.6f}  dH/deps~{slope:9.4f}")

    slopes = np.array(slopes)
    frac_negligible = np.mean(np.abs(slopes) < 0.01)
    print(f"\nH_base = {H_base:.6f}")
    print(f"dH/deps across {n_dirs} random directions: "
          f"mean={slopes.mean():.4f}  std={slopes.std():.4f}  "
          f"min={slopes.min():.4f}  max={slopes.max():.4f}")
    print(f"fraction of directions with |dH/deps| < 0.01 (~preserving): {frac_negligible:.3f}")
    print(f"fraction with dH/deps clearly nonzero (generic breaking): {1-frac_negligible:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(slopes, bins=15, color="#3b6fd6", edgecolor="white")
    ax.axvline(0, color="#d6603b", lw=2, linestyle="--", label="dH/deps = 0 (preserved)")
    ax.set_xlabel("dH/deps  (directional derivative of H at eps=0)")
    ax.set_ylabel("count (out of %d random cost-perturbation directions)" % n_dirs)
    ax.set_title("Distribution of dH/deps under random cost perturbations\n(H=1-ish equilibrium, A1 != A2)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiments/thorough_cost_perturbation_hist.png", dpi=150)
    print("\nsaved experiments/thorough_cost_perturbation_hist.png")
