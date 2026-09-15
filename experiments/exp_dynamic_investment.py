"""Dynamic investment game: does F_{i,k}(X) = A_{i,k}*log(1+X_{i,k}) with a fixed
quadratic investment cost c(a) = (gamma/2)*a^2 reproduce (1) eval-chasing,
(2) persistence, (3) roughly-constant cross-firm dispersion?

Each period t, firm i observes rivals' last PUBLISHED theta (predetermined,
simultaneous-move) and picks investment a_{i,k} >= 0 in each category to solve

    max_a  w_k * sum_{i' != i} sigma(A[i,k]*log1p(X[i,k]+a) - theta_prev[i',k])
           - (gamma/2) * a^2

This is separable across k (no budget constraint -- cost IS the constraint),
so the firm's problem is N*K independent 1-D concave-ish maximizations per period.
Stock accumulates: X_{i,k,t} = X_{i,k,t-1} + a_{i,k,t}.

Two regimes:
  - shock_sigma = 0     : A is a FIXED heterogeneous ability matrix, no noise.
  - shock_sigma > 0     : an AR(1) shock is added on top of the deterministic
                           A*log1p(X) term each period (mean-reverting, rho=RHO_SHOCK),
                           representing release-to-release idiosyncratic noise.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

N, K = 8, 8
T = 150
BURN_IN = 80
A_CAP = 200.0  # upper bound for the 1-D investment solve


def best_response(Aik, Xik, wk, gamma, rivals_theta):
    def negU(a):
        th = Aik * np.log1p(Xik + a)
        u = wk * np.sum(expit(th - rivals_theta)) - 0.5 * gamma * a * a
        return -u
    res = minimize_scalar(negU, bounds=(0.0, A_CAP), method='bounded',
                           options={'xatol': 1e-8})
    return res.x


def simulate(A, w, gamma, shock_sigma, rho_shock, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((N, K))
    shock = np.zeros((N, K))
    theta_pub = A * np.log1p(X) + shock
    theta_hist = np.zeros((T, N, K))
    X_hist = np.zeros((T, N, K))
    for t in range(T):
        a = np.zeros((N, K))
        for i in range(N):
            others = np.arange(N) != i
            for k in range(K):
                a[i, k] = best_response(A[i, k], X[i, k], w[k], gamma, theta_pub[others, k])
        X = X + a
        shock = rho_shock * shock + shock_sigma * rng.standard_normal((N, K))
        theta_pub = A * np.log1p(X) + shock
        theta_hist[t] = theta_pub
        X_hist[t] = X
    return theta_hist, X_hist


def two_way_demean(M):
    """M: (N,K) -> residual after subtracting row mean, col mean, adding grand mean."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def measure(theta_hist):
    Tn = theta_hist.shape[0]
    # (1) eval-chasing beta: pool all (i,k,t) consecutive-period pairs after burn-in,
    #     two-way (firm x category) demean gap and dtheta each period, like exp1.
    gaps, dthetas = [], []
    for t in range(BURN_IN, Tn - 1):
        th_prev, th_next = theta_hist[t], theta_hist[t + 1]
        frontier = np.array([np.max(np.delete(th_prev[:, k], i)) for k in range(K) for i in range(0)]) \
            if False else None
        gap_t = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                gap_t[i, k] = np.max(np.delete(th_prev[:, k], i)) - th_prev[i, k]
        dtheta_t = th_next - th_prev
        gaps.append(two_way_demean(gap_t).ravel())
        dthetas.append(two_way_demean(dtheta_t).ravel())
    g = np.concatenate(gaps); y = np.concatenate(dthetas)
    beta = np.sum(g * y) / np.sum(g * g)

    # (2) persistence: two-way demean theta EACH period cross-sectionally, then lag-1 autocorr
    resid = np.array([two_way_demean(theta_hist[t]) for t in range(BURN_IN, Tn)])
    r0 = resid[:-1].ravel(); r1 = resid[1:].ravel()
    persistence = np.corrcoef(r0, r1)[0, 1]

    # (3) dispersion over time: std across firms, averaged over categories, each period; trend
    disp = theta_hist[BURN_IN:].std(axis=1).mean(axis=1)  # shape (T-BURN_IN,)
    tt = np.arange(len(disp))
    slope = np.polyfit(tt, disp, 1)[0]
    rel_slope = slope * len(disp) / disp.mean()  # fractional change over the measured window

    return beta, persistence, disp.mean(), rel_slope


rng = np.random.default_rng(42)
alpha_i = np.linspace(0.7, 1.3, N)
a_k = np.ones(K)
gamma = 0.1

print('=== does FIXED structural comparative advantage delta_{i,k} (zero-summed per firm,')
print('    so it is NOT absorbed by the firm fixed effect) generate persistence on its own,')
print('    with only IID (rho=0) shocks driving gap-chasing? ===')
print(f'{"delta_sigma":>11} {"shock_sig":>10} {"rho":>5} {"beta_chase":>11} {"persistence":>12} {"mean_disp":>10} {"disp_%chg":>10}')
for delta_sigma in [0.0, 0.1, 0.2, 0.35]:
    delta_ik = rng.normal(0, delta_sigma, size=(N, K))
    delta_ik -= delta_ik.mean(axis=1, keepdims=True)  # zero-sum per firm across categories
    A = np.outer(alpha_i, a_k) + delta_ik
    A = np.clip(A, 0.05, None)
    w = np.ones(K) / K
    for rho_shock in [0.0, 0.3]:
        for shock_sigma in [1.0, 2.0]:
            theta_hist, X_hist = simulate(A, w, gamma, shock_sigma, rho_shock, seed=1)
            beta, pers, mdisp, rslope = measure(theta_hist)
            print(f'{delta_sigma:11.2f} {shock_sigma:10.2f} {rho_shock:5.1f} {beta:11.3f} {pers:12.3f} {mdisp:10.3f} {rslope:+9.1%}')
