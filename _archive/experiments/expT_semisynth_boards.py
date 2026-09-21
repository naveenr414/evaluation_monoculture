"""Experiment T (semi-synthetic, 2026-08-10 round): does splitting evaluation into
separate leaderboards preserve capability diversity, starting from the REAL observed
Arena state and using the REAL estimated response law + REAL release residuals?

This upgrades expF (fully stylized Gaussian sim) to semi-synthetic:
  theta_i^0    = real 8-category flagship profile per firm (latest snapshot)
  B_hat        = scalar beta0*I fit on real events (robustness: pooled ridge B)
  epsilon      = bootstrap-resampled REAL per-event residual vectors (not iid Gaussian) --
                 preserves the real cross-category correlation structure of release noise.

Design: L in {1,2,4,8} leaderboards partition the 8 CORE categories into L attention
blocks. w_common = uniform; w^ell upweights its own block. w_i = (1-s) w_common + s w^ell(i).
The relevant frontier for firm i is computed ONLY among rivals sharing its board (both
"who you compete against" and "what's measured" move together under s), falling back to
"no external pull" when a board is a singleton (L=8).

  Delta theta_i = B_hat @ (w_i * gap_i) + eps_resampled,   gap_i = board_frontier_i - theta_i

Main estimand: D_T(s) / D_T(s=0) for each L, after T releases -- how much does board
separation inflate steady-state cross-firm dispersion relative to one shared board.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, firm_profiles, CORE, FRONTIER

K, N = len(CORE), len(FRONTIER)
rng_master = np.random.default_rng(0)

# ---- real calibration: beta0, pooled ridge B, and the real residual pool ----
d = load_panel(cats=CORE)
ev = build_events(d, CORE)
G = ev.pivot_table(index='event', columns='cat', values='gap')[CORE].values
D = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE].values
n_ev = len(G)
beta0 = (G * D).sum() / (G * G).sum()
print(f'real events: {n_ev}, scalar beta0 = {beta0:.4f}')

lam = 1000.0
B_ridge = np.linalg.solve(G.T @ G + lam * np.eye(K), G.T @ D).T

resid_scalar = D - beta0 * G          # n_ev x K, residual pool for the scalar-B model
resid_ridge = D - G @ B_ridge.T       # n_ev x K, residual pool for the ridge-B model
print(f'residual pool: {n_ev} vectors, per-category std = '
      f'{resid_scalar.std(0).round(1).tolist()}')

# ---- real initial state theta_i^0: latest common snapshot with all 8 firms present ----
dts = sorted(d.dt.unique())
theta0 = names = None
for t in dts[::-1]:
    P, V, names = firm_profiles(d, CORE, t)
    if len(names) == N:
        theta0 = P
        print(f'theta_i^0 taken from snapshot {pd.Timestamp(t).date()}, firms: {names}')
        break
assert theta0 is not None, 'no snapshot with all 8 frontier firms present'

# ---- category partition into L blocks (K=8 divides evenly by 1,2,4,8) ----
def board_weight(K, L, own_boost=3.0):
    """w^ell for each of L boards: own block gets relative weight own_boost, rest get 1."""
    block = K // L
    W = np.ones((L, K))
    for ell in range(L):
        W[ell, ell * block:(ell + 1) * block] = own_boost
    return W / W.sum(1, keepdims=True)


def simulate(L, s, B_hat, resid_pool, T=15, reps=300, seed0=1):
    boards = np.arange(N) % L
    Wboard = board_weight(K, L)
    w_common = np.full(K, 1.0 / K)
    w_i = (1 - s) * w_common[None, :] + s * Wboard[boards]   # N x K

    D_traj = np.zeros((reps, T))
    eff_rank_final = np.zeros(reps)
    cos_final = np.zeros(reps)
    dist_final = np.zeros(reps)
    leaders_final = np.zeros(reps)

    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        theta = theta0.copy()
        for t in range(T):
            new = theta.copy()
            for b in np.unique(boards):
                idx = np.where(boards == b)[0]
                for i in idx:
                    rivals = [j for j in idx if j != i]
                    gap = (theta[rivals].max(axis=0) - theta[i]) if len(rivals) else 0.0
                    g = w_i[i] * gap
                    eps = resid_pool[rng.integers(len(resid_pool))]
                    new[i] = theta[i] + B_hat @ g + eps  # eps applies even with no rivals
            theta = new
            D_traj[r, t] = np.mean(np.sum((theta - theta.mean(0)) ** 2, axis=1))

        C = np.cov((theta - theta.mean(0)).T)
        lam_eig = np.clip(np.linalg.eigvalsh(C), 0, None)
        eff_rank_final[r] = lam_eig.sum() ** 2 / (lam_eig ** 2).sum()

        norms = np.linalg.norm(theta, axis=1, keepdims=True)
        cosm = (theta @ theta.T) / (norms @ norms.T)
        iu = np.triu_indices(N, 1)
        cos_final[r] = cosm[iu].mean()
        dmat = np.sqrt(((theta[:, None, :] - theta[None, :, :]) ** 2).sum(-1))
        dist_final[r] = dmat[iu].mean()
        leaders_final[r] = len(np.unique(theta.argmax(0)))

    return dict(D_traj=D_traj.mean(0), D_T=D_traj[:, -1].mean(),
                eff_rank=eff_rank_final.mean(), cos=cos_final.mean(),
                dist=dist_final.mean(), leaders=leaders_final.mean())


print('\n=== main sweep: scalar B_hat = beta0*I, T=15 releases, 300 reps ===')
Ls = [1, 2, 4, 8]
svals = [0.0, 0.25, 0.5, 0.75, 1.0]
B_scalar = beta0 * np.eye(K)

results = {}
for L in Ls:
    base = None
    for s in svals:
        if L == 1 and s > 0:
            continue  # s is meaningless with one common board
        res = simulate(L, s, B_scalar, resid_scalar)
        results[(L, s)] = res
        if L == 1:
            base = res['D_T']
    base = results[(1, 0.0)]['D_T']
    print(f'\nL={L}:')
    print(f'{"s":>5} {"D_T":>9} {"vs L=1":>8} {"eff_rank":>9} {"pair_cos":>9} '
          f'{"pair_dist":>10} {"leaders":>8}')
    for s in ([0.0] if L == 1 else svals):
        res = results[(L, s)]
        print(f'{s:5.2f} {res["D_T"]:9.1f} {res["D_T"]/base:7.2f}x {res["eff_rank"]:9.2f} '
              f'{res["cos"]:9.3f} {res["dist"]:10.2f} {res["leaders"]:8.2f}')

print('\n=== headline: max diversity gain per L (over s), scalar B_hat ===')
for L in Ls:
    if L == 1:
        continue
    ratios = {s: results[(L, s)]['D_T'] / results[(1, 0.0)]['D_T'] for s in svals}
    best_s = max(ratios, key=ratios.get)
    print(f'L={L}: best s={best_s:.2f} -> {ratios[best_s]:.2f}x diversity vs single common board '
          f'(cf. stylized expF: 1.4-2.1x)')

# ---- robustness: pooled ridge B_hat instead of scalar, at a couple of representative L ----
print('\n=== robustness: pooled ridge B_hat (not just isotropic), L in {2,4} ===')
for L in [2, 4]:
    print(f'\nL={L} (ridge B_hat):')
    base_ridge = simulate(1, 0.0, B_ridge, resid_ridge)['D_T']
    for s in svals:
        res = simulate(L, s, B_ridge, resid_ridge)
        print(f's={s:.2f}  D_T={res["D_T"]:9.1f}  ratio={res["D_T"]/base_ridge:6.2f}x  '
              f'eff_rank={res["eff_rank"]:5.2f}  leaders={res["leaders"]:.2f}')

print('\ncaveat: L=8 (singleton boards) has NO rivals ever, so it is a pure unanchored')
print('random walk -- D_T grows with T without bound, not a true steady state. Its ratio')
print('is not comparable to L=2/4, which have a real restoring force toward board-mates.')

print('\n=== interpretation ===')
print('D_T ratio > 1 at s>0 for a given L means genuinely separating what boards reward')
print('(both peer group and category weight) inflates steady-state cross-firm capability')
print('dispersion relative to one shared leaderboard, starting from the REAL observed')
print('profile dispersion and REAL release-noise structure -- not just a stylized Gaussian toy.')
