"""Experiment F: dose-response for the section 5.3 design levers, calibrated.

Calibrated on Experiment A:
    beta    = 0.4265   per-release gap closure at the current board (K0 = 8)
    sigma^2 = 232.9    innovation variance per release per category
and on Experiment C's finding that the board resolves ~3.8 independent dimensions,
not the nominal 8.

Levers:
  F1  number of scored categories K   (attention per category ~ w_k = 1/K)
  F2  splitting one board into L      (firms chase own board with share s of
                                       their attention, the global frontier with 1-s;
                                       the global term is what keeps levels anchored)
  F3  concentration of w              (uniform -> winner-take-all)
  F4  latent-dimension cap            (innovation is generated in d dimensions and
                                       mapped to K categories, holding TOTAL
                                       innovation fixed; diversity is then measured
                                       rank-aware, as in Experiment C)
"""
import numpy as np

BETA0, SIG2, K0, N = 0.4265, 232.9, 8, 8


def two_way(R):
    R = R - R.mean(1, keepdims=True)
    return R - R.mean(0, keepdims=True)


def simulate(K, beta_k, boards=None, s_own=1.0, d=None,
             rounds=500, burn=250, reps=10):
    """Steady-state cross-firm diversity.

    Returns (per-category dispersion, effective number of diversity dimensions).
    """
    if boards is None:
        boards = np.zeros(N, dtype=int)
    beta_k = np.asarray(beta_k, dtype=float)
    disp_all, rank_all = [], []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        if d is not None:
            L = rng.standard_normal((d, K))
            L /= np.linalg.norm(L, axis=0, keepdims=True) / np.sqrt(1.0)  # unit col norm
        R = rng.standard_normal((N, K)) * np.sqrt(SIG2)
        acc = []
        for t in range(rounds):
            glob = R.max(axis=0)
            new = R.copy()
            for b in np.unique(boards):
                idx = np.where(boards == b)[0]
                if len(idx) < 2:
                    continue
                own = R[idx].max(axis=0)
                target = s_own * own + (1 - s_own) * glob
                new[idx] = R[idx] + beta_k * (target[None, :] - R[idx])
            R = new
            if d is not None:
                # total innovation held fixed: rank-d shock, same per-category variance
                Z = rng.standard_normal((N, d))
                R = R + (Z @ L) * np.sqrt(SIG2)
            else:
                R = R + rng.standard_normal((N, K)) * np.sqrt(SIG2)
            R = two_way(R)
            if t >= burn:
                acc.append(R)
        A = np.stack(acc)
        disp_all.append(A.var(axis=1, ddof=1).mean())
        C = np.cov(np.concatenate(list(A), axis=0).T)
        lam = np.clip(np.linalg.eigvalsh(C), 0, None)
        rank_all.append(lam.sum() ** 2 / (lam ** 2).sum())
    return np.mean(disp_all), np.mean(rank_all)


print(f'calibration: beta={BETA0}  sigma^2={SIG2}  K0={K0}  N={N}\n')

print('=== F1  more scored categories (attention per category falls as 1/K) ===')
print(f'{"K":>4} {"beta_k":>7} {"per-cat disp":>13} {"eff dims":>9} {"vs K=8":>8}')
base = None
for K in [2, 4, 6, 8, 12, 16, 24, 32]:
    bk = np.full(K, min(BETA0 * K0 / K, 0.95))
    m, rk = simulate(K, bk)
    if K == K0:
        base = m
    rel = f'{m/base:7.2f}x' if base else ''
    print(f'{K:4d} {bk[0]:7.3f} {m:13.1f} {rk:9.2f} {rel:>8}')

print('\n=== F2  splitting into L boards; s = share of attention on own board ===')
print(f'{"L":>3} {"s":>5} {"firms/board":>12} {"disp":>9} {"eff dims":>9} {"vs L=1":>8}')
b1 = None
for L in [1, 2, 4]:
    for s in ([1.0] if L == 1 else [0.5, 0.75, 1.0]):
        boards = np.arange(N) % L
        m, rk = simulate(K0, np.full(K0, BETA0), boards=boards, s_own=s)
        if L == 1:
            b1 = m
        note = ' (unanchored)' if (L > 1 and s == 1.0) else ''
        print(f'{L:3d} {s:5.2f} {N//L:12d} {m:9.1f} {rk:9.2f} {m/b1:7.2f}x{note}')

print('\n=== F3  concentration of w (K=8), w_k proportional to k^-a ===')
print(f'{"a":>5} {"HHI(w)":>8} {"disp":>9} {"eff dims":>9} {"vs uniform":>11}')
u = None
for a in [0.0, 0.5, 1.0, 1.5, 2.5]:
    w = (np.arange(1, K0 + 1.0)) ** (-a)
    w /= w.sum()
    bk = np.clip(BETA0 * w * K0, 0, 0.95)
    m, rk = simulate(K0, bk)
    if u is None:
        u = m
    print(f'{a:5.1f} {(w**2).sum():8.4f} {m:9.1f} {rk:9.2f} {m/u:10.2f}x')

print('\n=== F4  latent-dimension cap: total innovation fixed, rank d ===')
print('    (Experiment C measured effective dims = 3.8 against a nominal K = 8)')
print(f'{"K":>4} | {"disp d=inf":>10} {"dims":>6} | {"disp d=4":>9} {"dims":>6}'
      f' | {"disp d=8":>9} {"dims":>6}')
for K in [4, 8, 16, 32]:
    bk = np.full(K, min(BETA0 * K0 / K, 0.95))
    m0, r0 = simulate(K, bk)
    m4, r4 = simulate(K, bk, d=4)
    m8, r8 = simulate(K, bk, d=8)
    print(f'{K:4d} | {m0:10.1f} {r0:6.2f} | {m4:9.1f} {r4:6.2f} | {m8:9.1f} {r8:6.2f}')

print('\nReading: "eff dims" is the diversity analogue of Experiment C\'s effective K.')
print('Under a rank-d cap it stops growing once K passes d, so declaring more')
print('categories over the SAME battle pool cannot buy diversity it cannot express.')
