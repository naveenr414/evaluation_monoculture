"""
Verify empirically that C*log(ax+b) has the same Nash equilibrium as log(x + b/a).

Two claims:
  a) Scale C does not matter: C*log(ax+b) ≡ log(ax+b)
  b) Inner scale a does not matter: log(ax+b) ≡ log(x + b/a)

Combined: C*log(ax+b) ≡ log(x + b/a)
"""

import math, itertools
import numpy as np
from competition_simulation import CompetitionSimulation

N, K = 3, 4
N_SEEDS = 3

def sample_instance(seed):
    rng = np.random.default_rng(seed * 17 + 3)
    B = np.sort(rng.uniform(1.0, 5.0, size=N))[::-1]
    w = rng.dirichlet(np.ones(K))
    return B, w

def make_f(C, a, b):
    f    = lambda x: C * math.log(a * max(x, 0) + b)
    fp   = lambda x: C * a / (a * max(x, 0) + b)
    return f, fp

def make_f_reduced(b_over_a):
    """log(x + b/a) — the claimed equivalent"""
    s = b_over_a
    f    = lambda x: math.log(max(x, 0) + s)
    fp   = lambda x: 1.0 / (max(x, 0) + s)
    return f, fp

def run_nash(f_fn, fp_fn, B, w):
    sim = CompetitionSimulation(N, K, B, w, w.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=100, tol=1e-6, verbose=False)
    return X

# Sweep over (C, a, b) combos — small grid
C_vals = [0.5, 1.0, 2.0]
a_vals = [0.5, 1.0, 2.0]
b_vals = [0.5, 1.0, 2.0]

combos = list(itertools.product(C_vals, a_vals, b_vals))
print(f"Testing {len(combos)} (C, a, b) combinations × {N_SEEDS} seeds\n", flush=True)

max_err_C   = 0.0   # claim a: C doesn't matter
max_err_a   = 0.0   # claim b: a doesn't matter (after b→b/a)
max_err_full = 0.0  # combined: C*log(ax+b) ≡ log(x+b/a)

failures = []

for combo_idx, (C, a, b) in enumerate(combos):
    print(f"  combo {combo_idx+1}/{len(combos)}: C={C}, a={a}, b={b}", flush=True)
    b_over_a = b / a
    f_full, fp_full     = make_f(C, a, b)         # C*log(ax+b)
    f_noC, fp_noC       = make_f(1, a, b)         # log(ax+b)
    f_reduced, fp_red   = make_f_reduced(b_over_a) # log(x + b/a)

    for seed in range(N_SEEDS):
        B, w = sample_instance(seed)
        X_full    = run_nash(f_full,    fp_full,  B, w)
        X_noC     = run_nash(f_noC,     fp_noC,   B, w)
        X_reduced = run_nash(f_reduced, fp_red,   B, w)

        err_C    = np.max(np.abs(X_full - X_noC))
        err_a    = np.max(np.abs(X_noC  - X_reduced))
        err_full = np.max(np.abs(X_full - X_reduced))

        max_err_C    = max(max_err_C,    err_C)
        max_err_a    = max(max_err_a,    err_a)
        max_err_full = max(max_err_full, err_full)

        thresh = 1e-4
        if err_C > thresh or err_a > thresh or err_full > thresh:
            failures.append((C, a, b, seed, err_C, err_a, err_full))

print(f"Max deviation across all combos/seeds:")
print(f"  Claim (a) — C scale:     max|X(C*log(ax+b)) - X(log(ax+b))|  = {max_err_C:.2e}")
print(f"  Claim (b) — a scale:     max|X(log(ax+b))   - X(log(x+b/a))| = {max_err_a:.2e}")
print(f"  Combined  — full equiv:  max|X(C*log(ax+b)) - X(log(x+b/a))| = {max_err_full:.2e}")

if failures:
    print(f"\n{len(failures)} cases exceeded threshold 1e-4:")
    for C, a, b, seed, ec, ea, ef in failures[:10]:
        print(f"  C={C}, a={a}, b={b}, seed={seed}: err_C={ec:.2e} err_a={ea:.2e} err_full={ef:.2e}")
else:
    print(f"\nAll {len(combos)*N_SEEDS} cases within 1e-4. Equivalences confirmed.")
