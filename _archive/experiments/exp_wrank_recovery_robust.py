"""Robustness check on the rank(w) -> recovery(F_i) result: repeat across multiple
independent seeds (both the w-basis/coefficient draw and the shock noise), report
mean +/- std of the recovery cosine per rank, instead of a single-seed point estimate.
"""
import numpy as np
from exp_wrank_semisynth import (A_true, GAMMA, SHOCK_SIGMA, SIGMA_W_TOTAL,
                                  build_w_sequence, simulate_wt, recovery)

N_SEEDS = 10
print(f'{"rank_r":>7} {"mean_cos":>9} {"std":>7} {"min":>7} {"max":>7}   all values')
for r in [1, 2, 4, 7]:
    vals = []
    for s in range(N_SEEDS):
        w_seq = build_w_sequence(r, SIGMA_W_TOTAL, seed=1000 + s)
        theta_hist, X_hist, a_hist = simulate_wt(A_true, w_seq, GAMMA, SHOCK_SIGMA, seed=2000 + s)
        vals.append(recovery(w_seq, a_hist))
    vals = np.array(vals)
    print(f'{r:7d} {vals.mean():9.3f} {vals.std():7.3f} {vals.min():7.3f} {vals.max():7.3f}   '
          f'{np.round(vals, 2)}')
