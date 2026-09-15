"""Stricter recovery test: fit an actual rank-2 factorization (matching the true
generative rank) of the estimated sensitivity matrix S_hat, and compare it to the
truth THREE ways, since individual u_i/v_k vectors are only identified up to an
orthogonal rotation within the top-2 subspace (SVD of a non-orthogonal random
U_true, V_true isn't the same basis as U_true, V_true themselves):

  1. full matrix cosine        : S_hat (all 64 cells) vs true log D          [as before]
  2. rank-2-denoised cosine    : top-2 SVD reconstruction of S_hat vs true log D
  3. subspace overlap          : mean cosine of principal angles between the
                                  TOP-2 firm subspace (and separately category
                                  subspace) of S_hat's SVD vs true log D's SVD
                                  -- this is the rotation-invariant way to ask
                                  "did we recover u_i^T v_k", since the individual
                                  columns of a rank-2 factorization aren't unique.
"""
import numpy as np
from exp_wrank_semisynth import (A_true, GAMMA, SHOCK_SIGMA, SIGMA_W_TOTAL,
                                  build_w_sequence, simulate_wt, two_way_demean,
                                  BURN_IN, N, K, logD_true)

D_true_dm = two_way_demean(logD_true)
Ut, St, Vtt = np.linalg.svd(D_true_dm)
Ut2, Vt2 = Ut[:, :2], Vtt[:2, :].T   # true top-2 firm / category subspaces (K x 2, orthonormal cols)

N_SEEDS = 8


def sensitivity_matrix(w_seq, a_hist):
    wk = w_seq[BURN_IN:]
    dwk = wk - wk.mean(axis=0, keepdims=True)
    a_post = a_hist[BURN_IN:]
    S_hat = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            x = dwk[:, k]
            if x.std() > 1e-9:
                y = a_post[:, i, k]
                S_hat[i, k] = np.sum(x * (y - y.mean())) / np.sum(x * x)
    return S_hat


def subspace_overlap(A, B):
    """mean cosine of principal angles between column spaces of orthonormal A, B."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return s.mean()


def mat_cos(A, B):
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return float(np.sum(A * B) / denom) if denom > 1e-12 else np.nan


print(f'true log D: rank-2 generative, top-2 singular values = {np.round(np.linalg.svd(D_true_dm, compute_uv=False)[:2], 3)}')
print()
print(f'{"rank_r":>7} {"full_cos":>9} {"rank2_cos":>10} {"firm_subsp":>11} {"cat_subsp":>10}   (mean +/- std over 8 seeds)')
for r in [1, 2, 4, 7]:
    full_vals, r2_vals, fs_vals, cs_vals = [], [], [], []
    for s in range(N_SEEDS):
        w_seq = build_w_sequence(r, SIGMA_W_TOTAL, seed=1000 + s)
        theta_hist, X_hist, a_hist = simulate_wt(A_true, w_seq, GAMMA, SHOCK_SIGMA, seed=2000 + s)
        S_hat = sensitivity_matrix(w_seq, a_hist)
        S_dm = two_way_demean(S_hat)

        full_vals.append(mat_cos(S_dm, D_true_dm))

        Uh, Sh, Vth = np.linalg.svd(S_dm)
        S_rank2 = Uh[:, :2] @ np.diag(Sh[:2]) @ Vth[:2, :]
        r2_vals.append(mat_cos(S_rank2, D_true_dm))

        fs_vals.append(subspace_overlap(Uh[:, :2], Ut2))
        cs_vals.append(subspace_overlap(Vth[:2, :].T, Vt2))

    def fmt(v):
        v = np.array(v)
        return f'{v.mean():.3f}+/-{v.std():.3f}'

    print(f'{r:7d} {fmt(full_vals):>9} {fmt(r2_vals):>10} {fmt(fs_vals):>11} {fmt(cs_vals):>10}')
