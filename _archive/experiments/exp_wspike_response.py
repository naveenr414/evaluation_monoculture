"""Discrete 'spike' shocks instead of smooth continuous w-variation: each period,
ONE category gets a large weight boost (rotating through all K categories so each
gets spiked equally often), the rest share the remainder. Measures:
  1) investment response: how much MORE do firms invest in category k when it's
     spiked, vs when it isn't
  2) productivity response: how much MORE does theta grow in k when spiked
  3) firm-level heterogeneity in response, and whether it recovers true
     comparative advantage (u_i^T v_k) -- same cosine test as the smooth-w case
  4) category-level: does the SIZE of a category's response to being spiked
     depend on its true a_k (baseline category ability)?
"""
import numpy as np
from exp_wrank_semisynth import (A_true, GAMMA, SHOCK_SIGMA, simulate_wt,
                                  two_way_demean, logD_true, N, K, T, BURN_IN, a_k)

D_true_dm = two_way_demean(logD_true)
w_bar = np.ones(K) / K
BOOST = 2.0  # logit boost added to the spiked category


def build_spike_sequence(boost, offset=0):
    w_seq = np.zeros((T, K))
    for t in range(T):
        k_spike = (t + offset) % K
        logits = np.log(w_bar).copy()
        logits[k_spike] += boost
        w = np.exp(logits)
        w_seq[t] = w / w.sum()
    return w_seq


w_seq = build_spike_sequence(BOOST)
print('example spike weight vector (category 3 spiked):', np.round(w_seq[3], 3))
print(f'spiked category gets {w_seq[3].max():.1%} of weight, others share the rest ~{(1-w_seq[3].max())/(K-1):.1%} each')
print()

N_SEEDS = 6
full_cos_vals = []
cat_response_investment = np.zeros(K)
cat_response_theta = np.zeros(K)

for s in range(N_SEEDS):
    theta_hist, X_hist, a_hist = simulate_wt(A_true, w_seq, GAMMA, SHOCK_SIGMA, seed=100 + s)
    dtheta_hist = theta_hist[1:] - theta_hist[:-1]  # (T-1, N, K), dtheta[t] = theta[t+1]-theta[t]

    # boolean: which category was spiked at time t (post burn-in only)
    spiked_at = np.array([(t) % K for t in range(BURN_IN, T)])

    S_hat = np.zeros((N, K))
    for k in range(K):
        is_spike = spiked_at == k
        for i in range(N):
            a_series = a_hist[BURN_IN:, i, k]
            S_hat[i, k] = a_series[is_spike].mean() - a_series[~is_spike].mean()
        # category-level (averaged over firms)
        a_series_allfirms = a_hist[BURN_IN:, :, k]  # (T', N)
        cat_response_investment[k] += (a_series_allfirms[is_spike].mean() - a_series_allfirms[~is_spike].mean()) / N_SEEDS
        # dtheta[t] = theta[t+1] - theta[t], driven by the weight in effect AT time t
        dth = dtheta_hist[BURN_IN:, :, k]  # (T'-1, N), times BURN_IN..T-2
        is_spike_d = spiked_at[:-1] == k
        cat_response_theta[k] += (dth[is_spike_d].mean() - dth[~is_spike_d].mean()) / N_SEEDS

    S_dm = two_way_demean(S_hat)
    denom = np.linalg.norm(S_dm) * np.linalg.norm(D_true_dm)
    full_cos_vals.append(float(np.sum(S_dm * D_true_dm) / denom))

print(f'=== 1) & 2) per-category response to being spiked (avg over {N_SEEDS} seeds) ===')
print(f'{"cat":>4} {"true_a_k":>9} {"invest_excess":>14} {"dtheta_excess":>14}')
for k in range(K):
    print(f'{k:4d} {a_k[k]:9.3f} {cat_response_investment[k]:14.4f} {cat_response_theta[k]:14.4f}')

corr_invest = np.corrcoef(a_k, cat_response_investment)[0, 1]
corr_theta = np.corrcoef(a_k, cat_response_theta)[0, 1]
print(f'\ncorrelation(true a_k, investment excess response): {corr_invest:+.3f}')
print(f'correlation(true a_k, theta-growth excess response): {corr_theta:+.3f}')

print(f'\n=== 3) firm-level recovery from SPIKE design (cosine vs true u_i^T v_k) ===')
print(f'  mean={np.mean(full_cos_vals):.3f}  std={np.std(full_cos_vals):.3f}  '
      f'values={np.round(full_cos_vals, 3)}')
print('  (cf. smooth continuous-w design: rank1=0.47, rank2=0.63, rank4=0.72, rank7=0.70-0.73)')
