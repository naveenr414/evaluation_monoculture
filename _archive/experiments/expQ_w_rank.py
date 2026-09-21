"""Priority 3: empirical identification rank of the historical (c,w) perturbations.

Theorem 9 says variation in (c,w) identifies F up to pointwise scale -- but only along the
directions that variation actually spans. Using the w_share panel built in expP
(category x month evaluator-attention shares), how many independent directions of
attention-shift occurred historically?

Effective rank = participation ratio (Sum(lambda))^2 / Sum(lambda^2) of the eigenvalues of
the month-to-month change matrix's covariance, benchmarked against two nulls:
  (i) a structural cap: category shares sum to 1, so Delta w_share sums to 0 each month --
      rank is mechanically capped at K-1 = 7 regardless of how "diverse" the changes are.
  (ii) a permutation null: shuffle each category's own time series independently (breaks
       cross-category co-movement, keeps each category's own marginal variance), telling us
       whether categories move together in FEWER effective dimensions than chance.
"""
import numpy as np, pandas as pd

w_share = pd.read_parquet('experiments/_w_share_panel.parquet')
w_share = w_share.sort_index()
K = w_share.shape[1]
print(f'panel: {len(w_share)} months x {K} categories')
print(f'columns: {list(w_share.columns)}\n')

dw = w_share.diff().dropna()
print(f'Delta w_share: {len(dw)} month-to-month changes')
print(f'row sums (should be ~0 by construction, shares sum to 1): '
      f'mean={dw.sum(axis=1).mean():.2e}, max|.|={dw.sum(axis=1).abs().max():.2e}\n')


def participation_ratio(X):
    C = np.cov(X.T)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)
    return (lam.sum() ** 2) / (lam ** 2).sum(), lam[::-1]


pr_obs, lam_obs = participation_ratio(dw.values)
print(f'observed effective rank (participation ratio) = {pr_obs:.2f}  of nominal K={K} '
      f'(structural cap K-1={K-1} from the sum-to-one constraint)')
print(f'eigenvalues (largest first): {np.array2string(lam_obs, precision=1)}')
print(f'share of variance in top 1/2/3 components: '
      f'{lam_obs[:1].sum()/lam_obs.sum():.3f} / {lam_obs[:2].sum()/lam_obs.sum():.3f} / '
      f'{lam_obs[:3].sum()/lam_obs.sum():.3f}\n')

# permutation null: shuffle each category's row order independently
rng = np.random.default_rng(0)
n_perm = 2000
null_pr = []
X = dw.values
for _ in range(n_perm):
    Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    pr_p, _ = participation_ratio(Xp)
    null_pr.append(pr_p)
null_pr = np.array(null_pr)
print(f'permutation null (independent per-category shuffles, n={n_perm}): '
      f'mean={null_pr.mean():.2f}, 5th-95th pct=[{np.percentile(null_pr,5):.2f}, '
      f'{np.percentile(null_pr,95):.2f}]')
z = (pr_obs - null_pr.mean()) / null_pr.std()
p_below = np.mean(null_pr <= pr_obs)
print(f'observed rank {pr_obs:.2f} vs null: z={z:+.2f}, '
      f'fraction of null draws <= observed = {p_below:.3f}\n')

if pr_obs < np.percentile(null_pr, 10):
    verdict = ('Categories move TOGETHER along far fewer independent directions than if their '
               'attention shares moved independently -- the historical (c,w) variation is even '
               'more low-rank than the raw category count suggests.')
elif pr_obs > np.percentile(null_pr, 90):
    verdict = ('Categories move MORE independently than a naive shuffle -- surprising; check for '
               'a data artifact before trusting this.')
else:
    verdict = ('Observed rank is not distinguishable from the permutation null -- consistent with '
               'shares moving close to independently, modulo the mechanical sum-to-one constraint.')
print('>>> ' + verdict)

# connect to §2.4's resolving-power numbers
print('\n=== context: what this rank means for identification ===')
print(f'nominal Arena categories: 8 (mechanically capped at 7 independent w-shift directions)')
print(f'measured effective rank of historical w-shifts: {pr_obs:.2f}')
print(f'cf. §2.4 effective rank of the OUTPUT instrument itself: Arena 3.81 of 8 nominal; '
      f'39-leaf battery 2.52-3.13 of 39 nominal')
print(f'=> the (c,w) perturbations available to identify F are themselves only '
      f'~{pr_obs:.1f}-dimensional, in the same range as the resolving power of the output '
      f'instrument -- Theorem 9\'s identification claim has, at most, a handful of empirically '
      f'realized directions to work with, not 7.')
