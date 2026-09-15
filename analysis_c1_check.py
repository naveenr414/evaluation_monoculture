"""
Diagnose the 2025H1 near-zero correlation (n=9).
Check 1: Who are the 9 models, what do their scores look like?
Check 2: Is low correlation due to clustering, outliers, or restricted range?
Check 3: Widen to full 2025 cohort — does the pattern hold?
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
matched = df[df['avg_score'].notna()].copy()
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])

first = (matched.sort_values('leaderboard_publish_date')
         .groupby('model_name').first().reset_index())
first['cohort'] = pd.cut(
    first['leaderboard_publish_date'],
    bins=pd.to_datetime(['2023-01-01','2023-07-01','2024-01-01',
                         '2024-07-01','2025-01-01','2025-07-01','2026-01-01']),
    labels=['2023H1','2023H2','2024H1','2024H2','2025H1','2025H2'],
    right=False
)

def spearman_ci(x, y, n_boot=5000):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return np.nan, np.nan, np.nan, len(x)
    rho, pval = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot.append(r)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rho, lo, hi, len(x)

# ── Check 1: Inspect the 9 models ────────────────────────────────────────────
print("=" * 65)
print("CHECK 1: The 9 models in 2025H1 cohort")
print("=" * 65)

c2025h1 = first[first['cohort'] == '2025H1'].sort_values('rating', ascending=False)
cols = ['model_name', 'leaderboard_publish_date', 'rating', 'avg_score',
        'ifeval', 'bbh', 'math', 'gpqa', 'mmlu_pro', 'params_b']
print(c2025h1[cols].to_string(index=False))

print(f"\nELO   : mean={c2025h1['rating'].mean():.1f}  std={c2025h1['rating'].std():.1f}  "
      f"range=[{c2025h1['rating'].min():.0f}, {c2025h1['rating'].max():.0f}]")
print(f"avg_sc: mean={c2025h1['avg_score'].mean():.1f}  std={c2025h1['avg_score'].std():.1f}  "
      f"range=[{c2025h1['avg_score'].min():.1f}, {c2025h1['avg_score'].max():.1f}]")

# Rank each model by ELO and by benchmark — do the rankings agree?
c2025h1 = c2025h1.copy()
c2025h1['elo_rank']   = c2025h1['rating'].rank(ascending=False).astype(int)
c2025h1['bench_rank'] = c2025h1['avg_score'].rank(ascending=False).astype(int)
c2025h1['rank_diff']  = (c2025h1['elo_rank'] - c2025h1['bench_rank']).abs()
print(f"\nRank comparison (ELO rank vs benchmark rank):")
print(c2025h1[['model_name','elo_rank','bench_rank','rank_diff','rating','avg_score']].to_string(index=False))

# ── Check 2: Compare variance to other cohorts ────────────────────────────────
print("\n" + "=" * 65)
print("CHECK 2: Score variance by cohort (restricted range test)")
print("=" * 65)
print(f"\n{'Cohort':<10}  {'N':>4}  {'ELO std':>8}  {'ELO range':>14}  "
      f"{'Bench std':>10}  {'Bench range':>14}  {'Spearman ρ':>10}")
print("-" * 80)
for cohort in ['2023H1','2023H2','2024H1','2024H2','2025H1']:
    sub = first[first['cohort'] == cohort]
    if len(sub) < 4:
        continue
    rho, lo, hi, n = spearman_ci(sub['rating'], sub['avg_score'])
    elo_r = f"[{sub['rating'].min():.0f},{sub['rating'].max():.0f}]"
    bench_r = f"[{sub['avg_score'].min():.1f},{sub['avg_score'].max():.1f}]"
    print(f"{cohort:<10}  {n:>4}  {sub['rating'].std():>8.1f}  {elo_r:>14}  "
          f"{sub['avg_score'].std():>10.1f}  {bench_r:>14}  {rho:>10.3f}")

# ── Check 3: Widen to full 2025 ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("CHECK 3: Wider cohort definitions")
print("=" * 65)

wider_cohorts = {
    'Full 2025 (H1+H2)' : ('2025-01-01', '2026-01-01'),
    '2024H2+2025H1'     : ('2024-07-01', '2025-07-01'),
    'Post-2024H1'       : ('2024-07-01', '2026-01-01'),
}
for label, (lo_d, hi_d) in wider_cohorts.items():
    sub = first[
        (first['leaderboard_publish_date'] >= lo_d) &
        (first['leaderboard_publish_date'] <  hi_d)
    ]
    rho, lo, hi, n = spearman_ci(sub['rating'], sub['avg_score'])
    _, pval = stats.spearmanr(sub['rating'].dropna(), sub['avg_score'].dropna())
    print(f"\n{label}  (n={n})")
    print(f"  ρ = {rho:.3f}  [{lo:.3f}, {hi:.3f}]  p={pval:.4f}")
    print(f"  Models: {sorted(sub['model_name'].tolist())}")

# ── Check 4: Leave-one-out — which model is driving the near-zero? ────────────
print("\n" + "=" * 65)
print("CHECK 4: Leave-one-out Spearman for 2025H1")
print("=" * 65)
sub = first[first['cohort'] == '2025H1'].reset_index(drop=True)
print(f"\n{'Left out':<45}  {'ρ (n-1)':>8}")
print("-" * 58)
for i, row in sub.iterrows():
    loo = sub.drop(i)
    if len(loo) < 4:
        continue
    rho_loo, _, _, _ = spearman_ci(loo['rating'], loo['avg_score'])
    print(f"{row['model_name']:<45}  {rho_loo:>8.3f}")
