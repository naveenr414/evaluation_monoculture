"""
C1 robustness check: fix the model set to control for composition confound.
Compute rolling Spearman(ELO_t, benchmark) on a frozen cohort of models
that have been in the leaderboard since the beginning.
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
matched = df[df['avg_score'].notna()].copy()
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])

def spearman_ci(x, y, n_boot=2000, ci=95):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    rho, _ = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot.append(r)
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return rho, lo, hi, len(x)

# ── Define fixed cohorts ──────────────────────────────────────────────────────
first_seen = matched.groupby('model_name')['leaderboard_publish_date'].min()

cutoffs = {
    'All 2023 entrants'  : '2024-01-01',
    'All 2023+2024H1'    : '2024-07-01',
    'All 2023+2024'      : '2025-01-01',
}

for label, cutoff in cutoffs.items():
    cohort_models = first_seen[first_seen < cutoff].index.tolist()
    cohort_data   = matched[matched['model_name'].isin(cohort_models)]

    print(f"\n{'='*65}")
    print(f"Fixed cohort: {label}  ({len(cohort_models)} models, cutoff {cutoff})")
    print(f"{'='*65}")
    print(f"Models: {sorted(cohort_models)}")

    snap_results = []
    for date, grp in cohort_data.groupby('leaderboard_publish_date'):
        rho, lo, hi, n = spearman_ci(grp['rating'], grp['avg_score'])
        snap_results.append({'date': date, 'rho': rho, 'lo': lo, 'hi': hi, 'n': n})

    snaps = pd.DataFrame(snap_results).dropna(subset=['rho'])
    snaps['year'] = snaps['date'].dt.year

    print(f"\n{'Date':<14} {'N':>4}  {'Spearman ρ':>10}  {'95% CI':>20}")
    print("-" * 55)
    step = max(1, len(snaps) // 25)
    for _, row in snaps.iloc[::step].iterrows():
        print(f"{str(row['date'].date()):<14} {int(row['n']):>4}  {row['rho']:>10.3f}"
              f"  [{row['lo']:>6.3f}, {row['hi']:>6.3f}]")

    print(f"\n--- Annual summary ---")
    print(f"{'Year':<6} {'Snaps':>6}  {'Mean ρ':>8}  {'Min ρ':>8}  {'Max ρ':>8}")
    print("-" * 45)
    for yr, g in snaps.groupby('year'):
        print(f"{yr:<6} {len(g):>6}  {g['rho'].mean():>8.3f}  "
              f"{g['rho'].min():>8.3f}  {g['rho'].max():>8.3f}")

    snaps['t'] = (snaps['date'] - snaps['date'].min()).dt.days
    slope, _, r, p, _ = stats.linregress(snaps['t'], snaps['rho'])
    print(f"\nTrend: slope={slope*365:.4f}/yr  R²={r**2:.3f}  p={p:.4f}  "
          f"({'significant' if p < 0.05 else 'not significant'})")

    # Also check: is ELO still varying within this fixed cohort over time?
    print(f"\nELO variance within cohort over time (does it freeze?):")
    elo_var = cohort_data.groupby('leaderboard_publish_date')['rating'].std()
    elo_var.index = pd.to_datetime(elo_var.index)
    for yr in [2023, 2024, 2025, 2026]:
        subset = elo_var[elo_var.index.year == yr]
        if len(subset):
            print(f"  {yr}: mean ELO std = {subset.mean():.1f}  "
                  f"(min={subset.min():.1f}, max={subset.max():.1f})")
