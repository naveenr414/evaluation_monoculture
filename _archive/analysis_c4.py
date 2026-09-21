"""
C4 Analysis: Are pairwise win rates among frontier models compressing toward 0.5?

At each snapshot, take top-K models, compute all pairwise BT win rates,
measure variance of those win rates and ELO spread. Test for trend over time.

Also checks: full leaderboard spread, new-entrant-only spread.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

arena = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
arena['leaderboard_publish_date'] = pd.to_datetime(arena['leaderboard_publish_date'])

def bt_winrate(elo_i, elo_j, scale=400):
    """Bradley-Terry win probability for i over j."""
    return 1 / (1 + np.exp(-(elo_i - elo_j) / scale))

def pairwise_stats(ratings):
    """Given a list of ELO ratings, compute pairwise win rate variance and spread."""
    ratings = np.array(ratings)
    if len(ratings) < 2:
        return np.nan, np.nan, np.nan, np.nan
    pairs = list(combinations(range(len(ratings)), 2))
    win_rates = [bt_winrate(ratings[i], ratings[j]) for i, j in pairs]
    # Win rates in (0.5, 1) — reflect to get full [0,1] range
    all_wr = win_rates + [1 - w for w in win_rates]
    return (
        np.var(win_rates),           # variance of win rates (higher = more spread)
        np.std(ratings),             # ELO std among this group
        np.mean(np.abs(np.diff(sorted(ratings, reverse=True)))),  # mean gap between adjacent models
        np.max(ratings) - np.min(ratings),  # ELO range
    )

results = []
dates = sorted(arena['leaderboard_publish_date'].unique())

for date in dates:
    snap = arena[arena['leaderboard_publish_date'] == date].sort_values('rating', ascending=False)
    n_total = len(snap)

    for K in [5, 10, 20]:
        top = snap.head(K)
        if len(top) < K:
            continue
        wr_var, elo_std, mean_gap, elo_range = pairwise_stats(top['rating'].values)
        results.append({
            'date': date,
            'K': K,
            'wr_var': wr_var,
            'elo_std': elo_std,
            'mean_gap': mean_gap,
            'elo_range': elo_range,
            'n_total': n_total,
        })

    # Full leaderboard (all models present)
    wr_var, elo_std, mean_gap, elo_range = pairwise_stats(snap['rating'].values)
    results.append({
        'date': date, 'K': 'all',
        'wr_var': wr_var, 'elo_std': elo_std,
        'mean_gap': mean_gap, 'elo_range': elo_range,
        'n_total': n_total,
    })

df = pd.DataFrame(results)
df['year'] = df['date'].dt.year
df['t'] = (df['date'] - df['date'].min()).dt.days

# ── Report ────────────────────────────────────────────────────────────────────
for K in [5, 10, 20, 'all']:
    sub = df[df['K'] == K].dropna(subset=['wr_var'])
    print(f"\n{'='*65}")
    print(f"Top-{K} models per snapshot  ({len(sub)} snapshots)")
    print(f"{'='*65}")

    # Annual summary
    print(f"\n{'Year':<6}  {'Snaps':>5}  {'Mean WR-var':>11}  {'Mean ELO-std':>12}  {'Mean ELO-range':>14}")
    print("-" * 55)
    for yr, g in sub.groupby('year'):
        print(f"{yr:<6}  {len(g):>5}  {g['wr_var'].mean():>11.5f}  "
              f"{g['elo_std'].mean():>12.1f}  {g['elo_range'].mean():>14.1f}")

    # Trend test on win rate variance
    slope, intercept, r, p, se = stats.linregress(sub['t'], sub['wr_var'])
    print(f"\nWR-variance trend: slope={slope*365:.6f}/yr  R²={r**2:.3f}  p={p:.4f}  "
          f"({'decreasing' if slope < 0 else 'increasing'}, "
          f"{'significant' if p < 0.05 else 'not significant'})")

    # Trend test on ELO std
    slope2, _, r2, p2, _ = stats.linregress(sub['t'], sub['elo_std'])
    print(f"ELO-std trend:     slope={slope2*365:.3f}/yr      R²={r2**2:.3f}  p={p2:.4f}  "
          f"({'decreasing' if slope2 < 0 else 'increasing'}, "
          f"{'significant' if p2 < 0.05 else 'not significant'})")

    # Print every ~15th snapshot for readability
    print(f"\n{'Date':<14}  {'WR-var':>9}  {'ELO-std':>8}  {'ELO-range':>10}  {'N-total':>7}")
    print("-" * 55)
    step = max(1, len(sub) // 20)
    for _, row in sub.iloc[::step].iterrows():
        print(f"{str(row['date'].date()):<14}  {row['wr_var']:>9.5f}  "
              f"{row['elo_std']:>8.1f}  {row['elo_range']:>10.1f}  {int(row['n_total']):>7}")

# ── New entrants only ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("New entrants only (models first seen in past 180 days at each snapshot)")
print(f"{'='*65}")

first_seen = arena.groupby('model_name')['leaderboard_publish_date'].min().to_dict()
new_results = []
for date in dates:
    snap = arena[arena['leaderboard_publish_date'] == date]
    cutoff = date - pd.Timedelta(days=180)
    new = snap[snap['model_name'].map(first_seen) >= cutoff]
    if len(new) < 4:
        continue
    wr_var, elo_std, mean_gap, elo_range = pairwise_stats(new['rating'].values)
    new_results.append({'date': date, 'n': len(new),
                        'wr_var': wr_var, 'elo_std': elo_std, 'elo_range': elo_range})

ndf = pd.DataFrame(new_results)
ndf['year'] = ndf['date'].dt.year
ndf['t'] = (ndf['date'] - ndf['date'].min()).dt.days

print(f"\n{'Year':<6}  {'Snaps':>5}  {'Mean WR-var':>11}  {'Mean ELO-std':>12}  {'Mean N-new':>10}")
print("-" * 50)
for yr, g in ndf.groupby('year'):
    print(f"{yr:<6}  {len(g):>5}  {g['wr_var'].mean():>11.5f}  "
          f"{g['elo_std'].mean():>12.1f}  {g['n'].mean():>10.1f}")

slope, _, r, p, _ = stats.linregress(ndf['t'], ndf['wr_var'])
print(f"\nWR-variance trend (new entrants): slope={slope*365:.6f}/yr  "
      f"R²={r**2:.3f}  p={p:.4f}  "
      f"({'decreasing' if slope < 0 else 'increasing'}, "
      f"{'significant' if p < 0.05 else 'not significant'})")
