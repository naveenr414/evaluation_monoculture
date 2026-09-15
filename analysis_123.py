"""
Three analyses:
1. C1 without reasoning models (qwq, deepseek-r1, o1-series)
2. Empirical win rate distribution: 2023 vs 2025 battle data
3. Votes needed to statistically resolve adjacent top-K rankings
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import os, glob

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: C1 without reasoning models
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ANALYSIS 1: C1 cohort analysis — reasoning models removed")
print("=" * 65)

matched = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])

REASONING = {
    'qwq-32b', 'qwq-32b-preview', 'deepseek-r1', 'deepseek-r1-0528',
    'o1-2024-12-17', 'o1-mini', 'o1-preview', 'o3-2025-04-16',
    'o3-mini', 'o3-mini-high', 'o4-mini-2025-04-16',
    'claude-3-7-sonnet-20250219-thinking-32k',
    'gemini-2.5-pro', 'gemini-2.5-flash',  # native thinking
}

def spearman_ci(x, y, n_boot=5000):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
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

# First appearance, all matched models
has_bench = matched[matched['avg_score'].notna()].copy()
first_all = (has_bench.sort_values('leaderboard_publish_date')
             .groupby('model_name').first().reset_index())

# With and without reasoning models
first_no_reason = first_all[~first_all['model_name'].isin(REASONING)]

removed = set(first_all['model_name']) & REASONING
print(f"\nReasoning models removed from matched set: {removed}\n")

for label, df in [('All matched models', first_all),
                  ('Reasoning models removed', first_no_reason)]:
    df = df.copy()
    df['cohort'] = pd.cut(
        df['leaderboard_publish_date'],
        bins=pd.to_datetime(['2023-01-01','2023-07-01','2024-01-01',
                             '2024-07-01','2025-01-01','2025-07-01','2026-01-01']),
        labels=['2023H1','2023H2','2024H1','2024H2','2025H1','2025H2'],
        right=False
    )
    print(f"--- {label} (n={len(df)}) ---")
    print(f"{'Cohort':<10} {'N':>4}  {'ρ':>7}  {'95% CI':>18}  {'p':>7}")
    print("-" * 52)
    for cohort in ['2023H1','2023H2','2024H1','2024H2','2025H1','2025H2']:
        sub = df[df['cohort'] == cohort]
        rho, lo, hi, n = spearman_ci(sub['rating'], sub['avg_score'])
        if np.isnan(rho):
            print(f"{cohort:<10} {len(sub):>4}  {'—':>7}")
            continue
        _, p = stats.spearmanr(sub['rating'].dropna(), sub['avg_score'].dropna())
        print(f"{cohort:<10} {n:>4}  {rho:>7.3f}  [{lo:.3f}, {hi:.3f}]  {p:>7.4f}"
              f"{'*' if p < 0.05 else ''}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Empirical win rate distribution — 2023 vs 2025 battle data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ANALYSIS 2: Empirical win rate distribution 2023 vs 2025")
print("=" * 65)

def load_battles(path, winner_col='winner'):
    df = pd.read_parquet(path)
    return df

# 2023 data
path_2023 = '/Users/naveenr/Downloads/cyclic_optimization/data/data/train-00000-of-00001-cced8514c7ed782a.parquet'
b23 = pd.read_parquet(path_2023)
b23['date'] = pd.to_datetime(b23['tstamp'], unit='s')

# 2025 data — all shards
shard_dir = '/Users/naveenr/Downloads/cyclic_optimization/data/140k'
shards = sorted(glob.glob(os.path.join(shard_dir, '**/*.parquet'), recursive=True))
b25_list = []
for s in shards:
    df = pd.read_parquet(s, columns=['model_a','model_b','winner','timestamp'])
    b25_list.append(df)
b25 = pd.concat(b25_list, ignore_index=True)

def compute_pairwise_winrates(battles, winner_col, min_battles=20):
    """Compute empirical win rate for each model pair."""
    # Normalise winner col: keep only model_a / model_b wins, drop ties
    if winner_col == 'winner':
        w = battles[winner_col].astype(str)
        battles = battles[w.isin(['model_a','model_b'])].copy()
        battles['a_wins'] = (battles[winner_col] == 'model_a').astype(int)
    else:
        battles = battles.copy()
        battles['a_wins'] = (battles[winner_col] == battles['model_a']).astype(int)

    pair_stats = (battles.groupby(['model_a','model_b'])
                  .agg(a_wins=('a_wins','sum'), n=('a_wins','count'))
                  .reset_index())
    pair_stats = pair_stats[pair_stats['n'] >= min_battles]
    pair_stats['win_rate'] = pair_stats['a_wins'] / pair_stats['n']
    # Reflect so all win rates are for higher-ELO model over lower
    pair_stats['win_rate_abs'] = pair_stats['win_rate'].apply(
        lambda x: x if x >= 0.5 else 1 - x
    )
    return pair_stats

wr23 = compute_pairwise_winrates(b23, 'winner', min_battles=10)
wr25 = compute_pairwise_winrates(b25, 'winner', min_battles=20)

print(f"\n2023 dataset: {len(b23):,} battles, {b23['model_a'].nunique()} models A-side")
print(f"2023 pairs with ≥10 battles: {len(wr23)}")
print(f"\n2025 dataset: {len(b25):,} battles, {b25['model_a'].nunique()} models A-side")
print(f"2025 pairs with ≥20 battles: {len(wr25)}")

def wr_summary(wr, label):
    print(f"\n{label} win rate distribution (higher model over lower):")
    wr_abs = wr['win_rate_abs']
    print(f"  mean={wr_abs.mean():.3f}  std={wr_abs.std():.3f}  "
          f"median={wr_abs.median():.3f}")
    print(f"  % pairs within [0.45, 0.55]: "
          f"{((wr['win_rate']>=0.45)&(wr['win_rate']<=0.55)).mean():.1%}")
    print(f"  % pairs within [0.40, 0.60]: "
          f"{((wr['win_rate']>=0.40)&(wr['win_rate']<=0.60)).mean():.1%}")
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    labels = ['50-55%','55-60%','60-65%','65-70%','70-80%','80%+']
    counts = np.histogram(wr_abs, bins=bins)[0]
    total = counts.sum()
    print(f"  Histogram:")
    for lbl, cnt in zip(labels, counts):
        bar = '█' * int(cnt / total * 30)
        print(f"    {lbl}: {bar} {cnt} ({cnt/total:.1%})")

wr_summary(wr23, "2023")
wr_summary(wr25, "2025")

# KS test: are the two win rate distributions different?
ks_stat, ks_p = stats.ks_2samp(wr23['win_rate_abs'], wr25['win_rate_abs'])
print(f"\nKS test (2023 vs 2025 win rate distributions):")
print(f"  KS statistic={ks_stat:.3f}  p={ks_p:.4f}  "
      f"({'significantly different' if ks_p < 0.05 else 'not significantly different'})")

# Mann-Whitney on win_rate_abs
mw_stat, mw_p = stats.mannwhitneyu(wr23['win_rate_abs'], wr25['win_rate_abs'],
                                    alternative='greater')
print(f"Mann-Whitney (2023 win rates > 2025): p={mw_p:.4f}  "
      f"({'significant' if mw_p < 0.05 else 'not significant'})")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Votes needed to resolve adjacent rankings
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("ANALYSIS 3: Votes needed to statistically resolve top-K rankings")
print(f"{'='*65}")

arena = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
arena['leaderboard_publish_date'] = pd.to_datetime(arena['leaderboard_publish_date'])

def votes_needed(elo_i, elo_j, alpha=0.05, power=0.80):
    """
    Votes needed to detect that model i beats model j more than 50% of the time.
    Uses BT win probability and one-sided proportion test sample size formula.
    """
    p = 1 / (1 + np.exp(-(abs(elo_i - elo_j)) / 400))
    if p <= 0.5:
        return np.inf
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    n = ((z_a * np.sqrt(0.25) + z_b * np.sqrt(p * (1-p))) / (p - 0.5)) ** 2
    return n

results = []
for date, snap in arena.groupby('leaderboard_publish_date'):
    snap = snap.sort_values('rating', ascending=False).reset_index(drop=True)
    for K in [5, 10, 20]:
        top = snap.head(K)
        if len(top) < K:
            continue
        # Adjacent pairs only
        adj_votes = []
        for i in range(len(top) - 1):
            n = votes_needed(top.loc[i,'rating'], top.loc[i+1,'rating'])
            adj_votes.append(n)
        # All pairs
        all_votes = []
        for i, j in combinations(range(K), 2):
            n = votes_needed(top.loc[i,'rating'], top.loc[j,'rating'])
            all_votes.append(n)
        results.append({
            'date': date, 'K': K,
            'median_adj_votes': np.nanmedian(adj_votes),
            'mean_adj_votes':   np.nanmean([v for v in adj_votes if np.isfinite(v)]),
            'median_all_votes': np.nanmedian(all_votes),
            'pct_inf_adj': np.mean([np.isinf(v) for v in adj_votes]),
        })

vdf = pd.DataFrame(results)
vdf['year'] = vdf['date'].dt.year
vdf['t'] = (vdf['date'] - vdf['date'].min()).dt.days

for K in [5, 10, 20]:
    sub = vdf[vdf['K'] == K]
    print(f"\n--- Top-{K} adjacent pairs ---")
    print(f"{'Year':<6} {'Snaps':>5}  {'Median adj votes':>17}  "
          f"{'Mean adj votes':>15}  {'% pairs need ∞':>15}")
    print("-" * 62)
    for yr, g in sub.groupby('year'):
        finite = g['median_adj_votes'].replace([np.inf], np.nan)
        print(f"{yr:<6} {len(g):>5}  {finite.median():>17,.0f}  "
              f"{g['mean_adj_votes'].median():>15,.0f}  "
              f"{g['pct_inf_adj'].mean():>15.1%}")

    # Trend on log(votes)
    sub_fin = sub[np.isfinite(sub['median_adj_votes']) & (sub['median_adj_votes'] > 0)]
    if len(sub_fin) > 10:
        sl, _, r, p, _ = stats.linregress(sub_fin['t'],
                                           np.log10(sub_fin['median_adj_votes']))
        print(f"  Log10(votes) trend: slope={sl*365:.3f}/yr  R²={r**2:.3f}  p={p:.4f}  "
              f"({'significant' if p<0.05 else 'not significant'})")
        print(f"  → Votes needed doubling every "
              f"{0.301/abs(sl*365):.1f} years" if sl > 0 else "  → Votes needed decreasing")

# Concrete: how many votes exist vs how many are needed?
print(f"\n--- Concrete comparison: votes available vs votes needed (latest snapshot) ---")
latest = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
latest = latest.sort_values('rating', ascending=False).head(20).reset_index(drop=True)
print(f"{'Rank':<5} {'Model':<45} {'ELO':>7}  {'Gap to next':>12}  {'Votes needed':>13}  {'Votes have':>11}")
print("-" * 100)
for i in range(len(latest)-1):
    gap  = latest.loc[i,'rating'] - latest.loc[i+1,'rating']
    need = votes_needed(latest.loc[i,'rating'], latest.loc[i+1,'rating'])
    have = latest.loc[i,'vote_count']
    flag = ' ← INSUFFICIENT' if (pd.notna(have) and np.isfinite(need) and have < need) else ''
    print(f"{i+1:<5} {latest.loc[i,'model_name']:<45} {latest.loc[i,'rating']:>7.1f}  "
          f"{gap:>12.1f}  {need:>13,.0f}  {have:>11,.0f}{flag}")
