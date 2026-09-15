"""
Option A: Discriminate genuine convergence from v-inflation in C4.

Test: among matched models (those with benchmark data), does ELO spread
compress independently of benchmark spread?

- If both ELO and benchmark spread compress → genuine convergence
- If ELO compresses but benchmark spread stays constant → v-inflation
  (Arena can't tell them apart but benchmarks still can)

Also: at each snapshot, does the ELO rank of matched models within the
full Arena still correlate with their benchmark scores?
(i.e., is the Arena still ordering matched models consistently with benchmarks?)
"""

import pandas as pd
import numpy as np
from scipy import stats

matched = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
arena   = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])
arena['leaderboard_publish_date']   = pd.to_datetime(arena['leaderboard_publish_date'])

has_bench = matched[matched['avg_score'].notna()].copy()

# Benchmark scores are fixed per model — compute once
bench_scores = has_bench.groupby('model_name')['avg_score'].first()
bench_std_global = bench_scores.std()
bench_range_global = bench_scores.max() - bench_scores.min()
print(f"Benchmark score spread (fixed, all 79 models):")
print(f"  std={bench_std_global:.2f}  range={bench_range_global:.2f}  "
      f"min={bench_scores.min():.2f}  max={bench_scores.max():.2f}\n")

# ── Analysis 1: ELO spread vs benchmark spread among matched models ───────────
print("=" * 65)
print("ANALYSIS 1: ELO spread vs benchmark spread over time")
print("(among matched models present at each snapshot)")
print("=" * 65)

results = []
for date, grp in has_bench.groupby('leaderboard_publish_date'):
    if len(grp) < 10:
        continue
    elo_std   = grp['rating'].std()
    elo_range = grp['rating'].max() - grp['rating'].min()
    bench_std = grp['avg_score'].std()   # fixed per model, but subset changes
    bench_rng = grp['avg_score'].max() - grp['avg_score'].min()
    # Ratio: if ELO std / bench std is declining, ELO compresses relative to benchmarks
    ratio = elo_std / bench_std if bench_std > 0 else np.nan
    rho, _ = stats.spearmanr(grp['rating'], grp['avg_score'])
    # ELO rank among ALL Arena models at this snapshot
    all_snap = arena[arena['leaderboard_publish_date'] == date].set_index('model_name')['rating']
    grp = grp.copy()
    grp['elo_rank_in_full'] = grp['model_name'].map(
        lambda m: (all_snap > all_snap.get(m, np.nan)).sum() + 1 if m in all_snap else np.nan
    )
    results.append({
        'date': date, 'n': len(grp),
        'elo_std': elo_std, 'elo_range': elo_range,
        'bench_std': bench_std, 'bench_range': bench_rng,
        'elo_bench_ratio': ratio,
        'spearman_rho': rho,
        'n_total_arena': len(all_snap),
        'median_elo_rank': grp['elo_rank_in_full'].median(),
    })

df = pd.DataFrame(results)
df['year'] = df['date'].dt.year
df['t'] = (df['date'] - df['date'].min()).dt.days

print(f"\n{'Year':<6} {'Snaps':>5}  {'ELO std':>8}  {'Bench std':>10}  "
      f"{'ELO/Bench':>10}  {'Spearman ρ':>11}  {'Median ELO rank':>15}")
print("-" * 70)
for yr, g in df.groupby('year'):
    print(f"{yr:<6} {len(g):>5}  {g['elo_std'].mean():>8.1f}  "
          f"{g['bench_std'].mean():>10.2f}  {g['elo_bench_ratio'].mean():>10.2f}  "
          f"{g['spearman_rho'].mean():>11.3f}  {g['median_elo_rank'].mean():>15.1f}")

# Trend tests
for col, label in [('elo_std','ELO std'), ('bench_std','Bench std'),
                   ('elo_bench_ratio','ELO/Bench ratio'), ('spearman_rho','Spearman ρ')]:
    sl, _, r, p, _ = stats.linregress(df['t'], df[col])
    sig = 'significant' if p < 0.05 else 'not significant'
    print(f"\n{label} trend: slope={sl*365:.4f}/yr  R²={r**2:.3f}  p={p:.4f}  ({sig})")

# ── Analysis 2: Top-K matched models — ELO vs benchmark spread ───────────────
print(f"\n{'='*65}")
print("ANALYSIS 2: Top-K matched models by ELO at each snapshot")
print("(Do the highest-ELO matched models compress relative to benchmark?)")
print(f"{'='*65}")

for K in [10, 20]:
    print(f"\n--- Top-{K} matched models ---")
    topk_results = []
    for date, grp in has_bench.groupby('leaderboard_publish_date'):
        top = grp.nlargest(K, 'rating')
        if len(top) < K:
            continue
        topk_results.append({
            'date': date,
            'elo_std':   top['rating'].std(),
            'bench_std': top['avg_score'].std(),
            'elo_range': top['rating'].max() - top['rating'].min(),
            'bench_range': top['avg_score'].max() - top['avg_score'].min(),
            'ratio': top['rating'].std() / top['avg_score'].std() if top['avg_score'].std() > 0 else np.nan,
            'rho': stats.spearmanr(top['rating'], top['avg_score'])[0],
        })
    tk = pd.DataFrame(topk_results)
    tk['year'] = tk['date'].dt.year
    tk['t'] = (tk['date'] - tk['date'].min()).dt.days

    print(f"  {'Year':<6} {'ELO std':>8}  {'Bench std':>10}  {'ELO/Bench':>10}  {'Spearman ρ':>11}")
    print("  " + "-" * 50)
    for yr, g in tk.groupby('year'):
        print(f"  {yr:<6} {g['elo_std'].mean():>8.1f}  {g['bench_std'].mean():>10.2f}  "
              f"{g['ratio'].mean():>10.2f}  {g['rho'].mean():>11.3f}")

    sl, _, r, p, _ = stats.linregress(tk['t'], tk['ratio'])
    print(f"  ELO/Bench ratio trend: slope={sl*365:.4f}/yr  R²={r**2:.3f}  p={p:.4f}  "
          f"({'significant' if p < 0.05 else 'not significant'})")
    sl2, _, r2, p2, _ = stats.linregress(tk['t'], tk['elo_std'])
    sl3, _, r3, p3, _ = stats.linregress(tk['t'], tk['bench_std'])
    print(f"  ELO std trend:         slope={sl2*365:.4f}/yr  R²={r2**2:.3f}  p={p2:.4f}")
    print(f"  Bench std trend:       slope={sl3*365:.4f}/yr  R²={r3**2:.3f}  p={p3:.4f}")

# ── Analysis 3: Top vs bottom matched model ELO/benchmark gap ────────────────
print(f"\n{'='*65}")
print("ANALYSIS 3: ELO gap vs benchmark gap — top half vs bottom half")
print(f"{'='*65}\n")

gap_results = []
for date, grp in has_bench.groupby('leaderboard_publish_date'):
    if len(grp) < 10:
        continue
    grp_sorted = grp.sort_values('rating', ascending=False)
    mid = len(grp_sorted) // 2
    top_half = grp_sorted.head(mid)
    bot_half = grp_sorted.tail(mid)
    elo_gap   = top_half['rating'].mean()   - bot_half['rating'].mean()
    bench_gap = top_half['avg_score'].mean() - bot_half['avg_score'].mean()
    gap_results.append({'date': date, 'elo_gap': elo_gap, 'bench_gap': bench_gap,
                        'ratio': elo_gap / bench_gap if bench_gap != 0 else np.nan})

gdf = pd.DataFrame(gap_results)
gdf['year'] = gdf['date'].dt.year
gdf['t'] = (gdf['date'] - gdf['date'].min()).dt.days

print(f"{'Year':<6} {'ELO gap (top-bot)':>18}  {'Bench gap (top-bot)':>20}  {'Ratio':>8}")
print("-" * 60)
for yr, g in gdf.groupby('year'):
    print(f"{yr:<6} {g['elo_gap'].mean():>18.1f}  {g['bench_gap'].mean():>20.2f}  "
          f"{g['ratio'].mean():>8.2f}")

sl, _, r, p, _ = stats.linregress(gdf['t'], gdf['elo_gap'])
sl2, _, r2, p2, _ = stats.linregress(gdf['t'], gdf['bench_gap'])
print(f"\nELO gap trend:   slope={sl*365:.3f}/yr  R²={r**2:.3f}  p={p:.4f}  "
      f"({'significant' if p < 0.05 else 'not significant'})")
print(f"Bench gap trend: slope={sl2*365:.4f}/yr  R²={r2**2:.3f}  p={p2:.4f}  "
      f"({'significant' if p2 < 0.05 else 'not significant'})")
print(f"\nInterpretation:")
print(f"  ELO gap {'compressing' if sl < 0 else 'expanding'}, "
      f"Bench gap {'compressing' if sl2 < 0 else 'expanding'}")
if sl < 0 and sl2 >= 0:
    print("  -> ELO compresses but benchmarks stay spread = V-INFLATION SIGNATURE")
elif sl < 0 and sl2 < 0:
    print("  -> Both compress = GENUINE CONVERGENCE")
elif sl >= 0:
    print("  -> No ELO compression in this matched subset")
