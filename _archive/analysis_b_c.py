"""
Option B: C1 for proprietary models using Kaggle MMLU/MATH/GPQA scores
Option C: CI width / ELO gap ratio — are rankings becoming statistically unresolvable?
"""

import pandas as pd
import numpy as np
from scipy import stats
import re

arena   = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
arena['leaderboard_publish_date'] = pd.to_datetime(arena['leaderboard_publish_date'])
frontier = pd.read_csv('/Users/naveenr/Downloads/cyclic_optimization/data/kaggle_llm_tracker/frontier_models1.csv')
frontier['release_date'] = pd.to_datetime(frontier['release_date'], errors='coerce')

def spearman_ci(x, y, n_boot=5000):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    rho, pval = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot = [stats.spearmanr(
                x[idx := rng.integers(0, len(x), len(x))],
                y[idx])[0]
            for _ in range(n_boot)
            if len(np.unique(x[idx := rng.integers(0, len(x), len(x))])) > 1]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rho, lo, hi, len(x)

# ══════════════════════════════════════════════════════════════════════════════
# OPTION B: Proprietary model C1
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("OPTION B: C1 for proprietary models (Kaggle MMLU/MATH/GPQA)")
print("=" * 65)

def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Build Arena first-appearance ELO map
first_elo = (arena.sort_values('leaderboard_publish_date')
             .groupby('model_name')['rating'].first().to_dict())
arena_norm_map = {norm(m): m for m in first_elo}

# Match Kaggle models to Arena
matched_rows = []
for _, row in frontier.iterrows():
    kn = norm(row['model_name'])
    arena_match = None
    # Try substring match
    for an, am in arena_norm_map.items():
        if len(kn) >= 5 and len(an) >= 5 and (kn in an or an in kn):
            arena_match = am
            break
    if arena_match:
        matched_rows.append({
            'kaggle_name'  : row['model_name'],
            'arena_name'   : arena_match,
            'release_date' : row['release_date'],
            'open_source'  : row['open_source'],
            'arena_elo'    : first_elo[arena_match],
            'mmlu'         : row['mmlu_score'],
            'math'         : row['math_score'],
            'gpqa'         : row['gpqa_score'],
            'humaneval'    : row['humaneval_score'],
        })

mdf = pd.DataFrame(matched_rows).dropna(subset=['arena_elo'])
prop = mdf[mdf['open_source'] == 'No']
print(f"\nMatched: {len(mdf)} total, {len(prop)} proprietary\n")
print(prop[['kaggle_name','release_date','arena_elo','mmlu','math','gpqa']].to_string(index=False))

# Overall correlation: proprietary models
print(f"\n--- Overall Spearman ρ (proprietary, n={len(prop)}) ---")
for bench, col in [('MMLU','mmlu'), ('MATH','math'), ('GPQA','gpqa'), ('HumanEval','humaneval')]:
    sub = prop.dropna(subset=[col])
    rho, lo, hi, n = spearman_ci(sub['arena_elo'], sub[col])
    print(f"  Arena ELO vs {bench:<12}: ρ={rho:.3f}  [{lo:.3f},{hi:.3f}]  n={n}")

# Cohort analysis
print(f"\n--- Cohort analysis: Spearman(ELO, MMLU) by release year ---")
prop = prop.copy()
prop['release_year'] = prop['release_date'].dt.year
print(f"{'Year':<6} {'N':>3}  {'ρ (MMLU)':>9}  {'95% CI':>18}  {'ρ (MATH)':>9}  {'ρ (GPQA)':>9}")
print("-" * 62)
for yr in sorted(prop['release_year'].dropna().unique()):
    sub = prop[prop['release_year'] == yr]
    rm, lm, hm, nm = spearman_ci(sub['arena_elo'], sub['mmlu'])
    rmt, *_ = spearman_ci(sub['arena_elo'], sub['math'])
    rg, *_ = spearman_ci(sub['arena_elo'], sub['gpqa'])
    if np.isnan(rm):
        print(f"{int(yr):<6} {nm:>3}  {'too few':>9}")
        continue
    print(f"{int(yr):<6} {nm:>3}  {rm:>9.3f}  [{lm:.3f},{hm:.3f}]  {rmt:>9.3f}  {rg:>9.3f}")

# Trend: does correlation decline with release date?
prop_clean = prop.dropna(subset=['mmlu','arena_elo','release_date'])
sl, _, r, p, _ = stats.linregress(
    prop_clean['release_date'].astype(np.int64) / 1e18,
    [stats.spearmanr([row['arena_elo']], [row['mmlu']])[0]
     if False else row['mmlu'] for _, row in prop_clean.iterrows()]
)
# Instead: rank correlation between release date and residual from ELO-MMLU fit
prop_clean = prop_clean.copy()
prop_clean['elo_rank']  = prop_clean['arena_elo'].rank()
prop_clean['mmlu_rank'] = prop_clean['mmlu'].rank()
prop_clean['rank_diff'] = (prop_clean['elo_rank'] - prop_clean['mmlu_rank']).abs()
sl2, _, r2, p2, _ = stats.linregress(
    prop_clean['release_date'].dt.year + prop_clean['release_date'].dt.month/12,
    prop_clean['rank_diff']
)
print(f"\nRank divergence (|ELO rank - MMLU rank|) vs release date:")
print(f"  slope={sl2:.3f}/yr  R²={r2**2:.3f}  p={p2:.4f}  "
      f"({'diverging' if sl2 > 0 else 'converging'}, "
      f"{'significant' if p2 < 0.05 else 'not significant'})")

# ══════════════════════════════════════════════════════════════════════════════
# OPTION C: CI width / ELO gap — are rankings becoming unresolvable?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("OPTION C: CI width / ELO gap — ranking resolvability over time")
print(f"{'='*65}")

ci_results = []
for date, snap in arena.groupby('leaderboard_publish_date'):
    snap = snap.dropna(subset=['rating_lower','rating_upper'])
    if len(snap) < 5:
        continue
    snap = snap.sort_values('rating', ascending=False).reset_index(drop=True)
    snap['ci_width'] = snap['rating_upper'] - snap['rating_lower']
    snap['gap_to_next'] = snap['rating'].diff(-1).abs()  # gap to model below
    snap['overlap'] = snap['ci_width'] / 2 + snap['gap_to_next'].shift(-1).fillna(0) / 2

    for K in [5, 10, 20]:
        top = snap.head(K)
        adj_pairs = K - 1
        # Fraction of adjacent pairs whose CIs overlap
        overlapping = 0
        for i in range(len(top) - 1):
            gap = top.loc[i,'rating'] - top.loc[i+1,'rating']
            ci_i = top.loc[i,'ci_width'] / 2
            ci_j = top.loc[i+1,'ci_width'] / 2
            if ci_i + ci_j > gap:
                overlapping += 1
        frac_overlap = overlapping / adj_pairs if adj_pairs > 0 else np.nan
        mean_ratio = (top['ci_width'] / top['gap_to_next'].replace(0, np.nan)).mean()
        mean_ci    = top['ci_width'].mean()
        mean_gap   = snap.head(K+1)['rating'].diff(-1).abs().head(K).mean()

        ci_results.append({
            'date': date, 'K': K,
            'frac_overlap': frac_overlap,
            'mean_ci_width': mean_ci,
            'mean_gap': mean_gap,
            'ci_gap_ratio': mean_ci / mean_gap if mean_gap > 0 else np.nan,
        })

cdf = pd.DataFrame(ci_results)
cdf['year'] = cdf['date'].dt.year
cdf['t'] = (cdf['date'] - cdf['date'].min()).dt.days

for K in [5, 10, 20]:
    sub = cdf[cdf['K'] == K].dropna()
    print(f"\n--- Top-{K} models ---")
    print(f"{'Year':<6} {'Snaps':>5}  {'Frac overlap':>13}  "
          f"{'Mean CI width':>14}  {'Mean ELO gap':>13}  {'CI/gap ratio':>13}")
    print("-" * 70)
    for yr, g in sub.groupby('year'):
        print(f"{yr:<6} {len(g):>5}  {g['frac_overlap'].mean():>13.3f}  "
              f"{g['mean_ci_width'].mean():>14.1f}  "
              f"{g['mean_gap'].mean():>13.1f}  "
              f"{g['ci_gap_ratio'].mean():>13.2f}")

    sl, _, r, p, _ = stats.linregress(sub['t'], sub['frac_overlap'])
    sl2, _, r2, p2, _ = stats.linregress(sub['t'], sub['ci_gap_ratio'])
    print(f"  Frac-overlap trend: slope={sl*365:.4f}/yr  R²={r**2:.3f}  p={p:.4f}  "
          f"({'significant' if p<0.05 else 'not significant'})")
    print(f"  CI/gap ratio trend: slope={sl2*365:.4f}/yr  R²={r2**2:.3f}  p={p2:.4f}  "
          f"({'significant' if p2<0.05 else 'not significant'})")
