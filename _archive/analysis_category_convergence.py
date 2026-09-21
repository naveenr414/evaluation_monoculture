"""
Category popularity × benchmark convergence analysis.

Question: Do Arena categories that receive more votes (higher optimization
pressure) also show faster convergence in benchmark scores?

Three pieces stitched together:
1. Arena category vote shares — from raw leaderboard cumulative vote counts
2. Benchmark convergence rates — slope of top-50 std 2023→2025 (OLLMv2)
3. 2025 battle data category win-rate std — cross-sectional convergence proxy

Benchmark → Arena category mapping:
  IFEval       → instruction_following
  MATH Lvl 5   → math
  BBH          → hard_prompts_english (reasoning-heavy)
  GPQA         → hard_prompts_english (domain-expert)
  MMLU-PRO     → general (no specific Arena category)
  MUSR         → general (no specific Arena category)
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob, ast

N_BOOT = 5000
BENCHMARKS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']

# ── 1. Arena category vote shares ─────────────────────────────────────────────
print("=" * 65)
print("1. ARENA CATEGORY VOTE SHARES (cumulative, latest snapshot)")
print("=" * 65)

arena_raw = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
arena_raw['leaderboard_publish_date'] = pd.to_datetime(arena_raw['leaderboard_publish_date'])
latest_date = arena_raw['leaderboard_publish_date'].max()
latest = arena_raw[arena_raw['leaderboard_publish_date'] == latest_date]

ARENA_CATS = ['coding', 'instruction_following', 'math', 'creative_writing',
              'hard_prompts_english', 'longer_query', 'multi_turn']

cat_votes = {}
for cat in ARENA_CATS:
    sub = latest[latest['category'] == cat]
    cat_votes[cat] = sub['vote_count'].sum()
total_votes = sum(cat_votes.values())

vote_df = pd.DataFrame([
    {'category': c, 'votes': v, 'share': v / total_votes}
    for c, v in cat_votes.items()
]).sort_values('votes', ascending=False).reset_index(drop=True)
print(f"\nTotal category-specific votes: {total_votes:,.0f}")
print(vote_df.to_string(index=False))

# ── 2. Benchmark convergence rates (OLLMv2) ───────────────────────────────────
print("\n" + "=" * 65)
print("2. BENCHMARK CONVERGENCE RATES (top-50 by avg_score per year)")
print("=" * 65)

bench = pd.read_parquet('data/benchmarks.parquet')
bench['upload_date'] = pd.to_datetime(bench['upload_date'])
bench['year'] = bench['upload_date'].dt.year

# top-50 frontier models by avg_score within each upload year
print("\nBenchmark std per year (top-50 frontier models):")
hdr = f"{'benchmark':<12} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8} {'slope/yr':>10} {'direction':>12}"
print(hdr)
print("-" * len(hdr))

conv_rates = {}
year_stds = {}
for bm in BENCHMARKS:
    rows = []
    for yr in [2022, 2023, 2024, 2025]:
        sub = bench[(bench['year'] == yr)].dropna(subset=['avg_score', bm])
        top50 = sub.nlargest(50, 'avg_score')
        rows.append({'year': yr, 'std': top50[bm].std(), 'mean': top50[bm].mean(), 'n': len(top50)})
    df_bm = pd.DataFrame(rows)
    year_stds[bm] = df_bm.set_index('year')['std'].to_dict()

    # Slope 2023→2025
    sub23_25 = df_bm[df_bm['year'].isin([2023, 2024, 2025])]
    slope, _, r, _, _ = stats.linregress(sub23_25['year'], sub23_25['std'])
    conv_rates[bm] = slope
    direction = 'converging' if slope < 0 else 'diverging'
    print(f"{bm:<12} {year_stds[bm].get(2022, np.nan):>8.2f} "
          f"{year_stds[bm][2023]:>8.2f} {year_stds[bm][2024]:>8.2f} "
          f"{year_stds[bm][2025]:>8.2f} {slope:>+10.2f} {direction:>12}")

# ── 3. 2025 battle category win-rate std ──────────────────────────────────────
print("\n" + "=" * 65)
print("3. 2025 BATTLE DATA — category win-rate std (cross-section)")
print("=" * 65)

files_25 = sorted(glob.glob('data/140k/data/*.parquet'))
df25 = pd.concat([pd.read_parquet(f) for f in files_25], ignore_index=True)

def parse_tag(t):
    if isinstance(t, dict): return t
    try: return ast.literal_eval(str(t))
    except: return {}

tags = df25['category_tag'].apply(parse_tag)
df25['_math'] = tags.apply(lambda x: x.get('math_v0.1', {}).get('math', False))
df25['_creative'] = tags.apply(lambda x: x.get('creative_writing_v0.1', {}).get('creative_writing', False))
df25['_if'] = tags.apply(lambda x: x.get('if_v0.1', {}).get('if', False))

def categorize(row):
    if row['_math']: return 'math'
    if row['_creative']: return 'creative_writing'
    if row['is_code']: return 'coding'
    if row['_if']: return 'instruction_following'
    return 'general'

df25['category'] = df25.apply(categorize, axis=1)
BATTLE_CATS = ['math', 'coding', 'creative_writing', 'instruction_following', 'general']

# Per-model win rate per category (min 50 battles)
records = {}
for cat in BATTLE_CATS:
    sub = df25[df25['category'] == cat].copy()
    sub['wa'] = sub['winner'].map({'model_a': 1., 'tie': 0.5, 'model_b': 0.})
    sub['wb'] = 1. - sub['wa']
    combined = pd.concat([
        sub[['model_a', 'wa']].rename(columns={'model_a': 'model', 'wa': 'w'}),
        sub[['model_b', 'wb']].rename(columns={'model_b': 'model', 'wb': 'w'})
    ])
    agg = combined.groupby('model')['w'].agg(['mean', 'count'])
    agg.columns = [cat + '_wr', cat + '_n']
    records[cat] = agg

wr = pd.concat([records[c][c + '_wr'] for c in BATTLE_CATS], axis=1)
wr.columns = BATTLE_CATS
for cat in BATTLE_CATS:
    n_col = records[cat][cat + '_n'].reindex(wr.index)
    wr.loc[n_col < 50, cat] = np.nan
full_wr = wr.dropna()

battle_cat_votes = df25['category'].value_counts()

print(f"\nCategory win-rate std (n={len(full_wr)} models with >=50 battles/cat):")
battle_rows = []
for cat in BATTLE_CATS:
    std_val = full_wr[cat].std()
    n_battles = battle_cat_votes.get(cat, 0)
    arena_cat = cat  # same names for coding, math, creative_writing, instruction_following
    arena_votes = cat_votes.get(cat, None)
    battle_rows.append({
        'category': cat,
        'wr_std': std_val,
        'battles_2025': n_battles,
        'arena_cumulative_votes': arena_votes
    })
    print(f"  {cat:<25}: wr_std={std_val:.4f}  battles={n_battles:,}  "
          f"arena_votes={'N/A' if arena_votes is None else f'{arena_votes:,.0f}'}")

# ── 4. Joint table: popularity × convergence ──────────────────────────────────
print("\n" + "=" * 65)
print("4. JOINT TABLE: Arena category popularity × benchmark convergence")
print("=" * 65)

# Benchmark → Arena category mapping
BM_TO_ARENA = {
    'ifeval':    'instruction_following',
    'math':      'math',
    'bbh':       'hard_prompts_english',
    'gpqa':      'hard_prompts_english',
    'mmlu_pro':  '(general, no Arena cat)',
    'musr':      '(general, no Arena cat)',
}

# Arena category vote share for the mapped categories
arena_vote_share = {c: v / total_votes for c, v in cat_votes.items()}

print(f"\n{'Benchmark':<12} {'Arena cat':<28} {'Vote share':>11} "
      f"{'Std slope/yr':>13} {'Direction':>12}")
print("-" * 82)
for bm in BENCHMARKS:
    arena_cat = BM_TO_ARENA[bm]
    share = arena_vote_share.get(arena_cat, np.nan)
    share_str = f"{share:.1%}" if not np.isnan(share) else "—"
    slope = conv_rates[bm]
    direction = 'converging ↓' if slope < 0 else 'diverging ↑'
    print(f"{bm:<12} {arena_cat:<28} {share_str:>11} {slope:>+13.2f} {direction:>12}")

# ── 5. The 2025 cross-section: popularity vs. win-rate std ────────────────────
print("\n" + "=" * 65)
print("5. 2025 CROSS-SECTION: popularity vs. win-rate std")
print("=" * 65)

# For the 5 categories with both battle std and arena vote data
joint = []
for cat in ['math', 'coding', 'creative_writing', 'instruction_following']:
    std_val = full_wr[cat].std()
    arena_v = cat_votes.get(cat, np.nan)
    joint.append({'cat': cat, 'wr_std': std_val, 'arena_votes': arena_v,
                  'arena_share': arena_v / total_votes})

joint_df = pd.DataFrame(joint).sort_values('arena_votes', ascending=False)
print(f"\n{'Category':<25} {'Arena vote share':>17} {'Win-rate std':>13} {'Convergence':>12}")
print("-" * 72)
for _, row in joint_df.iterrows():
    conv_label = 'more converged' if row['wr_std'] < joint_df['wr_std'].mean() else 'less converged'
    print(f"{row['cat']:<25} {row['arena_share']:>17.1%} {row['wr_std']:>13.4f} {conv_label:>12}")

# Spearman rho: popularity vs. win-rate std
if len(joint_df) >= 3:
    rho, pval = stats.spearmanr(joint_df['arena_votes'], -joint_df['wr_std'])  # flip std so higher=more convergence
    print(f"\nSpearman rho(vote_share, convergence): ρ={rho:.3f}, p={pval:.3f} (n={len(joint_df)})")
    print("  (Note: n=4, treat qualitatively)")

# ── 6. Confound: task difficulty / benchmark mean score ────────────────────────
print("\n" + "=" * 65)
print("6. CONFOUND: task saturation vs. optimization pressure")
print("=" * 65)

print("\nBenchmark mean score (top-50, 2025) — higher = more saturated:")
for bm in BENCHMARKS:
    sub = bench[bench['year'] == 2025].dropna(subset=['avg_score', bm])
    top50 = sub.nlargest(50, 'avg_score')
    mean_score = top50[bm].mean()
    slope = conv_rates[bm]
    arena_cat = BM_TO_ARENA[bm]
    print(f"  {bm:<12}: mean={mean_score:>6.1f}  conv_slope={slope:>+6.2f}  "
          f"arena_cat={arena_cat}")

print("\nInterpretation:")
print("  Saturation hypothesis: high mean → near-ceiling → less variance → convergence")
print("  Optimization pressure hypothesis: high Arena votes → labs optimize → convergence")
print("  Both predict the same direction; these data cannot distinguish them.")
print()

# Check: correlation between saturation (mean score) and convergence rate
means_2025 = {}
for bm in BENCHMARKS:
    sub = bench[bench['year'] == 2025].dropna(subset=['avg_score', bm])
    top50 = sub.nlargest(50, 'avg_score')
    means_2025[bm] = top50[bm].mean()

slopes = [conv_rates[bm] for bm in BENCHMARKS]
means  = [means_2025[bm] for bm in BENCHMARKS]
rho_sat, p_sat = stats.spearmanr(means, slopes)
print(f"  Spearman rho(mean_score_2025, conv_slope): ρ={rho_sat:.3f}  p={p_sat:.3f}  (n={len(BENCHMARKS)})")
print("  (Negative rho = higher-scoring benchmarks converge more — saturation signal)")
