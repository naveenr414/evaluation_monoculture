"""
C2 Analysis: Does Arena ELO correlate more with instruct benchmark scores
than with the corresponding base model's benchmark scores?

If Arena rewards v (post-training surface features) rather than u (true capability),
we expect: Spearman(ELO, instruct_bench) > Spearman(ELO, base_bench)
And this gap should be larger for newer cohorts.
"""

import pandas as pd
import numpy as np
from scipy import stats
import re

matched  = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
benchmarks = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/benchmarks.parquet')
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])

# ── Step 1: Build base model lookup ──────────────────────────────────────────
# benchmarks has hf_model_id and base_model field
# We need: for each Arena model, find its base model's avg_score

def normalize(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Index full benchmark data by normalized hf_model_id
bench_by_id = {}
for _, row in benchmarks.iterrows():
    if pd.notna(row['hf_model_id']):
        key = normalize(row['hf_model_id'])
        # Keep highest avg_score if duplicates
        if key not in bench_by_id or row['avg_score'] > bench_by_id[key]['avg_score']:
            bench_by_id[key] = row.to_dict()

# Get the first-appearance snapshot for each matched Arena model
first = (matched[matched['avg_score'].notna()]
         .sort_values('leaderboard_publish_date')
         .groupby('model_name').first().reset_index())

# For each Arena model, look up its declared base_model in benchmark data
print("=== Step 1: Matching Arena instruct models to their base models ===\n")

base_matches = []
for _, row in first.iterrows():
    base_hf_id = row.get('base_model')
    if pd.isna(base_hf_id) or not base_hf_id or base_hf_id == row['hf_model_id']:
        continue
    # Skip "Removed" or merge entries
    if 'removed' in str(base_hf_id).lower() or 'merge' in str(base_hf_id).lower():
        continue

    base_key = normalize(base_hf_id)
    base_row = bench_by_id.get(base_key)
    if base_row is None:
        # Try partial match
        for k, v in bench_by_id.items():
            if base_key in k or k in base_key:
                base_row = v
                break

    if base_row and pd.notna(base_row.get('avg_score')):
        base_matches.append({
            'arena_model'      : row['model_name'],
            'hf_model_id'      : row['hf_model_id'],
            'base_hf_id'       : base_hf_id,
            'first_date'       : row['leaderboard_publish_date'],
            'elo'              : row['rating'],
            'instruct_avg'     : row['avg_score'],
            'instruct_ifeval'  : row['ifeval'],
            'instruct_bbh'     : row['bbh'],
            'instruct_math'    : row['math'],
            'instruct_gpqa'    : row['gpqa'],
            'instruct_mmlu_pro': row['mmlu_pro'],
            'base_avg'         : base_row['avg_score'],
            'base_ifeval'      : base_row['ifeval'],
            'base_bbh'         : base_row['bbh'],
            'base_math'        : base_row['math'],
            'base_gpqa'        : base_row['gpqa'],
            'base_mmlu_pro'    : base_row['mmlu_pro'],
            'base_model_type'  : base_row['model_type'],
        })

bdf = pd.DataFrame(base_matches)
print(f"Arena instruct models with base model match: {len(bdf)} / {len(first)}")
print()
print(bdf[['arena_model','base_hf_id','elo','instruct_avg','base_avg']].to_string(index=False))

# ── Step 2: Overall correlation comparison ────────────────────────────────────
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

print(f"\n{'='*65}")
print("Step 2: Overall — ELO vs instruct vs base benchmark scores")
print(f"{'='*65}\n")

rho_inst, lo_i, hi_i, n = spearman_ci(bdf['elo'], bdf['instruct_avg'])
rho_base, lo_b, hi_b, _ = spearman_ci(bdf['elo'], bdf['base_avg'])
print(f"n = {n} models with both instruct and base benchmark scores")
print(f"  ELO vs instruct avg_score : ρ = {rho_inst:.3f}  [{lo_i:.3f}, {hi_i:.3f}]")
print(f"  ELO vs base avg_score     : ρ = {rho_base:.3f}  [{lo_b:.3f}, {hi_b:.3f}]")
print(f"  Δρ (instruct - base)      : {rho_inst - rho_base:.3f}")

print()
print("Per-benchmark breakdown:")
print(f"  {'Benchmark':<12}  {'ρ instruct':>10}  {'ρ base':>10}  {'Δρ':>8}")
print("  " + "-"*45)
for b in ['ifeval','bbh','math','gpqa','mmlu_pro']:
    ri, *_ = spearman_ci(bdf['elo'], bdf[f'instruct_{b}'])
    rb, *_ = spearman_ci(bdf['elo'], bdf[f'base_{b}'])
    print(f"  {b:<12}  {ri:>10.3f}  {rb:>10.3f}  {ri-rb:>8.3f}")

# ── Step 3: By cohort — does the gap grow over time? ─────────────────────────
print(f"\n{'='*65}")
print("Step 3: Cohort analysis — does the instruct-base gap grow?")
print(f"{'='*65}\n")

bdf['cohort'] = pd.cut(
    bdf['first_date'],
    bins=pd.to_datetime(['2023-01-01','2024-01-01','2024-07-01','2025-01-01','2026-01-01']),
    labels=['2023','2024H1','2024H2','2025'],
    right=False
)

print(f"{'Cohort':<10}  {'N':>3}  {'ρ instruct':>10}  {'95% CI':>18}  "
      f"{'ρ base':>8}  {'95% CI':>18}  {'Δρ':>6}")
print("-" * 82)
for cohort in ['2023','2024H1','2024H2','2025']:
    sub = bdf[bdf['cohort'] == cohort]
    ri, li1, hi1, n = spearman_ci(sub['elo'], sub['instruct_avg'])
    rb, lb1, hb1, _ = spearman_ci(sub['elo'], sub['base_avg'])
    if np.isnan(ri):
        print(f"{cohort:<10}  {len(sub):>3}  {'too few':>10}")
        continue
    print(f"{cohort:<10}  {n:>3}  {ri:>10.3f}  [{li1:.3f},{hi1:.3f}]  "
          f"{rb:>8.3f}  [{lb1:.3f},{hb1:.3f}]  {ri-rb:>6.3f}")

# ── Step 4: Model-level: which models show biggest ELO-base divergence? ──────
print(f"\n{'='*65}")
print("Step 4: Model-level — biggest ELO vs base benchmark mismatches")
print(f"{'='*65}\n")

bdf['elo_rank']  = bdf['elo'].rank(ascending=False).astype(int)
bdf['inst_rank'] = bdf['instruct_avg'].rank(ascending=False).astype(int)
bdf['base_rank'] = bdf['base_avg'].rank(ascending=False).astype(int)
bdf['elo_vs_base_diff'] = (bdf['elo_rank'] - bdf['base_rank']).abs()

print("Models where ELO rank diverges most from base model benchmark rank:")
top_div = bdf.nlargest(15, 'elo_vs_base_diff')[
    ['arena_model','elo_rank','inst_rank','base_rank','elo_vs_base_diff',
     'elo','instruct_avg','base_avg','first_date']
]
print(top_div.to_string(index=False))

print(f"\nModels where ELO rank matches base model benchmark rank best:")
low_div = bdf.nsmallest(10, 'elo_vs_base_diff')[
    ['arena_model','elo_rank','inst_rank','base_rank','elo_vs_base_diff',
     'elo','instruct_avg','base_avg']
]
print(low_div.to_string(index=False))
